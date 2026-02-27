import openai
import json
import os
import statistics
import ast
import math
import numpy as npy
from dotenv import load_dotenv
from tqdm import tqdm
import nltk
nltk.download('wordnet')  # This will download the WordNet data
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

def parse_chatgpt_json(response_text):
    try:
        # Attempt to parse the entire response as JSON
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, attempt to extract JSON from markdown or text
        try:
            start_index = response_text.index('{')
            end_index = response_text.rindex('}') + 1
            json_string = response_text[start_index:end_index]
            data = json.loads(json_string)
        except (ValueError, json.JSONDecodeError) as e:
             return f"Error: Could not parse JSON: {e}"
        
    return data

client = openai.AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("API_VERSION")
)

def query(system_prompt, user_prompt, temp=0.3, max_token=512):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        max_tokens=max_token,
        temperature=temp,
    )

    text = response.choices[0].message.content
    return text


def break_sentence(document):
    system_prompt = """
        Break the following document into individual sentences. Ensure that each sentence is complete and stands on its own, ending with the appropriate punctuation (e.g., period, question mark, or exclamation mark). Do not merge or split sentences incorrectly. Return each sentence on a new line.
    """

    prompt = '\n'.join(document)

    output = query(system_prompt, prompt, temp=0.1, max_token=1024)
    return output


def detecting_vague(setence):
    system_prompt = """
    You are an assistant that help detecting the vagueness of a setence. A concept is vague if it admits borderline cases. Another definition for the vagueness is: a concept is vague if it would not be admissible in a law.

    Here is an instruction for detecting a vague in a sentence: 
        - Step 1. Take the longest possible sequence of words that constitutes a noun phrase, including adjectives, verbs, adverbs, prepositions, and nouns, but excluding subordinate phrases. 
        - Step 2. Classifying whether the noun phrase from Step 1 are plural or singular. We are interested here in the syntactic surface form, not the semantic plurality. Thus, the following are all instances: “A committee” (even if it is a group of persons), “a bird”, "the set of integers", “Europe” (even if it a group of countries), and “intelligence” (even if it is a mass noun). “Men”, “soldiers”, and “groups”, in contrast, are singular.
        - Step 3. Classifying each noun phrase in Step 1 into the following top-level classes. The choice depends on the context: “America” can be a place (in sentences where it is the location where something happens), but also an organization (in sentences where it is a state actor). The classes are:  
            1. CreativeWork: anything that can be conceived and created by the human mind, including art works, books, ideas, theories, names, and languages: Mulholland Drive (movie), political philosophy, The radiocarbon work of Jonathan Haas, his campaign slogan, a graphic adventure puzzle video game, Climate engineering techniques.
            2. Action: any activity that can be pursued or engaged in by one or several humans, when it is considered in its generality (independently of a specific instance): smoking, comeback attempts, sexual abuse, the emission of greenhouse gases, geoengineering.
            3. Event: any time period, as well as anything that did, will, or could happen: his death, the last decades, in 2000, the 5th century BCE, the Age of Enlightenment, the early years of the Nazi regime, wildfires, heat waves, the resulting large-scale shifts in weather patterns.
            4. Organization: any group of people that act together under a common name and head for an extended period of time: Sun Records, multiple music halls of fame, a company, countries (if used in the sense of a state actor)
            5. Place: any location, in particular also countries when they are used in a geographical context: various sites, That coastal strip, the lands of Anjou in France
            6. Person: any person or group of persons: Elvis, a commitee
            7. Taxon: any living being except humans: all non-avian dinosaurs, birds
            8. MedicalEntity: any entity that is an object of the field of medicine, including illnesses and parts of the body: nasopharyngeal cancer, the disease
            9. Product: any physical object not yet covered by the above: six million CDs, the food supplies, computers, city trains, an asteroid
            10. Intangible: any non-physical object not yet covered by the above: freezing temperatures, A distinct national identity, Canada's allegiance, The painting's influence, the geological feature.

        - Step 4. Indicates the nature of the head noun and its determiner. It can contain one of the following values: 
            + determined: generally “the XYZ”, or anything that is otherwise uniquely determined. Proper names do not need this value.
            + undetermined: generally “a(n) XYZ”, or plural words without determiner
            + qualified by anaphora: a noun phrase whose determiner is a possessive pronoun or demonstrative: his men, this literature, their start in video game development. However, in "Haas's dates", dates is not qualified by an anaphora.
            + numbered: a noun phrase with a quantity (by a number or adverbial phrases): 80 women, a few occasions, some of the problems, Many different flags, much of this water.
            + anaphora: a reference to a previously mentioned entity (except by full name): she, the volume, the battle
            + named: a proper name, usually with a capitalized head noun: his brother John. This does not apply if the noun phrase merely contains a noun, as in Mongol forces
            + mass noun: anything that cannot be counted without adding a unit: water, addition (some words are mass nouns only in specific contexts)
        - Step 5: Specified how the head noun is modified. Zero or more values are allowed:
            + adjective: the head noun is accompanied by one or several adjectives: the big city
            + preposition: the head noun is accompanied by one or several prepositions: the vote in the city
            + noun: the head noun is accompanied by one or several preceding nouns: vaccination movement
            + contains_name: the noun phrase as a whole contains a named entity: the vote in the US, Robert Kennedy's assassination
        - Step 6: Indicates the type of vagueness. Possible values are:
            + degree (scalar): a threshold (even unknown or relative) can be used to determine if the expression is true. An example is the Heap paradox. Others are: the highland areas of the Andes (highland > T), the first book-length attempt (book-length > T).
            + portions (quantitative): the number, the portion, or the quantity of the entity is unspecified: observers, some scripts, more than 100 people
            + subjective: the concept is vague by nature. It can apply to a certain degree, and there is no consensus on how to measure this degree: The painting's influence, beautiful painting.

        Based on the instruction above, following Step 1 to Step 6 to examines the vagueness in the input sentence belows, and response the result in json format. The json must have these keys for each expression extracted:
        + "expression": contain the noun phrase from Step 1,
        + "plurality": indicate whether the noun phrase is plural or singular from Step 2.
        + "class": hold a single top-level class of the Yago taxonomy from Step 3.
        + "determiner": indicate the nature of the head noun and its determiner from Step 4.
        + "modification": specified how the head noun is modified. Can have multiple value as list from Step 5.
        + "vagueness": an indicator for the type of vagueness from Step 6. If no type indicated, the result is not_vague.
    """

    prompt = '\n'.join(setence)
    # print(setence)

    output = query(system_prompt, prompt, temp=0.0, max_token=128)
    # print(output)
    return output


def detecting_specific(sentence):
    def get_synset_level(word):
        word_synsets = wn.synsets(word)
        if not word_synsets:
            return 0
        # Get the hypernym path for each synset and calculate the depth
        # synset_depths = {}
        # synset_depths = {synset.name(): synset.hypernym_paths() for synset in word_synsets}
        # depth_levels = {synset: len(paths[0]) if paths else 0 for synset, paths in synset_depths.items()}
        # d_level = [v for k, v in depth_levels.items()]
        # # print(depth_levels.items())
        # # print(synset_depths)
        # return sum(d_level)
        return get_specific_level(word_synsets)

    
    def get_specific_level(synsets):
        total = 1
        for syn in synsets:
            total = total + get_specific_level(syn.hyponyms())
        
        return total

    system_prompt = """
    You are an assistant that help breaking a sentence into noun phrases. 
    Take the longest possible sequence of words that constitutes a noun phrase, including adjectives, verbs, adverbs, prepositions, and nouns, but excluding subordinate phrases. 
    
    Return the result as a list of strings, where each element is a noun phrase. Do not include any explanation, just return the list with format look likes ["noun phrase 1", "noun phrase 2", "noun phrase 3"].
    """
    
    prompt = '\n'.join(sentence)
    output = query(system_prompt, prompt, temp=0.0, max_token=256)
    print(sentence)
    print(output)
    output = ast.literal_eval(output)
    
    total_depth = []
    for np in output:
        lst_token = nltk.word_tokenize(np)
        level = [get_synset_level(o) for o in lst_token]
        print(level)
        total_depth.append(sum(level) / len(level))
        # total_depth = total_depth + level
    
    # return total_depth
    # total_depth = [1 / td if td > 0 else 1 for td in total_depth]
    print(total_depth)
    
    # scaler = scaler.fit([specific])

   
    return (1 / statistics.mean(total_depth))



def format_result(response):
    parsed_data = parse_chatgpt_json(response)
    if isinstance(parsed_data, str) and parsed_data.startswith("Error:"):
        print(parsed_data)
        raise Exception
    else:
        return parsed_data


def compute_vagueness(lst_sentences):
    lst_score = []
    lst_result = []
    for s in tqdm(lst_sentences):
        try:
            result = format_result(detecting_vague(s))
            vague = 0
            # print(result)
            # raise Exception
            if len(result) > 0:
                for ex in result:
                    if ex['vagueness'] != 'not_vague':
                        vague = vague + 1
                lst_score.append(float(vague/len(result)))
                lst_result.append(result)
            else:
                if result['vagueness'] != 'not_vague':
                    vague = vague + 1
                lst_score.append(vague)
                lst_result.append(result)
        except Exception as e:
            print(e)
            print(s)
            # raise e
    print(lst_score)
    return statistics.mean(lst_score), lst_result


if __name__ == '__main__':
    LST_TEST = [
        """Alpha Kappa Alpha Sorority, Inc. (ΑΚΑ) is the first intercollegiate historically African-American sorority.[3] The sorority was founded on January 15, 1908, at the historically black Howard University in Washington, D.C., by a group of Howard University students led by Ethel Hedgemon Lyle. Forming a sorority broke barriers for African-American women in areas where they had little power or authority due to a lack of opportunities for Black Americans in the early 20th century.[4] Alpha Kappa Alpha was incorporated on January 29, 1913. 
        """,
        """The image depicts a vibrant and colorful assortment of foods arranged in a blue tray, likely set in an indoor environment such as a kitchen or a dining area. The scene is bathed in natural light, which appears to be coming from a window on the left side, casting a warm and inviting glow across the entire setup.\n\nThe lighting highlights the textures and colors of the food items, making them look appetizing and fresh. The yellow container in the foreground holds a substantial portion of bright green broccoli, which is beautifully lit and appears crisp and vibrant. Next to it, a pink container holds a serving of orange-colored food, possibly a stew or curry, with a rich and warm hue that suggests it has been cooked and is ready to be enjoyed. The pink container also contains a slice of bread with a spread of butter, which glistens slightly under the light.
        """,
        """The image showcases a vibrant and colorful meal presented in a blue lunchbox, divided into three compartments. In the top left compartment, there's a pink container holding a slice of bread with a spread that appears to be butter, accompanied by several almonds. The top right compartment features a similar pink container filled with chunks of pineapple. The bottom compartment, which is yellow, contains a serving of broccoli and a meatball covered in tomato sauce. The meal is visually appealing with its mix of textures and colors, from the creamy butter to the fresh green broccoli and the juicy pineapple. The light source in the image appears to be natural, coming from the upper left, casting soft shadows and highlighting the textures of the food, particularly the glossy surface of the pineapple and the moist appearance of the bread. The overall atmosphere is bright and inviting, making the meal look delicious and well-prepared.
        """,
        """Moreover , if 9GAG or substantially all of its assets were acquired , or in the unlikely event that 9GAG goes out of business or enters bankruptcy , user information would be one of the assets that is transferred or acquired by a third party .""",
        """Alpha Kappa Alpha Sorority, Inc. (ΑΚΑ) is the first historically African American Greek-lettered sorority. The organization was founded on five basic tenets:

        To cultivate and encourage high scholastic and ethical standards, to promote unity and friendship among college women, to study and help alleviate problems concerning girls and women in order to improve their social stature, to maintain a progressive interest in college life, and to be of 'Service to All Mankind'.

        The sorority was founded on January 15, 1908, at the historically black Howard University in Washington, D.C., by a group of sixteen students led by Ethel Hedgeman Lyle. Forming a sorority broke barriers for African-American women in areas where they had little power or authority due to a lack of opportunities for minorities and women in the early 20th century. Alpha Kappa Alpha was incorporated on January 29, 1913.

        The sorority is one of the nation's largest Greek-letter organizations consisting of college-educated women of many diverse backgrounds from around the world, it serves through a membership of more than 300,000 women in 1,024 chapters in the United States and several other countries. Women may join through undergraduate chapters at a college or university or they may also join through a graduate chapter after acquiring an undergraduate or advanced college degree.

        Since the organization's establishment, Alpha Kappa Alpha has helped to improve social and economic conditions through community service programs. Members have improved education through independent initiatives, contributed to community-building by creating programs and associations, such as the Mississippi Health Clinic, and influenced federal legislation by Congressional lobbying through the National Non-Partisan Lobby on Civil and Democratic Rights. The sorority works with communities through service initiatives and progressive programs relating to education, family, health, and business.

        Alpha Kappa Alpha is part of the National Pan-Hellenic Council (NPHC). The current International President is Dr. Glenda Glover and the sorority's document and pictorial archives are located at Moorland-Spingarn Research Center. 
        """,
        """
        In the center of the image, a vibrant blue lunch tray holds four containers, each brimming with a variety of food items. The containers, two in pink and two in yellow, are arranged in a 2x2 grid.\n\nIn the top left pink container, a slice of bread rests, lightly spread with butter and sprinkled with a handful of almonds. The bread is cut into a rectangle, and the almonds are scattered across its buttery surface.\n\nAdjacent to it in the top right corner, another pink container houses a mix of fruit. Sliced apples with their fresh white interiors exposed share the space with juicy chunks of pineapple. The colors of the apple slices and pineapple chunks contrast beautifully against the pink container.\n\nBelow these, in the bottom left corner of the tray, a yellow container holds a single meatball alongside some broccoli. The meatball, round and browned, sits next to the vibrant green broccoli florets.\n\nFinally, in the bottom right yellow container, there's a sweet treat - a chocolate chip cookie. The golden-brown cookie is dotted with chocolate chips, their dark color standing out against the cookie's lighter surface.\n\nThe arrangement of these containers on the blue tray creates a visually appealing and balanced meal, with each component neatly separated yet part of a cohesive whole.
        """,
        """The image shows a variety of foods placed in different containers. There are two bowls, one containing broccoli and another with fruit. The broccoli is located towards the right side of the image, while the fruit bowl is on the left side. In addition to these, there is a sandwich and some cookies spread across the scene. A banana can be seen in the middle of the image, and an apple is placed near the top-left corner. The arrangement of food items suggests that they are ready for consumption or preparation.
        """
    ]
    # # test = break_sentence(LST_TEST[2])
    # # print("------")

    # # lst_sentence = test.split("\n")
    # # print(lst_sentence)
    # # score, lst_result = compute_vagueness(lst_sentence)
    # # print(json.dumps(lst_result, indent=4))
    # # print("=======")
    # # print(score)
    
    # # with open("dump_result_new.json", "w") as f:
    # #     json.dump(lst_result, f, indent=4, ensure_ascii=False)
    # # f.close()

    # # with open("dump_sentence_new.txt", "w") as f:
    # #     for s in lst_sentence:
    # #         f.write(s + "\n")
    # # f.close()

    # # # print(lst_sentence)
    # result = detecting_vague("The picture shows a sedan.")
    # list_data = format_result(result)

    # print(json.dumps(list_data, indent=4))

    TEST_SENTENCE = [
        "The picture shows a meal.",
        "The picture shows a thing.",
        "The picture shows a white car.",
        "The picture shows a white sedan.",
        "In the center of the image, a vibrant blue lunch tray holds four containers, each brimming with a variety of food items.",
        "The image shows a variety of foods placed in different containers",
        "There are two bowls, one containing broccoli and another with fruit",
        "a white castle",
        "a white marble structure",
        "the picture shows a man standing in front of a micro",
        "the picture shows President of USA speaking in public",
        "the picture shows Joe Biden speaking in public",
        "The broccoli is located towards the right side of the image, while the fruit bowl is on the left side",
        "In addition to these, there is a sandwich and some cookies spread across the scene. A banana can be seen in the middle of the image, and an apple is placed near the top-left corner."
    ]
    # output = detecting_specific(TEST_SENTENCE[10])
    # output = detecting_specific("There are some eggs on the table")
    # print(output)
    # print(statistics.mean(output))

    # print(detecting_specific("The picture shows a meal"))
    # print("--------")
    # print(detecting_specific("The picture shows a thing"))
    # print("--------")
    # print(detecting_specific("The picture shows a white car"))
    # print("--------")
    # print(detecting_specific("The picture shows a white sedan"))
    # print("--------")
    # print(detecting_specific("The picture shows a man standing in front of a microphone."))
    # print("--------")
    # print(detecting_specific("The picture shows President of USA speaking in public."))
    # print("--------")
    # print(detecting_specific("The picture shows Joe Biden speaking in public."))
    # print("--------")
    # print(detecting_specific("The picture shows 4 eggs."))
    # print("--------")
    # print(detecting_specific("The picture shows somes eggs."))
    # print("--------")
    print(detecting_specific("The picture shows a white Camry SE"))
    print("--------")
