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
from scipy.special import expit, logit
import stanza
import time
# import google.generativeai as genai, types
from google import genai

load_dotenv()
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

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

client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) 

def query(system_prompt, user_prompt, temp=0.3, max_token=512):
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=messages,
        max_completion_tokens=max_token,
        temperature=temp,
    )
    # print(response)
    text = response.choices[0].message.content
    return text


def query2(user_prompt, temp=0.3, max_token=512):
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=messages,
        max_completion_tokens=max_token,
        temperature=temp,
    )
    # print(response)
    text = response.choices[0].message.content
    return text

def query3(user_prompt):
    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash", 
        contents=user_prompt
    )
    return response.text

def break_sentence(document):
    system_prompt = """
        Break the following document into individual sentences. Ensure that each sentence is complete and stands on its own, ending with the appropriate punctuation (e.g., period, question mark, or exclamation mark). Do not merge or split sentences incorrectly. Return each sentence on a new line.
    """

    prompt = '\n'.join(document)

    output = query(system_prompt, prompt, temp=0.0, max_token=512)
    return output


def extract_relation(document, verbose=True):
    print(document) if verbose else None
    # system_prompt = """
    # Instruction:
    # Given the paragraph below, extract a list of all distinct relationships between entities in the form of (subject; relation; object) triples. Each triple should represent a factual relation. Ignore non-informative or purely grammatical triples. Return only the list of relations—no explanations, no numbering, and no extra text.

    # Input Paragraph:
    # Paste your paragraph here.

    # Output Format (List of Relations):
    #     (Subject; Relation; Object)
    #     (Subject; Relation; Object)
    #     …
    # Example:
    # Input:
    # "Apple Inc., founded by Steve Jobs, is headquartered in Cupertino. Tim Cook leads the company."

    # Output:
    #     (Apple Inc.; was founded by; Steve Jobs)
    #     (Apple Inc.; is headquartered in; Cupertino)
    #     (Tim Cook; leads; Apple Inc.)
    # """
    # system_prompt = """
    # You are an advanced information extraction system. Given a document of text, identify all meaningful relationships between entities mentioned in the document.

    # For each relation, return it in the following format: (entity1; relation; entity2)
    # Return only the list of relations—no explanations, no numbering, and no extra text.
    # """
    
    system_prompt = """
    Extract all possible semantic relations between entities in the following paragraph.
    Return each relation in the format:
        (subject; relation; object)
    Do not include explanations or any additional text—only the list of relations.
    """
    
    prompt = '\n'.join(document)
    output = query(system_prompt, prompt, temp=0.2, max_token=512)
    print(output) if verbose else None
    relations = []
    try:
        new_out = output.split("\n")
        # relations = []
        for no in new_out:
            no = no.replace("(", "")
            no = no.replace(")", "")
            tmp = no.split(";")
            # assert len(tmp) > 1
            if len(tmp) > 1:
                relations.append([to.strip() for to in tmp])
        return relations
    except Exception as e:
        print("extract rel error, bypass")
    return relations
    # if output == "EMPTY":
    #     return []
    # new_out = output.split("\n")
    # relations = []
    # for no in new_out:
    #     no = no.replace("(", "")
    #     no = no.replace(")", "")
    #     tmp = no.split(";")
    #     assert len(tmp) == 3
    #     relations.append(tmp[0].strip()) if tmp[0].strip() != "EMPTY" else 1
    #     relations.append(tmp[2].strip()) if tmp[2].strip() != "EMPTY" else 1
    
    # return list(set(relations))


def extract_noun_phrase(sentence):
    system_prompt = """
    You are an assistant that help breaking a sentence into noun phrases. 
    Take the longest possible sequence of words that constitutes a noun phrase, including adjectives, verbs, adverbs, prepositions, and nouns, but excluding subordinate phrases. 
    
    Return the result as a list of strings, where each element is a noun phrase. Do not include any explanation, just return the list with format look likes ["noun phrase 1", "noun phrase 2", "noun phrase 3"].
    """
    
    prompt = '\n'.join(sentence)
    output = query(system_prompt, prompt, temp=0.1, max_token=1024)
    
    try:
        output = ast.literal_eval(output)
    except Exception as e:
        print(output)
        raise e
    
    return output

def extract_atomic_fact(paragraph):
    system_prompt = """
    Break down the following paragraph into a list of atomic facts. Each atomic fact must:
    - Contain only one piece of information.
    - Be clear, specific, and verifiable.
    - Stand alone without relying on other facts.
    - Avoid assumptions, opinions, or interpretations.
    
    Return results as a list of strings such that each element is an atomic fact. Do not include any explanation, just return the list with format look likes ["atomic fact 1", "atomic fact 2", "atomic fact 3"].
    """
    
    prompt = paragraph
    output = query(system_prompt, prompt, temp=0.0, max_token=2048)
    
    try:
        output = ast.literal_eval(output)
    except Exception as e:
        print(output)
        raise e
    
    return output
    

def detecting_specific(sentence, verbose=True):
    print(sentence) if verbose else None
    def get_synset_level(word):
        if word.ner != 'O':
            return 0
        word_synsets = wn.synsets(word.text)
        if not word_synsets:
            return 0
        # return get_specific_level(word_synsets)
        return len(get_specific_level(word_synsets))

    
    # def get_specific_level(synsets):
    #     total = 1
    #     # if len(synsets) == 0 or total > 1: # avoid max recursion depth
    #     #     return total
        
    #     if len(synsets) == 0 or total > 500: # avoid max recursion depth
    #         return 1
        
    #     for syn in synsets:
    #         total = total + get_specific_level(syn.hyponyms())
        
    #     return total
    
    # def get_specific_level(synsets):
    #     total = len(synsets)

    #     if total < 1:
    #         return total
        
    #     for syn in synsets:
    #         total = total + get_specific_level(syn.hyponyms())
        
    #     return total
    
    def get_specific_level(synset, k = 0):
        vehicle_words = set()
        # Add the lemmas of the current synset to the set
        # print(synset)
        for lemma in synset:
            vehicle_words.add(lemma)
        # Recursively add the lemmas of the hyponyms
        for hyponym in synset:
            if k > 500:
                break
            k = k + 1
            vehicle_words.update(get_specific_level(hyponym.hyponyms(), k))
        return vehicle_words

    # output = extract_noun_phrase(sentence)
    output = [sentence]
    
    total_depth = []
    for phrase in output:
        # lst_token = nltk.word_tokenize(np)
        print(phrase) if verbose else None
        lst_token = nlp(phrase).sentences[0].tokens
        print("Length of phrase: {}".format(len(lst_token))) if verbose else None
        level = [get_synset_level(o) for o in lst_token]
        print("Level of hyponym: {}\n".format(level)) if verbose else None
        # total_depth.append((len(level) / sum(level)) + ((sum(level)/len(level))))
        alpha = sum(level)
        beta = ((1 / alpha) if alpha > 0 else 1) + (len(level) / (max(level) if max(level) > 0 else 1))
        total_depth.append(beta)
        # total_depth = total_depth + level
    
    # return total_depth
    # total_depth = [1 / td if td > 0 else 1 for td in total_depth]
    # total_depth.sort(reverse=True)
    # weight = [k for k in reversed(range(1, len(total_depth)+1))]
    # new_total_depth = [total_depth[i]*weight[i] for i in range(0, len(total_depth))]

    print("Total hyponyms: {}\n".format(total_depth)) if verbose else None
    return statistics.mean(total_depth)


def detecting_specific_phrase(phrase, verbose=True):
    print(phrase) if verbose else None
    def get_synset_level(word):
        if word.ner != 'O':
            return 0
        word_synsets = wn.synsets(word.text)
        if not word_synsets:
            return 0
        # return get_specific_level(word_synsets)
        return len(get_specific_level(word_synsets))

    
    # def get_specific_level(synsets):
    #     total = 1
    #     if len(synsets) == 0 or total > 500: # avoid max recursion depth
    #         return 1
        
    #     for syn in synsets:
    #         total = total + get_specific_level(syn.hyponyms())
        
    #     return total
    
    # def get_specific_level(synsets):
    #     total = len(synsets)

    #     if total < 1:
    #         return total
        
    #     for syn in synsets:
    #         total = total + get_specific_level(syn.hyponyms())
        
    #     return total
    
    def get_specific_level(synset, k = 0):
        vehicle_words = set()
        # Add the lemmas of the current synset to the set
        # print(synset)
        for lemma in synset:
            vehicle_words.add(lemma)
        # Recursively add the lemmas of the hyponyms
        for hyponym in synset:
            if k > 500:
                break
            k = k + 1
            vehicle_words.update(get_specific_level(hyponym.hyponyms(), k))
        return vehicle_words

    output = nlp(phrase).sentences[0].tokens
    print("Length of phrase: {}".format(len(output))) if verbose else None
    level = [get_synset_level(o) for o in output]
    print("Level of hyponym: {}\n".format(level)) if verbose else None
    alpha = sum(level)
    beta = ((1 / alpha) if alpha > 0 else 1) + (len(level) / (max(level) if max(level) > 0 else 1))
    print(beta) if verbose else None
    return beta
    
    # return statistics.mean(level)


def detecting_specific_relation(relation, verbose=True):
    def get_synset_level(word):
        if word.ner != 'O':
            return 0
        word_synsets = wn.synsets(word.text)
        if not word_synsets:
            return 0
        return get_specific_level(word_synsets)

    
    # def get_specific_level(synsets):
    #     total = 1
    #     # if len(synsets) == 0 or total > 1: # avoid max recursion depth
    #     #     return total
        
    #     if len(synsets) == 0 or total > 500: # avoid max recursion depth
    #         return 1
        
    #     for syn in synsets:
    #         total = total + get_specific_level(syn.hyponyms())
        
    #     return total
    
    def get_specific_level(synsets):
        total = len(synsets)

        if total < 1:
            return total
        
        for syn in synsets:
            total = total + get_specific_level(syn.hyponyms())
        
        return total
        
    total_depth = []
    print(relation) if verbose else None
    for phrase in relation:
        # lst_token = nltk.word_tokenize(np)
        lst_token = nlp(phrase).sentences[0].tokens
        print("Length of phrase: {}".format(len(lst_token))) if verbose else None
        level = [get_synset_level(o) for o in lst_token]
        print("Level of hyponym: {}\n".format(level)) if verbose else None
        # total_depth.append((len(level) / sum(level)) + ((sum(level)/len(level))))
        alpha = sum(level)
        beta = 0.5*((1 / alpha) if alpha > 0 else 1) + 0.5*(len(level) / (max(level) if max(level) > 0 else 1))
        total_depth.append(beta)
        # total_depth = total_depth + level
    
    # return total_depth
    # total_depth = [1 / td if td > 0 else 1 for td in total_depth]
    # total_depth.sort(reverse=True)
    # weight = [k for k in reversed(range(1, len(total_depth)+1))]
    # new_total_depth = [total_depth[i]*weight[i] for i in range(0, len(total_depth))]

    print("Total hyponyms: {}\n".format(total_depth)) if verbose else None
    return statistics.mean(total_depth)


def specific_score(text, verbose=True, num_try_args=0):
    num_try = num_try_args
    try:
        # lst_sentence = break_sentence(text).split("\n")
        lst_sentence = extract_atomic_fact(text)
        print(lst_sentence) if verbose else None
        new_score = 0
        
        if len(lst_sentence) > 0:
            new_lst_sentence = []
            for se in lst_sentence:
                if se.strip() != "":
                    new_lst_sentence.append(se)
            
            scores = []
            for sen in new_lst_sentence:
                scores.append(detecting_specific(sen, verbose))
            
            scores.sort(reverse=True)
            weight = [k for k in reversed(range(1, len(scores)+1))]
            new_score = [scores[i]*weight[i] for i in range(0, len(scores))]

        return statistics.mean(new_score)
    except Exception as e:
        num_try = num_try + 1
        print(e)
        print("error extract phrase, try {} times".format(num_try))
        time.sleep(2) if num_try > 3 else None
        return 0 if num_try > 3 else specific_score(text, verbose=verbose, num_try_args=num_try)


def specific_score2(text, verbose=True, num_try_args=0):
    num_try = num_try_args
    def trace(mat):
        temp = 0
        for i in range(0, len(mat)):
            temp = temp + mat[i][i]
        
        return temp
    
    def count_triangle_upper(mat):
        temp = 0
        k = 0
        for i in range(0, len(mat)-1):
            temp = temp + sum(mat[i][i+1:len(mat)])
            k = k + 1
                
        return temp, k
        
    
    def count_triangle_lower(mat):
        temp = 0
        k = 0
        for i in range(1, len(mat)):
            temp = temp + sum(mat[i][0:i-1])
            k = k + 1
            
        return temp, k
    
    def find_entity(ent, lst_ent):
        for i in range(0, len(lst_ent)):
            if ent == lst_ent[i]:
                return i
        return -1
    
    try:
        lst_rel = extract_relation(text, verbose=verbose)
        lst_objs = []
        for r in lst_rel:
            if len(r) > 1:
                lst_objs.append(r[0])
                lst_objs.append(r[2])
            
        lst_objs = list(set(lst_objs))
        print(lst_objs) if verbose else None
        mat = []
        for o in lst_objs:
            mat.append([0 for i in range(0, len(lst_objs))])
        
        for i in range(0, len(lst_objs)):
            mat[i][i] = detecting_specific_phrase(lst_objs[i], verbose=verbose)
        
        for r in lst_rel:
            if len(r) > 1:
                ind_en1 = find_entity(r[0], lst_objs)
                ind_en2 = find_entity(r[2], lst_objs)
                # assert ind_en1 > -1 and ind_en2 > -1
                
                spec_rel = detecting_specific_phrase(r[1], verbose=verbose)
                mat[ind_en1][ind_en2] = spec_rel
        
        if verbose:
            for row in mat:
                print(row)
                print("\n")
        
        trace_mat = trace(mat)
        up_mat = count_triangle_upper(mat)
        low_mat = count_triangle_lower(mat)
        
        # print(up_mat[1])
        # print(low_mat[1])
        assert up_mat[1] == low_mat[1]
        
        if verbose:
            print("Trace: {}".format(trace_mat))
            print("Upper triangle: {}".format(up_mat[0])) 
            print("Lower triangle: {}".format(low_mat[0]))   
        
        return trace_mat / len(lst_objs) if len(lst_objs) > 0 else 0 + (up_mat[0] + low_mat[0]) / 2
        # return trace_mat / len(lst_objs) if len(lst_objs) > 0 else 0 + (up_mat[0] + low_mat[0]) / up_mat[1] if up_mat[1] > 0 else 0
    except Exception as e:
        num_try = num_try + 1
        # raise e
        print(e)
        print("error extract relation, try {} times".format(num_try))
        time.sleep(2) if num_try > 3 else None
        return 0 if num_try > 3 else specific_score2(text, verbose=verbose, num_try_args=num_try)


def average_specific_score(text, verbose=False):
    spec_sent = specific_score(text, verbose)
    spec_visual = specific_score_new(text, verbose)
    
    # score_avg = 0.67851031*spec_sent + 0.53123079*spec_visual
    # score_avg = 0.72356658*spec_sent + 0.32411562*spec_visual
    # score_avg = 0.76756938*spec_sent + 0.23584852*spec_visual[0]
    
    
    # score_avg = 0.86044898*spec_sent + 0.06385769*spec_visual[1]['Visual Detail'] + -0.01287558*spec_visual[1]['Spatial Specificity'] + -0.05029409*spec_visual[1]['Color & Texture'] +  0.11765478*spec_visual[1]['Interpretive Language']
    
    # total 
    # score_avg = 0.86044898*spec_sent + 0.06385769*spec_visual[1]['Visual Detail'] + -0.05029409*spec_visual[1]['Spatial Specificity'] +  0.11765478*spec_visual[1]['Interpretive Language']
    score_avg = 0.85398681*spec_sent + 0.06280092*spec_visual[1]['Visual Detail'] + -0.01397166*spec_visual[1]['Spatial Specificity'] +  0.1165257*spec_visual[1]['Interpretive Language']
    
    # # dci 
    # score_avg = 1.02437286*spec_sent + 0.0469331*spec_visual[1]['Visual Detail'] + -0.13088941*spec_visual[1]['Spatial Specificity'] + 0.07826575*spec_visual[1]['Color & Texture'] +  0.21119611*spec_visual[1]['Interpretive Language']
    
    # # docci
    # score_avg = 1.27199533*spec_sent + 0.00612458*spec_visual[1]['Visual Detail'] + 0.03703116*spec_visual[1]['Spatial Specificity'] + -0.0812842*spec_visual[1]['Color & Texture'] +  -0.03702984*spec_visual[1]['Interpretive Language']
    
    # iiw
    # score_avg = 0.45468603*spec_sent + 0.12481487*spec_visual[1]['Visual Detail'] + 0.04091814*spec_visual[1]['Spatial Specificity'] + -0.20505159*spec_visual[1]['Color & Texture'] +  0.11052564*spec_visual[1]['Interpretive Language']
   
    # score_avg = spec_visual
    return score_avg


def spec_score_gpt4(text, verbose=False):
    instruction = """
        You are given a caption that describes an image or scene at the paragraph level.
        Assess its specificity on a scale from 1 to 5, where:

        1 = extremely vague and generic (very few concrete details)  
        2 = somewhat vague (some details, but still general)  
        3 = moderately specific (clear but not richly detailed)  
        4 = specific (contains several concrete, precise details)  
        5 = highly specific (rich, precise, and concrete in all aspects)

        Consider:
        - Level of visual detail
        - Spatial and compositional clarity
        - Color and texture precision
        - Linguistic focus (descriptive precision vs. generic/interpretive wording)

        Return only the integer score from 1 to 5 with no explanation.
    """
    
    output = query(instruction, text, temp=0, max_token=5)
    
    return output
    
    
def spec_score_gemini(text, verbose=False):
    instruction = f"""
        You are given a caption that describes an image or scene at the paragraph level.
        Assess its specificity on a scale from 1 to 5, where:

        1 = extremely vague and generic (very few concrete details)  
        2 = somewhat vague (some details, but still general)  
        3 = moderately specific (clear but not richly detailed)  
        4 = specific (contains several concrete, precise details)  
        5 = highly specific (rich, precise, and concrete in all aspects)

        Consider:
        - Level of visual detail
        - Spatial and compositional clarity
        - Color and texture precision
        - Linguistic focus (descriptive precision vs. generic/interpretive wording)

        Return only the integer score from 1 to 5 with no explanation.
        The caption: {text}
    """
    
    output = query3(instruction)
    
    return output
    

def compare_specific_gpt4(sentence1, sentence2):
    instruction = """Given two captions, caption_1 and caption_2, compare their specificity. 
        Return one integer based on the following scale:

        2 if caption_1 is substantially more specific than caption_2  
        1 if caption_1 is marginally more specific than caption_2  
        0 if both are equally specific  
        -1 if caption_1 is marginally less specific than caption_2  
        -2 if caption_1 is substantially less specific than caption_2  

        Return only the score, no explanation.
    """
    
    prompt = f"""Compare the specificity of the two captions below:
        caption_1: {sentence1}  
        caption_2: {sentence2}
    """
    output = query(instruction, prompt, temp=0.1, max_token=10)
    
    return output


def compare_hallucination_gpt4(sentence1, sentence2):
    instruction = """Given two captions, caption_1 and caption_2, compare their hallucination. 
        Return one integer based on the following scale:
        
        2 if caption_1 is substantially more hallucinatory than caption_2  
        1 if caption_1 is marginally more hallucinatory than caption_2  
        0 if both are equally hallucinatory  
        -1 if caption_1 is marginally less hallucinatory than caption_2  
        -2 if caption_1 is substantially less hallucinatory than caption_2  

        Return only the score, no explanation.
    """
    
    prompt = f"""Compare the specificity of the two captions below:
        caption_1: {sentence1}  
        caption_2: {sentence2}
    """
    output = query(instruction, prompt, temp=0.1, max_token=10)
    
    return output


def compare_hallucination_gemini(sentence1, sentence2):
    instruction = f"""Given two captions, caption_1 and caption_2, compare their hallucination. 
        Return one integer based on the following scale:
        
        2 if caption_1 is substantially more hallucinatory than caption_2  
        1 if caption_1 is marginally more hallucinatory than caption_2  
        0 if both are equally hallucinatory  
        -1 if caption_1 is marginally less hallucinatory than caption_2  
        -2 if caption_1 is substantially less hallucinatory than caption_2  
        
        Based on the instruction belows, let's compare the specificity of the two captions below:
        caption_1: {sentence1}.  
        caption_2: {sentence2}.
        
        Return only the score, no explanation. If cannot determine, please choose your most approriate score. 
    """
    output = query3(instruction)
    
    return output


def extract_visual_phrase(paragraph):
    instruction_prompt = f"""
        You are given a paragraph that describes an image (caption).  
        Your task is to extract and categorize key **phrases** from the paragraph under the following four criteria:

        1. Level of Visual Detail (Granularity): Extract phrases that describe specific object features, such as size, shape, quantity, material, pattern, or structural composition.

        2. Spatial & Compositional Specificity: Extract phrases that describe spatial relationships, such as position in the frame, order of elements, perspective (e.g., top/bottom, left/right), or how elements are arranged in space.

        3. Color & Texture Specificity: Extract phrases that describe precise color names, combinations, shades, or texture-related descriptors.

        4. Linguistic Focus (Descriptive vs. Interpretive): Extract interpretive, subjective, or stylistic phrases (e.g., “visually appealing”, “striking contrast”, “reminiscent of”, or metaphorical language) that go beyond objective visual description.

        Caption:
        \"\"\"{paragraph.strip()}\"\"\"

        Return the result as a Python dictionary like this:
        {{
        "Visual Detail": [...],
        "Spatial Specificity": [...],
        "Color & Texture": [...],
        "Interpretive Language": [...]
        }}
    """
        
    output = query2(instruction_prompt, temp=0.1, max_token=2048)
    output = output.replace("```python", "")
    output = output.replace("```", "")
    return output


def specific_score_new(paragraph, verbose=False):
    visual_component = json.loads(extract_visual_phrase(paragraph))
    scores = []
    results = {}
    for k, v in visual_component.items():
        lst_comp_scores = [detecting_specific_phrase(v_i, verbose) for v_i in v] if len(v) > 0 else [0]
        tmp_score = sum(lst_comp_scores)
        # tmp_score = statistics.mean(lst_comp_scores)
        results[k] = tmp_score
        # if len(v) > 0:
        scores.append(tmp_score)
    
    # print(visual_component)
    # # print(scores)
    # print("-----")
    # print(results)
    scores = [sc + 1 for sc in scores]
    # print(scores)
    scores.sort(reverse=True)
    weight = [k for k in reversed(range(1, len(scores)+1))]
    new_score = [scores[i]*weight[i] for i in range(0, len(scores))]
    # return statistics.harmonic_mean(scores, weights=[2,4,1,3]) / len(scores), results, visual_component
    # final_score = statistics.harmonic_mean(new_score)
    final_score = statistics.harmonic_mean(new_score)
    return final_score, results, visual_component
      

if __name__ == '__main__':
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
        "the picture shows a man standing in front of a microphone",
        "the picture shows President of USA speaking in public",
        "the picture shows Joe Biden speaking in public",
        "The broccoli is located towards the right side of the image, while the fruit bowl is on the left side",
        "In addition to these, there is a sandwich and some cookies spread across the scene. A banana can be seen in the middle of the image, and an apple is placed near the top-left corner."
    ]
    # output = detecting_specific(TEST_SENTENCE[10])
    # print(output)
    # output = detecting_specific(TEST_SENTENCE[11])
    # print(output)

    # output = detecting_specific(TEST_SENTENCE[-1])
    # print(output)
    # output = detecting_specific("There are some eggs on the table")
    # print(output)
    # print(statistics.mean(output))

    # print(detecting_specific("The picture shows a man standing in front of a microphone"))
    # print("--------")
    # print(detecting_specific("The picture shows President of USA speaking in public"))
    # print("--------")
    # print(detecting_specific("The picture shows Donald Trump speaking in public"))
    # print("--------")
    # print(detecting_specific("The picture shows a white car"))
    # print("--------")
    # print(detecting_specific("The picture shows a white sedan"))
    # print("--------")
    # print(detecting_specific("The picture shows 4 eggs."))
    # print("--------")
    # print(detecting_specific("The picture shows somes eggs."))
    # print("--------")
    # print(detecting_specific("The picture shows a white Honda"))
    # print("--------")
    # print(detecting_specific("The picture shows a white Honda CRV"))
    # print("--------")
    # # print(detecting_specific("The picture shows a car"))
    # # print("--------")
    # # print(detecting_specific("The picture shows a white car"))
    # # print("--------")
    # print(detecting_specific("The picture shows a thing"))
    # print("--------")
    # print(detecting_specific("The picture shows a meal"))
    # print("--------")

    # print(detecting_specific("The picture shows a car"))
    # print("--------")
    
    # print(detecting_specific("The picture shows a white car"))
    # print("--------")

    # print(detecting_specific("The picture shows a yellow car"))
    # print("--------")

    # print(detecting_specific("The picture shows a blueberry car"))
    # print("--------")

    
    # print(average_specific_score("A view of a plastic swan that is next to a pile of rocks the pile of rocks are standing on top of a brown pool there is a light reflecting off the bottom of the pool", verbose=False))
    # print("------")
    # print(average_specific_score("One soccer team wearing red jerseys and another with white jersey are walking around on a soccer pitch.  There is a large metal structure behind them.", verbose=False))
    # print("========")
    
    # print(average_specific_score("A close up of a phone and a computer screen that are both watching stocks or coin. On the phone there is a zoom in on one specific  stock there also. Everything is very blurry besides the focus of the phone which is on a stock that says EOS. The top right of the phone has a lot of widgets or notifications visible. The screen the the background whos two different formats for the stock chart and a standard chart and a bar chart on the bottom and there are three colors used here bright red green and bright blue.", verbose=True))
    
    A = "In a low-angle shot, a wall decorated with a pattern of square tiles stands below a wide blue sky with thick, fluffy white and gray cumulus clouds scattered throughout it. The top of the wall is a solid slab, and below it are square beige tiles arranged in a grid of two rows. Below that is a segment of darker-colored tan square tiles about a quarter the size of the beige tiles, with a pattern of dark brown tiles forming plus-sign shapes in two alternating staggered rows. Below this segment, another row of the larger beige tiles is visible, as well as the top of a second row which continues below the frame. The sky fills up roughly the top two-thirds of the frame, and is a dark blue with large clouds that are white with darker gray undersides."
    # B = "Captured from a low-angle perspective, a wall adorned with rectangular tiles in varying shades of brown, white, and blue forms a striking contrast against the backdrop of a cerulean sky dotted with fluffy white clouds. The tiles, arranged in a grid pattern reminiscent of a checkerboard, are meticulously arranged in a horizontal row across the wall. The bottom row of tiles is a blend of brown and white, while the top row showcases a mix of brown and white, creating a visually appealing pattern."
    
    # C = "In a very, very dark black and white medium shot, the viewer looks out through two dark tall parallel windows to a dark greenish-black forest beyond. The window on the left, taking up the entire left half of the photo, is framed with dark black rectangular frames. The top half of the window is somewhat lighter, with the heavy dark green leafy trees more visible. The lower half of the left window is quite dark, and only a dark horizontal line can be discerned where the dark greenish-black lawn touches the completely black base of the trees. The window on the right is partly obscured by a dark black plant with long, narrow, pointed leaves jutting upward at irregular angles. The top two leaf tips extend above the middle frame bar of this right window. Again, the top half of the window shows more light on the far dark trees and even reveals two tiny bits of dark grey overcast sky. The lower half of the right window is so dark that essentially nothing but the plant leaves can be seen."
    
    # D = "Captured from a low-angle perspective, a window reveals a dimly lit outdoor scene, punctuated by a solitary plant positioned in the lower right corner. The window frame is shrouded in darkness, casting a stark silhouette of the plant against the dark backdrop. The plant, with its elongated, narrow leaves, stands out against the surrounding darkness. The background is dominated by a dense congregation of trees, their leaves a blend of light and dark shades. The sky, visible through the window, is a canvas of dark clouds, adding a layer of mystery to the scene."
    
    # print(json.loads(extract_visual_phrase(A)))
    # print(json.loads(extract_visual_phrase(B)))
    
    # print(specific_score_new(A, verbose=False)[0])
    # print(specific_score_new(B, verbose=False)[0])
    # print(specific_score_new(C, verbose=False)[0])
    # print(specific_score_new(D, verbose=False)[0])
    # print("----")
    # print(average_specific_score(A, verbose=False))
    # print(average_specific_score(B, verbose=False))
    # print(average_specific_score(C, verbose=False))
    # print(average_specific_score(D, verbose=False))
    
    
    # print(specific_score_new(A, verbose=False)[0])
    # print(json.dumps(specific_score_new(A, verbose=False)[1], indent=4))
    # print(json.dumps(specific_score_new(A, verbose=False)[2], indent=4))
    # print("-------")
    # print(specific_score_new(B, verbose=False)[0])
    # print(json.dumps(specific_score_new(B, verbose=False)[1], indent=4))
    # print(json.dumps(specific_score_new(B, verbose=False)[2], indent=4))
    
    # print(spec_score_gpt4(C))

    
    # print(detecting_specific_phrase("a Sedan white car", verbose=False))
    # print(detecting_specific_phrase("a car", verbose=False))
    # print(detecting_specific_phrase("a Sedan purple car", verbose=False))
    
    # print(specific_score_new(A, verbose=False)[1])
    # print("-------")
    # print(specific_score(A, verbose=False))
    # print("-------")
    # print(detecting_specific("The picture shows a white car", verbose=True))
    # print("-------")
    # print(detecting_specific("The picture shows a white sedan", verbose=True))
    
    print(spec_score_gemini(A))