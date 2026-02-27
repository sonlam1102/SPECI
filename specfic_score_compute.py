import ast
from nltk.corpus import wordnet as wn
from misc_func import *
import statistics
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time


def break_sentence(document):
    system_prompt = """
        Break the following document into individual sentences. Ensure that each sentence is complete and stands on its own, ending with the appropriate punctuation (e.g., period, question mark, or exclamation mark). Do not merge or split sentences incorrectly. Return each sentence on a new line.
    """

    prompt = '\n'.join(document)

    output = query(system_prompt, prompt, temp=0.0, max_token=512)
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
    output = query(system_prompt, prompt, temp=0.1, max_token=2048)
    
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
        return len(get_specific_level(word_synsets))

    
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


def average_specific_score(text, verbose=False):
    spec_sent = specific_score(text, verbose)
    spec_visual = specific_score_new(text, verbose)

    score_avg = 0.86044898*spec_sent + 0.06385769*spec_visual[1]['Visual Detail'] + -0.01287558*spec_visual[1]['Spatial Specificity'] + -0.05029409*spec_visual[1]['Color & Texture'] +  0.11765478*spec_visual[1]['Interpretive Language']
    return score_avg

    

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
        results[k] = tmp_score
        scores.append(tmp_score)
    
    scores = [sc + 1 for sc in scores]
    scores.sort(reverse=True)
    weight = [k for k in reversed(range(1, len(scores)+1))]
    new_score = [scores[i]*weight[i] for i in range(0, len(scores))]
    final_score = statistics.harmonic_mean(new_score)
    return final_score, results, visual_component
      

def compute_specific_score(text):
    spec_score = average_specific_score(text, verbose=False)
    new_score = spec_score  / 10
    
    return new_score
    
if __name__ == '__main__':
    A = "A brown dog, likely a working breed, is captured mid-leap over a textured agility ramp, showcasing its muscular build and focused expression. The dog's mouth is wide open in a bark, revealing its teeth, which adds a sense of excitement to the scene. The ramp, covered in a sandy material, is positioned on a grassy field, suggesting an outdoor training or competition setting. In the background, blurred figures of spectators can be seen, some wearing casual clothing, indicating a lively atmosphere. The bright sunlight enhances the vivid colors of the scene, highlighting the dog's shiny coat and the green grass beneath."
    
    detecting_specific_phrase("brown dog", verbose=True)
    