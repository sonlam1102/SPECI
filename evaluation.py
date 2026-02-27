import json
from specific_score import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from tqdm import tqdm 
import time
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
# from inference import predict_specific as fine_tune_predict_specific
# from inference import predict_hallucination as fine_tune_predict_hallucination
import pandas as pd


def round_up_to_half(number):
    return math.ceil(number * 2) / 2

def normalized_score(number):
    if -2 <= number < -1.5:
        return -2
    elif -1.5 <= number <= -1:
        return -1
    elif -1 < number <= -0.5:
        return 0
    elif -0.5 < number <= 0:
        return 0
    elif 0 < number <= 1.5:
        return 1
    else:
        return 2

def compute_iiw():
    data = []
    with open("./dataset/iiw_new.jsonl", 'r') as f:
        
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()

    results = []
    indx = 0
    for d in tqdm(data):
        try:
            text1 = d['IIW']
            text2 = d['IIW-P5B']

            human = d['iiw-human-sxs-iiw-p5b']['metrics/Specificity']
            score_iiw = average_specific_score(text1, verbose=False)
            score_p5b = average_specific_score(text2, verbose=False)

            results.append({
                "IIW": text1,
                "IIW-P5B": text2,
                "score_iiw": score_iiw,
                "score_p5b": score_p5b,
                "image": d["image/key"],
                "human_specific": human,
            })
        except Exception as e:
            # raise e
            print("{}:{}".format(indx, e))

        time.sleep(2)
        indx = indx + 1

    with open('./iiw_specific.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()


def compute_iiw_gpt():
    data = []
    with open("./dataset/iiw_new.jsonl", 'r') as f:
        
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()

    results = []
    indx = 0
    for d in tqdm(data):
        try:
            text1 = d['IIW']
            text2 = d['IIW-P5B']

            human = d['iiw-human-sxs-iiw-p5b']['metrics/Hallucination']

            results.append({
                "IIW": text1,
                "IIW-P5B": text2,
                "gpt_specific": compare_hallucination_gemini(text1, text2),
                "image": d["image/key"],
                "human_specific": human,
            })
        except Exception as e:
            # raise e
            print("{}:{}".format(indx, e))

        time.sleep(2)
        indx = indx + 1

    with open('./iiw_specific_gpt.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()


def compute_dci():
    data = []
    with open("./dataset/dci_new.jsonl", 'r') as f:
        
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()

    results = []
    indx = 0
    for d in tqdm(data):
        try:
            text1 = d['IIW']
            text2 = d['DCI_extra_caption']
            text3 = d['DCI_short_caption']

            human = d['metrics/Specificity']
            score_iiw = spec_score_gemini(text1, verbose=False)
            score_extra = spec_score_gemini(text2, verbose=False)
            # score_short = specific_score2(text3, verbose=False)

            results.append({
                "IIW": text1,
                "DCI_extra": text2,
                "DCI_short": text3,
                "score_iiw": score_iiw,
                "score_extra_dci": score_extra,
                # "score_short_dci": score_short,
                "image": d["image"],
                "human_specific": human,
            })
        except Exception as e:
            # raise e
            print("{}:{}".format(indx, e))

        time.sleep(2)
        indx = indx + 1

    with open('./dci_specific.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()
    
    
def compute_dci_gpt():
    data = []
    with open("./dataset/dci_new.jsonl", 'r') as f:
        
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()

    results = []
    indx = 0
    for d in tqdm(data):
        try:
            text1 = d['IIW']
            text2 = d['DCI_extra_caption']

            human = d['metrics/Hallucination']
            # score_short = specific_score2(text3, verbose=False)

            results.append({
                "IIW": text1,
                "gpt_specific": compare_hallucination_gemini(text1, text2), 
                # "score_short_dci": score_short,
                "image": d["image"],
                "human_specific": human,
            })
        except Exception as e:
            # raise e
            print("{}:{}".format(indx, e))

        time.sleep(2)
        indx = indx + 1

    with open('./dci_specific_gpt.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()


def compute_docci():
    data = []
    with open("./dataset/docci.jsonl", 'r') as f:
        
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()

    results = []
    indx = 0
    for d in tqdm(data):
        try:
            text1 = d['IIW']
            text2 = d['DOCCI']

            human = d['metrics/Specificity']
            score_iiw = spec_score_gemini(text1, verbose=False)
            score_docci = spec_score_gemini(text2, verbose=False)

            results.append({
                "IIW": text1,
                "DOCCI": text2,
                "score_iiw": score_iiw,
                "score_docci": score_docci,
                "image": d["image"],
                "human_specific": human,
            })
        except Exception as e:
            # raise e
            print("{}:{}".format(indx, e))

        time.sleep(2)
        indx = indx + 1

    with open('./docci_specific.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()
    

def compute_docci_gpt():
    data = []
    with open("./dataset/docci.jsonl", 'r') as f:
        
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()

    results = []
    indx = 0
    for d in tqdm(data):
        try:
            text1 = d['IIW']
            text2 = d['DOCCI']

            human = d['metrics/Hallucination']

            results.append({
                "IIW": text1,
                "DOCCI": text2,
                "gpt_specific": compare_hallucination_gemini(text1, text2), 
                "image": d["image"],
                "human_specific": human,
            })
        except Exception as e:
            # raise e
            print("{}:{}".format(indx, e))

        time.sleep(2)
        indx = indx + 1

    with open('./docci_specific_gpt.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()


def corr_dci():
    with open("./dci_specific.json", 'r') as f:
        iiw_data = json.load(f)
    f.close()


    diff = []
    human = []

    main_obj = "IIW"
    target_obj = "DCI"

    mapping_dict = {
        "{} is substantially better".format(main_obj): 2,
        "{} is marginally better".format(main_obj): 1,
        "Neutral": 0,
        "{} is marginally better".format(target_obj): -1,
        "{} is substantially better".format(target_obj): -2
    }

    for d in iiw_data:
        # diff.append(d['score_iiw'] - d['score_extra_dci'])
        diff.append(int(d['score_iiw']) - int(d['score_extra_dci']))
        human.append(mapping_dict[d['human_specific']])

    scaler = MinMaxScaler(feature_range=(-2, 2), clip=True)
    new_diff = scaler.fit_transform(np.array(diff).reshape(-1, 1)).flatten().tolist()
    
    # new_diff = [normalized_score(k) for k in new_diff]
    # new_diff = diff
    # print(new_diff)
    # print(human)

    print("DCI")
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")
    # print("Accuracy")
    # acc = accuracy_score(new_diff, human)
    # print(acc)
    # print("===========")
    

def corr_dci_gpt():
    with open("./dci_specific_gpt.json", 'r') as f:
        iiw_data = json.load(f)
    f.close()


    diff = []
    human = []

    main_obj = "IIW"
    target_obj = "DCI"

    mapping_dict = {
        "{} is substantially better".format(main_obj): 2,
        "{} is marginally better".format(main_obj): 1,
        "Neutral": 0,
        "{} is marginally better".format(target_obj): -1,
        "{} is substantially better".format(target_obj): -2
    }

    for d in iiw_data:
        diff.append(int(d['gpt_specific']) if d['gpt_specific'] != "-" else 0)
        human.append(mapping_dict[d['human_specific']])

    new_diff = diff
    print(new_diff)
    print(human)

    print("DCI")
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")
    print("Accuracy")
    acc = accuracy_score(new_diff, human)
    print(acc)
    print("===========")


def corr_docci():
    with open("./docci_specific.json", 'r') as f:
        iiw_data = json.load(f)
    f.close()


    diff = []
    human = []

    main_obj = "IIW"
    target_obj = "DOCCI"

    mapping_dict = {
        "{} is substantially better".format(main_obj): 2,
        "{} is marginally better".format(main_obj): 1,
        "Neutral": 0,
        "{} is marginally better".format(target_obj): -1,
        "{} is substantially better".format(target_obj): -2
    }

    for d in iiw_data:
        # diff.append(d['score_iiw'] - d['score_docci'])
        diff.append(int(d['score_iiw']) - int(d['score_docci']))
        human.append(mapping_dict[d['human_specific']])

    scaler = MinMaxScaler(feature_range=(-2, 2), clip=True)
    new_diff = scaler.fit_transform(np.array(diff).reshape(-1, 1)).flatten().tolist()
    
    # new_diff = [normalized_score(k) for k in new_diff]
    # new_diff = diff
    # print(new_diff)
    # print(human)

    print("DOCCI")
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")
    
    # print("Accuracy")
    # acc = accuracy_score(new_diff, human)
    # print(acc)
    # print("===========")
    

def corr_docci_gpt():
    with open("./docci_specific_gpt.json", 'r') as f:
        iiw_data = json.load(f)
    f.close()


    diff = []
    human = []

    main_obj = "IIW"
    target_obj = "DOCCI"

    mapping_dict = {
        "{} is substantially better".format(main_obj): 2,
        "{} is marginally better".format(main_obj): 1,
        "Neutral": 0,
        "{} is marginally better".format(target_obj): -1,
        "{} is substantially better".format(target_obj): -2
    }

    for d in iiw_data:
        diff.append(int(d['gpt_specific']) if d['gpt_specific'] != "-" else 0)
        human.append(mapping_dict[d['human_specific']])

    new_diff = diff
    print(new_diff)
    print(human)

    print("DOCCI")
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")
    
    # print("Accuracy")
    # acc = accuracy_score(new_diff, human)
    # print(acc)
    # print("===========")


def corr_iiw():
    with open("./iiw_specific.json", 'r') as f:
        iiw_data = json.load(f)
    f.close()


    diff = []
    human = []

    main_obj = "IIW-Human"
    target_obj = "IIW-P5B"

    mapping_dict = {
        "{} is substantially better".format(main_obj): 2,
        "{} is marginally better".format(main_obj): 1,
        "Neutral": 0,
        "{} is marginally better".format(target_obj): -1,
        "{} is substantially better".format(target_obj): -2
    }

    for d in iiw_data:
        # diff.append(d['score_iiw'] - d['score_p5b'])
        diff.append(int(d['score_iiw']) - int(d['score_p5b']))
        human.append(mapping_dict[d['human_specific']])

    scaler = MinMaxScaler(feature_range=(-2, 2))
    new_diff = scaler.fit_transform(np.array(diff).reshape(-1, 1)).flatten().tolist()
    
    # new_diff = [normalized_score(k) for k in new_diff]
    # new_diff = diff
    # print(new_diff)
    # print(human)

    print("IIW")
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")
    
    # print("Accuracy")
    # acc = accuracy_score(new_diff, human)
    # print(acc)
    # print("===========")
    

def corr_iiw_gpt():
    with open("./iiw_specific_gpt.json", 'r') as f:
        iiw_data = json.load(f)
    f.close()


    diff = []
    human = []

    main_obj = "IIW-Human"
    target_obj = "IIW-P5B"

    mapping_dict = {
        "{} is substantially better".format(main_obj): 2,
        "{} is marginally better".format(main_obj): 1,
        "Neutral": 0,
        "{} is marginally better".format(target_obj): -1,
        "{} is substantially better".format(target_obj): -2
    }

    for d in iiw_data:
        diff.append(int(d['gpt_specific']) if d['gpt_specific'] != "-" else 0)
        human.append(mapping_dict[d['human_specific']])

    new_diff = diff
    print(new_diff)
    print(human)

    print("IIW")
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")
    
    print("Accuracy")
    acc = accuracy_score(new_diff, human)
    print(acc)
    print("===========")


def prepare_data_linearfit():
    caption_source = []
    caption_target = []
    data_source = []
    score_text = []
    score_visual = []
    score_label = []
    
    score_visual_1 = []
    score_visual_2 = []
    score_visual_3 = []
    score_visual_4 = []
    with open("./dataset/dci_new.jsonl", 'r') as f:   
        main_obj = "IIW"
        target_obj = "DCI"

        mapping_dict = {
            "{} is substantially better".format(main_obj): 2,
            "{} is marginally better".format(main_obj): 1,
            "Neutral": 0,
            "{} is marginally better".format(target_obj): -1,
            "{} is substantially better".format(target_obj): -2
        }
        
        data = []
        indx = 0
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

        for d in tqdm(data):
            try:
                text1 = d['IIW']
                text2 = d['DCI_extra_caption']

                human = d['metrics/Specificity']
                spec_text = specific_score(text1, verbose=False) - specific_score(text2, verbose=False)
                vis_score_text_1 = specific_score_new(text1, verbose=False)
                vis_score_text_2 = specific_score_new(text2, verbose=False)
                
                spec_visual = (vis_score_text_1[0]) - (vis_score_text_2[0])
                
                caption_source.append(text1)
                caption_target.append(text2)
                data_source.append("dci")
                score_text.append(spec_text)
                score_visual.append(spec_visual)
                score_label.append(mapping_dict[human])
                score_visual_1.append(vis_score_text_1[1]['Visual Detail'] - vis_score_text_2[1]['Visual Detail'])
                score_visual_2.append(vis_score_text_1[1]['Spatial Specificity'] - vis_score_text_2[1]['Spatial Specificity'])
                score_visual_3.append(vis_score_text_1[1]['Color & Texture'] - vis_score_text_2[1]['Color & Texture'])
                score_visual_4.append(vis_score_text_1[1]['Interpretive Language'] - vis_score_text_2[1]['Interpretive Language'])
                
            except Exception as e:
                # raise e
                print("index {}:{}".format(indx, e))
            time.sleep(2)
            indx = indx + 1
    f.close()
    
    with open("./dataset/docci.jsonl", 'r') as f:
        main_obj = "IIW"
        target_obj = "DOCCI"

        mapping_dict = {
            "{} is substantially better".format(main_obj): 2,
            "{} is marginally better".format(main_obj): 1,
            "Neutral": 0,
            "{} is marginally better".format(target_obj): -1,
            "{} is substantially better".format(target_obj): -2
        }
        
        data = []
        indx = 0
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        for d in tqdm(data):
            try:
                text1 = d['IIW']
                text2 = d['DOCCI']

                human = d['metrics/Specificity']
                
                spec_text = specific_score(text1, verbose=False) - specific_score(text2, verbose=False)
                vis_score_text_1 = specific_score_new(text1, verbose=False)
                vis_score_text_2 = specific_score_new(text2, verbose=False)
                
                spec_visual = (vis_score_text_1[0]) - (vis_score_text_2[0])
                
                caption_source.append(text1)
                caption_target.append(text2)
                data_source.append("docci")
                score_text.append(spec_text)
                score_visual.append(spec_visual)
                score_label.append(mapping_dict[human])
                score_visual_1.append(vis_score_text_1[1]['Visual Detail'] - vis_score_text_2[1]['Visual Detail'])
                score_visual_2.append(vis_score_text_1[1]['Spatial Specificity'] - vis_score_text_2[1]['Spatial Specificity'])
                score_visual_3.append(vis_score_text_1[1]['Color & Texture'] - vis_score_text_2[1]['Color & Texture'])
                score_visual_4.append(vis_score_text_1[1]['Interpretive Language'] - vis_score_text_2[1]['Interpretive Language'])

            except Exception as e:
                # raise e
                print("index {}:{}".format(indx, e))

            time.sleep(2)
            indx = indx + 1
    f.close()
    
    with open("./dataset/iiw_new.jsonl", 'r') as f:
        
        main_obj = "IIW-Human"
        target_obj = "IIW-P5B"

        mapping_dict = {
            "{} is substantially better".format(main_obj): 2,
            "{} is marginally better".format(main_obj): 1,
            "Neutral": 0,
            "{} is marginally better".format(target_obj): -1,
            "{} is substantially better".format(target_obj): -2
        }
        
        data = []
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        indx = 0
        
        for d in tqdm(data):
            try:
                text1 = d['IIW']
                text2 = d['IIW-P5B']
                human = d['iiw-human-sxs-iiw-p5b']['metrics/Specificity']

                spec_text = specific_score(text1, verbose=False) - specific_score(text2, verbose=False)
                vis_score_text_1 = specific_score_new(text1, verbose=False)
                vis_score_text_2 = specific_score_new(text2, verbose=False)
                
                spec_visual = (vis_score_text_1[0]) - (vis_score_text_2[0])
                
                caption_source.append(text1)
                caption_target.append(text2)
                data_source.append("iiw")
                score_text.append(spec_text)
                score_visual.append(spec_visual)
                score_label.append(mapping_dict[human])
                score_visual_1.append(vis_score_text_1[1]['Visual Detail'] - vis_score_text_2[1]['Visual Detail'])
                score_visual_2.append(vis_score_text_1[1]['Spatial Specificity'] - vis_score_text_2[1]['Spatial Specificity'])
                score_visual_3.append(vis_score_text_1[1]['Color & Texture'] - vis_score_text_2[1]['Color & Texture'])
                score_visual_4.append(vis_score_text_1[1]['Interpretive Language'] - vis_score_text_2[1]['Interpretive Language'])
            except Exception as e:
                # raise e
                print("index {}:{}".format(indx, e))
            time.sleep(2)
            indx = indx + 1
    f.close()
    
    print(len(caption_source))
    print(len(caption_target))
    print(len(data_source))
    print(len(score_text))
    print(len(score_visual))
    print(len(score_label))
    print(len(score_visual_1))
    print(len(score_visual_2))
    print(len(score_visual_3))
    print(len(score_visual_4))
    
    datafr = pd.DataFrame({
        "caption_source": caption_source,
        "caption_target": caption_target,
        "type_data": data_source,
        "score_text": score_text,
        "score_visual": score_visual,
        "label": score_label,
        "score_visual_1": score_visual_1,
        "score_visual_2": score_visual_2,
        "score_visual_3": score_visual_3,
        "score_visual_4": score_visual_4
    })
    
    datafr.to_csv("data_fit7.csv")


if __name__ == '__main__':
    # compute_dci()
    # compute_docci()
    # compute_iiw()

    # # ## SAMPLE 
    # # # sample = data[37]
    # # # text1 = sample['IIW']
    # # # text2 = sample['IIW-P5B']
    # # # print(text1)
    # # # print(specific_score(text1))
    # # # print("-------")
    # # # print(text2)
    # # # print(specific_score(text2))


    # ## COMPUTE CORRELATION
    # corr_dci()
    # corr_docci()
    corr_iiw()
    
    
    # # GPT
    # print("GEMINI")
    # # compute_dci_gpt()
    # # compute_docci_gpt()
    # # compute_iiw_gpt()

    # # COMPUTE CORRELATION
    # # corr_dci_gpt()
    # # corr_docci_gpt()
    # corr_iiw_gpt()

    # # make fit
    # prepare_data_linearfit()