import json
import sys
import os
from vlm_as_a_judge import mllm_judge_pairs
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score

def calculate_caparena_auto_scores():
    with open("./image_db_spec_score_new_ver10.json", "r") as f:
        data = json.load(f)
    f.close()
    
    new_data = data
    cand_data = []
    for d in new_data:
        for k, v in d.items():
            cand_data.append({
                "img": k+".jpg",
                "caption1": v['gpt4o'],
                "caption2": v['instruct_blip'],
                "score_caption_1": v['score_gpt4'],
                "score_caption_2": v['score_blip'],
                "speci_judge": "Caption 1 is better" if v['score_gpt4'] >= v['score_blip'] else "Caption 2 is better"
            })      
    mllm_judge_pairs(cand_data, "./Dataset/flickr30k-images")


def count_win_rate(data):
    win = 0
    diff_spec = []
    diff_cap = []
    for d in data:
        if d["judge"] == "Caption 1 is better":
            diff_cap.append(1)
        else:
            diff_cap.append(0)
    
        if d['score_caption_1'] > d['score_caption_2']:
            diff_spec.append(1)
        else:
            diff_spec.append(0)
    
    print(diff_spec)
    print(diff_cap)
    kappa = accuracy_score(diff_spec, diff_cap)
    print(f"Acc: {kappa}")
    
    kappa = cohen_kappa_score(diff_spec, diff_cap)
    print(f"Kappa: {kappa}")
    
    return win / len(data)


if __name__ == '__main__':
    with open("./caparena_ver3.json", "r") as f:
        data = json.load(f)
    f.close()
    
    new_data = data[0:100]
    print(count_win_rate(new_data))
    
