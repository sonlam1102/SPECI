import asyncio
from evaluation.gveval.scorer import Scorer
import json 

from tqdm import tqdm 

async def compute_dci(scorer):
    data = []
    with open("../dataset/dci_new.jsonl", 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()
    
    results = []
    for d in tqdm(data):
        text1 = d['IIW']
        text2 = d['DCI_extra_caption']
        text3 = d['DCI_short_caption']
        human = d['metrics/Specificity']
        
        score = await scorer.gveval([text2], [text1], img="../dataset/dci_images/{}".format(d["image"]), visual='img', setting='ref-free', accr=False, resolution='low')
        
        results.append({
            "IIW": text1,
            "DCI_extra": text2,
            "DCI_short": text3,
            "score_iiw": score['final_score']+score['final_score'],
            "score_extra_dci": score['final_score'],
            # "score_short_dci": score_short,
            "image": d["image"],
            "human_specific": human,
        })
        
    with open('./dci_specific.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()
    
    

async def compute_docci(scorer):
    data = []
    with open("../dataset/docci.jsonl", 'r') as f:
        
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()
    
    results = []
    for d in tqdm(data):
        text1 = d['IIW']
        text2 = d['DOCCI']

        human = d['metrics/Specificity']
        
        score = await scorer.gveval([text2], [text1], img="../dataset/docci_images/images/{}.jpg".format(d["image"]), visual='img', setting='ref-free', accr=False, resolution='low')
        
        results.append({
            "IIW": text1,
            "DOCCI": text2,
            "score_iiw": score['final_score']+score['final_score'],
            "score_docci": score['final_score'],
            "image": d["image"],
            "human_specific": human,
        })
        
    with open('./docci_specific.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()
    

async def compute_iiw(scorer):
    data = []
    with open("../dataset/iiw_new.jsonl", 'r') as f:
        
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()
    
    results = []
    for d in tqdm(data):
        text1 = d['IIW']
        text2 = d['IIW-P5B']

        human = d['iiw-human-sxs-iiw-p5b']['metrics/Specificity']
        
        score = await scorer.gveval([text2], [text1], img="../dataset/docci_images/images_aar/{}.jpg".format(d["image/key"]), visual='img', setting='ref-free', accr=False, resolution='low')
        
        results.append({
            "IIW": text1,
            "IIW-P5B": text2,
            "score_iiw": score['final_score']+score['final_score'],
            "score_p5b": score['final_score'],
            "image": d["image/key"],
            "human_specific": human,
        })
        
    with open('./iiw_specific.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()
    

# Run the main function
if __name__ == "__main__":
    scorer = Scorer()
    print("dci")
    asyncio.run(compute_dci(scorer))
    print("docci")
    asyncio.run(compute_docci(scorer))
    print("iiw")
    asyncio.run(compute_iiw(scorer))