from polos.models import download_model, load_checkpoint
from PIL import Image
import json
from tqdm import tqdm 

def compute_dci(model):
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
        
        lst_text1 = text1.split(".")
        lst_text2 = text2.split(".")
        tmp = [
            {
                "img": Image.open("../dataset/dci_images/{}".format(d["image"])).convert("RGB"),
                "mt": tmp_text,
                "refs": lst_text1,
            } for tmp_text in lst_text2
        ]
        
        score = model.predict(tmp, batch_size=8, cuda=True)[1][0]
            
        results.append({
            "IIW": text1,
            "DCI_extra": text2,
            "DCI_short": text3,
            "score_iiw": score+score,
            "score_extra_dci": score,
            # "score_short_dci": score_short,
            "image": d["image"],
            "human_specific": human,
        })
        
    with open('./dci_specific.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()
    
    
def compute_docci(model):
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
        
        lst_text1 = text1.split(".")
        lst_text2 = text2.split(".")
        tmp = [
            {
                "img": Image.open("../dataset/docci_images/images/{}.jpg".format(d["image"])).convert("RGB"),
                "mt": tmp_text,
                "refs": lst_text1,
            } for tmp_text in lst_text2
        ]
        
        score = model.predict(tmp, batch_size=8, cuda=True)[1][0]
        
        results.append({
            "IIW": text1,
            "DOCCI": text2,
            "score_iiw": score+score,
            "score_docci": score,
            "image": d["image"],
            "human_specific": human,
        })
        
    with open('./docci_specific.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()


def compute_iiw(model):
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
        
        lst_text1 = text1.split(".")
        lst_text2 = text2.split(".")
        tmp = [
            {
                "img": Image.open("../dataset/docci_images/images_aar/{}.jpg".format(d["image/key"])).convert("RGB"),
                "mt": tmp_text,
                "refs": lst_text1,
            } for tmp_text in lst_text2
        ]
        
        score = model.predict(tmp, batch_size=8, cuda=True)[1][0]
        
        results.append({
            "IIW": text1,
            "IIW-P5B": text2,
            "score_iiw": score+score,
            "score_p5b": score,
            "image": d["image/key"],
            "human_specific": human,
        })
        
    with open('./iiw_specific.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()


if __name__ == '__main__':
    model_path = download_model("polos")
    model = load_checkpoint(model_path)
    print("dci")
    compute_dci(model)
    
    print("docci")
    compute_docci(model)
    
    print("iiw")
    compute_iiw(model)
