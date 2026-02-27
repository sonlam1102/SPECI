import json


if __name__ == '__main__':
    with open("./image_db_spec_score_new_ver10.json", "r") as f:
        data = json.load(f)
    f.close()
    
    capmas_gpt4o = {}
    capmas_blip = {}
    
    for d in data:
        for k, v in d.items():
            capmas_gpt4o[k+".jpg"] = v['gpt4o']
            capmas_blip[k+".jpg"] = v['instruct_blip']
    
    
    with open('./CapMAS/sample_captions/gpt4o.json', 'w', encoding='utf-8') as f:
        json.dump(capmas_gpt4o, f, ensure_ascii=False, indent=4)
    f.close()
    
    
    with open('./CapMAS/sample_captions/blip.json', 'w', encoding='utf-8') as f:
        json.dump(capmas_blip, f, ensure_ascii=False, indent=4)
    f.close()
