from transformers import BlipProcessor, BlipModel, BlipForImageTextRetrieval
from PIL import Image
import json
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau, pearsonr
from torchmetrics.functional.multimodal import clip_score
import numpy as np
from functools import partial
import torch

def calculate_clip_score(clip_score_fn, images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


def make_blip_score(processor, model):
    with open("./image_db_spec_score_new_ver10.json", "r") as f:
        data = json.load(f)
    f.close()
    
    win_count = 0
    new_data = data[0:100]
    
    diff_score_spec = []
    diff_score_bl = []
    with torch.no_grad():
        for d in tqdm(new_data):
                for k, v in d.items():
                    image_ground = Image.open("./Dataset/flickr30k-images/{}.jpg".format(k))
                    text_gpt = v['gpt4o']
                    text_blip = v['instruct_blip'].strip()
                    
                    inputs_gpt = processor(image_ground, text_gpt, return_tensors="pt").to("cuda")
                    outputs_gpt = model(**inputs_gpt, use_itm_head=False)
                    # itm_score_gpt = outputs_gpt[0]
                    itm_score_gpt = outputs_gpt[0].detach().cpu().item()
                    # itm_score_gpt = outputs_gpt[0].detach().cpu().numpy().flatten()[0].item()
                    
                    
                    # print(outputs_gpt[0].detach().cpu().item())
                    
                    inputs_blip = processor(image_ground, text_blip, return_tensors="pt").to("cuda")
                    outputs_blip = model(**inputs_blip, use_itm_head=False)
                    # itm_score_blip = outputs_blip[0]
                    itm_score_blip = outputs_blip[0].detach().cpu().item()
                    # itm_score_blip = outputs_blip[0].detach().cpu().numpy().flatten()[0].item()
                    
                    # print(outputs_blip[0])
                    # print(itm_score_blip)
                    
                    # raise Exception
                    
                    if v['score_gpt4'] >= v['score_blip'] and itm_score_gpt >= itm_score_blip:
                        win_count += 1 

                    diff_score_spec.append(v['score_gpt4'] - v['score_blip'])
                    diff_score_bl.append(itm_score_gpt - itm_score_blip)

    print("Win rate: {}".format((win_count / len(new_data))))
    
    # print(diff_score_bl)
    # print(diff_score_spec)
    print("Spearman Corr")
    correlation, p_value = spearmanr(diff_score_spec, diff_score_bl)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(diff_score_spec, diff_score_bl)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(diff_score_spec, diff_score_bl)
    print(correlation)
    # print(p_value)
    print("===========")
    


# def eval_clip_score():
#     with open("./image_db_spec_score_new_ver10.json", "r") as f:
#         data = json.load(f)
#     f.close()
#     new_data = data[0:200]
    
#     # clip_score_fn = partial(clip_score, model_name_or_path="/scr/huggingface_models/clip-vit-base-patch16")
#     clip_score_fn = partial(clip_score, model_name_or_path="/scr/huggingface_models/clip-vit-large-patch14")
    
#     score_spec = []
#     score_clip = []
#     win_count = 0
#     for d in new_data:
#         for k, v in d.items():
#             score_spec.append(v['score_gpt4'] - v['score_blip'])
#             image_ground = Image.open("./image_gen_results_ver10/original/{}.jpg".format(k)).convert("RGB")
#             image_ground = np.array(image_ground)
#             clip_score_gpt = calculate_clip_score(clip_score_fn, image_ground, v['score_gpt4'])
#             clip_score_blip = calculate_clip_score(clip_score_fn, image_ground, v['score_blip'])
            
#             score_clip.append(clip_score_gpt - clip_score_blip)
            
#             if v['score_gpt4'] >= v['score_blip'] and clip_score_gpt >= clip_score_blip:
#                 win_count += 1 
    
#     print("Win rate: {}".format((win_count / len(new_data))))
    
#     print("Spearman Corr")
#     correlation, p_value = spearmanr(score_spec, score_clip)
#     print(correlation)
#     # print(p_value)
#     print("===========")

#     print("Kendau Corr")
#     correlation, p_value = kendalltau(score_spec, score_clip)
#     print(correlation)
#     # print(p_value)
#     print("===========")

#     print("Pearson Corr")
#     correlation, p_value = pearsonr(score_spec, score_clip)
#     print(correlation)
#     # print(p_value)
#     print("===========")
    

if __name__ == '__main__':
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-flickr")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-flickr").to("cuda")
    model.eval()
    make_blip_score(processor, model)
