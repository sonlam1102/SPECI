import json
from misc import query_image_gen_diffusion, load_stable_diffusion, load_clip, load_blip
from tqdm import tqdm
import shutil
from PIL import Image
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import accuracy_score
from functools import partial
from torchmetrics.functional.multimodal import clip_score
import torch 
import numpy as np
from visual_bge.modeling import Visualized_BGE
from sklearn.metrics import cohen_kappa_score
 
def gen_image():
    pipeline = load_stable_diffusion()
    with open("./image_db_spec_score_new_ver10.json", "r") as f:
        data = json.load(f)
    f.close()
    
    # new_data = [d for k, v in d.items() if v["instruct_blip"] != "" for d in data]
    new_data = data[0:150]
    for d in tqdm(new_data):
        for k, v in d.items():
            if v['instruct_blip'] == "":
                continue
            else:
                query_image_gen_diffusion(pipeline, v['gpt4o'], "./image_gen_results_ver10/gpt4o/{}".format(k))
                query_image_gen_diffusion(pipeline, v['instruct_blip'], "./image_gen_results_ver10/blip/{}".format(k))
                shutil.copyfile("./Dataset/flickr30k-images/{}.jpg".format(k), "./image_gen_results_ver10/original/{}.jpg".format(k))

def eval_image():
    device = "cuda"
    clip_model, clip_process = load_clip(device)
    device = "cuda"
    with open("./image_db_spec_score_new_ver10.json", "r") as f:
        data = json.load(f)
    f.close()
    
    similarity_gpt_original = 0
    similarity_blip_original = 0
    new_data = data[0:100]
    
    diff_score_clip = []
    diff_score_spec = []
    
    kdiff_score_clip = []
    kdiff_score_spec = []
    
    win_count = 0
    with torch.no_grad():
        for d in tqdm(new_data):
            for k, v in d.items():
                image_gpt = Image.open("./image_gen_results_ver10/gpt4o/{}.png".format(k))
                image_blip = Image.open("./image_gen_results_ver10/blip/{}.png".format(k))
                image_ground = Image.open("./image_gen_results_ver10/original/{}.jpg".format(k))
                
                image_input_gpt = clip_process(image_gpt).unsqueeze(0).to(device)
                image_input_blip = clip_process(image_blip).unsqueeze(0).to(device)
                image_input_ground = clip_process(image_ground).unsqueeze(0).to(device)
                
                image_features_gpt = clip_model.encode_image(image_input_gpt)
                image_features_blip = clip_model.encode_image(image_input_blip)
                image_features_ground = clip_model.encode_image(image_input_ground)
                
                similarity_gpt_original = F.cosine_similarity(image_features_gpt, image_features_ground, dim=1).item()
                similarity_blip_original = F.cosine_similarity(image_features_blip, image_features_ground, dim=1).item()
                
                # similarity_gpt_original = (image_features_gpt @ image_features_ground.T).flatten().item()
                # similarity_blip_original = (image_features_blip @ image_features_ground.T).flatten().item()
                
                # print(similarity_gpt_original.flatten().item())
                # print(similarity_blip_original.flatten().item())
                # raise Exception

                diff_score_clip.append(similarity_gpt_original - similarity_blip_original)
                diff_score_spec.append(v['score_gpt4'] - v['score_blip'])
                
                # if v['score_gpt4'] >= v['score_blip'] and similarity_gpt_original >= similarity_blip_original:
                #     win_count = win_count + 1 
                
                if v['score_gpt4'] >= v['score_blip']:
                    kdiff_score_spec.append(1)
                else:
                    kdiff_score_spec.append(0)
                
                if similarity_gpt_original >= similarity_blip_original:
                    kdiff_score_clip.append(1)
                else:
                    kdiff_score_clip.append(0)
    
    # print(kdiff_score_clip)
    # print(kdiff_score_spec)
    
    # print("Win rate: {}".format((win_count / len(new_data))))
    acc = accuracy_score(kdiff_score_clip, kdiff_score_spec)
    print(f"Acc CLIP: {acc}")
    
    acc = cohen_kappa_score(kdiff_score_clip, kdiff_score_spec, weights="linear")
    print(f"Kappa CLIP: {acc}")
    
    print("Spearman Corr")
    correlation, p_value = spearmanr(diff_score_spec, diff_score_clip)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(diff_score_spec, diff_score_clip)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(diff_score_spec, diff_score_clip)
    print(correlation)
    # print(p_value)
    print("===========")
    
    
def eval_image2():
    device = "cuda"
    model_multimodal = Visualized_BGE(model_name_bge = "/scr/huggingface_models/bge-m3", model_weight="./Visualized_m3.pth")
    # model_multimodal = Visualized_BGE(model_name_bge = "/scr/huggingface_models/bge-base-en-v1.5", model_weight="./Visualized_base_en_v1.5.pth")
    model_multimodal.eval()
    device = "cuda"
    
    with open("./image_db_spec_score_new_ver10.json", "r") as f:
        data = json.load(f)
    f.close()
    similarity_gpt_original = 0
    similarity_blip_original = 0
    new_data = data[0:100]
    
    diff_score_clip = []
    diff_score_spec = []
    kdiff_score_clip = []
    kdiff_score_spec = []
    
    win_count = 0
    with torch.no_grad():
        for d in tqdm(new_data):
            for k, v in d.items():
                image_input_gpt = model_multimodal.encode(image="./image_gen_results_ver10/gpt4o/{}.png".format(k))
                image_input_blip = model_multimodal.encode(image="./image_gen_results_ver10/blip/{}.png".format(k))
                image_input_ground = model_multimodal.encode(image="./image_gen_results_ver10/original/{}.jpg".format(k))
                
                similarity_gpt_original = F.cosine_similarity(image_input_gpt, image_input_ground, dim=1).item()
                similarity_blip_original = F.cosine_similarity(image_input_blip, image_input_ground, dim=1).item()
                
                # similarity_gpt_original = (image_input_gpt @ image_input_ground.T).flatten().item()
                # similarity_blip_original = (image_input_blip @ image_input_ground.T).flatten().item()
                
                # print(similarity_gpt_original)
                # print(similarity_blip_original)
                # print("--------")
                # print(image_input_gpt)
                # print(image_input_blip)
                # raise Exception

                diff_score_clip.append(similarity_gpt_original - similarity_blip_original)
                diff_score_spec.append(v['score_gpt4'] - v['score_blip'])
                
                # if v['score_gpt4'] >= v['score_blip'] and similarity_gpt_original >= similarity_blip_original:
                #     win_count = win_count + 1 
                
                if v['score_gpt4'] - v['score_blip'] > 0 :
                    kdiff_score_spec.append(1)
                else:
                    kdiff_score_spec.append(0)
                
                if similarity_gpt_original - similarity_blip_original > 0:
                    kdiff_score_clip.append(1)
                else:
                    kdiff_score_clip.append(0)
    
    # print("Win rate: {}".format((win_count / len(new_data))))
    
    print(diff_score_clip)
    print(diff_score_spec)
    acc = accuracy_score(kdiff_score_clip, kdiff_score_spec)
    print(f"Acc BGE: {acc}")
    
    acc = cohen_kappa_score(kdiff_score_clip, kdiff_score_spec, weights="linear")
    print(f"Kappa BGE: {acc}")
    
    print("Spearman Corr")
    correlation, p_value = spearmanr(diff_score_spec, diff_score_clip)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(diff_score_spec, diff_score_clip)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(diff_score_spec, diff_score_clip)
    print(correlation)
    # print(p_value)
    print("===========")
    

if __name__ == '__main__':
    # gen_image()
    eval_image()
    eval_image2()
    # eval_clip_score()
