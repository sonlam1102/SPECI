import heapq
from specfic_score import compute_specific_score
import numpy as np
from visual_bge.modeling import Visualized_BGE
import json
from FlagEmbedding import BGEM3FlagModel
from copy import deepcopy
from tqdm import tqdm
import torch
from tqdm import tqdm
import numpy as np
import clip
from PIL import Image
from misc import load_clip
from numpy import dot
from numpy.linalg import norm
import time

def apk(actual, predicted, k=10):
    """
    Calculates the Average Precision at K (AP@K) for a single user.

    Args:
        actual (list): A list of actual relevant items for a user.
        predicted (list): A list of recommended items for a user, ranked by relevance.
        k (int): The number of top recommendations to consider.

    Returns:
        float: The AP@K score.
    """
    if not actual:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual_lists, predicted_lists, k=10):
    """
    Calculates the Mean Average Precision at K (MAP@K) across multiple users.

    Args:
        actual_lists (list of lists): A list where each sublist contains
                                      the actual relevant items for a user.
        predicted_lists (list of lists): A list where each sublist contains
                                       the recommended items for a user, ranked by relevance.
        k (int): The number of top recommendations to consider.

    Returns:
        float: The MAP@K score.
    """
    return sum(apk(a, p, k) for a, p in zip(actual_lists, predicted_lists)) / len(actual_lists)


def calculate_hit_rate(recommended_lists, relevant_items):
    """
    Calculates the hit rate for a set of recommendations.

    Args:
        recommended_lists (list of lists): Each inner list contains recommended items for a user.
        relevant_items (list of lists): Each inner list contains relevant items for a user.

    Returns:
        float: The hit rate.
    """
    hits = 0
    total_users = len(recommended_lists)

    for i in range(total_users):
        user_recommended = set(recommended_lists[i])
        user_relevant = set(relevant_items[i])

        if not user_recommended.isdisjoint(user_relevant):
            hits += 1
            
    return hits / total_users


def eval_retrieval(y_pred, y_true):    
    correct_retrieved = sum([1 if p in y_true else 0 for p in y_pred])

    precision = correct_retrieved / len(y_pred)
    recall = correct_retrieved / len(y_true)
    f2 = (4 * precision * recall / (5 * precision + recall + 1e-9))
    

    return f2, precision, recall, mapk(y_true, y_pred, k=len(y_pred))


def get_top_k(predict_output, top_k):
    lst_out_one_hot = np.zeros(len(predict_output))
    idx_top_k = heapq.nlargest(top_k, range(len(predict_output)), predict_output.take)

    i = 1
    for idx in idx_top_k:
        lst_out_one_hot[idx] = i
        i = i + 1

    return lst_out_one_hot


def make_new_image_db_with_score():
    with open("./image_db_ver10.json", "r") as f:
        data = json.load(f)
    f.close()
    new_data = deepcopy(data)
    
    for d in tqdm(new_data):
        for k, v in d.items():
            try:
                score_gpt = compute_specific_score(v['gpt4o'])
            except Exception as e:
                print(e)
                print(k)
                score_gpt = 0
            time.sleep(1)
            try:
                score_blip = compute_specific_score(v['instruct_blip'])
            except Exception as e:
                print(e)
                print(k)
                score_blip = 0
                
            v['score_gpt4'] = score_gpt
            v['score_blip'] = score_blip
        time.sleep(1)
    with open('./image_db_spec_score_new_ver10.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    f.close()
    

def make_image_db_embedding_multimodal(model):
    lst_ids = []
    with open("./image_db_spec_score_new_ver10.json", "r") as f:
        data = json.load(f)
    f.close()
    
    embedding_gpt_image = None
    embedding_blip_image = None
    embedding_best_image = None
    embedding_image_only = None
    lst_ids = []
    
    with torch.no_grad():
        for d in tqdm(data):
            for k, v in d.items():
                lst_ids.append(k)
                gpt_embd = model.encode(text=v['gpt4o']).cpu().detach().numpy()
                blip_emd = model.encode(text=v['instruct_blip'].strip()).cpu().detach().numpy()
                image_embed = model.encode(image="./Dataset/flickr30k-images/{}.jpg".format(k)).cpu().detach().numpy()
                best_capt = v['gpt4o'] if v['score_gpt4'] > v['score_blip'] else v['instruct_blip']
                choice_best = model.encode(text=best_capt.strip()).cpu().detach().numpy()
                
                if embedding_gpt_image is None:
                    embedding_gpt_image = gpt_embd
                else:
                    embedding_gpt_image = np.append(embedding_gpt_image, gpt_embd, axis=0)
                
                if embedding_blip_image is None:
                    embedding_blip_image = blip_emd
                else:
                    embedding_blip_image = np.append(embedding_blip_image, blip_emd, axis=0)
                
                if embedding_best_image is None:
                    embedding_best_image = choice_best
                else:
                    embedding_best_image = np.append(embedding_best_image, choice_best, axis=0)
                
                if embedding_image_only is None:
                    embedding_image_only = image_embed
                else:
                    embedding_image_only = np.append(embedding_image_only, image_embed, axis=0)
        
    print(embedding_gpt_image.shape)
    print(embedding_blip_image.shape)
    print(embedding_best_image.shape)
    print(embedding_image_only.shape)
    
    embedding_id = np.array(lst_ids)
    np.save("./multimodal_ver10/bge/images_id.npy", embedding_id)
    np.save("./multimodal_ver10/bge/embedding_image_gpt.npy", embedding_gpt_image)
    np.save("./multimodal_ver10/bge/embedding_image_blip.npy", embedding_blip_image)
    np.save("./multimodal_ver10/bge/embedding_image_best.npy", embedding_best_image)
    np.save("./multimodal_ver10/bge/embedding_image_only.npy", embedding_image_only)
    
    # embedding_id = np.array(lst_ids)
    # np.save("./multimodal_ver7/bge-m3/images_id.npy", embedding_id)
    # np.save("./multimodal_ver7/bge-m3/embedding_image_gpt.npy", embedding_gpt_image)
    # np.save("./multimodal_ver7/bge-m3/embedding_image_blip.npy", embedding_blip_image)
    # np.save("./multimodal_ver7/bge-m3/embedding_image_best.npy", embedding_best_image)
    # np.save("./multimodal_ver7/bge-m3/embedding_image_only.npy", embedding_image_only)


def get_result(one_hot, ids):
    assert len(one_hot) == len(ids)
    lst_results = []
    for i in range(0, len(one_hot)):
        if one_hot[i] >= 1:
            lst_results.append(ids[i])
            
    return lst_results
    

def retrieva_data_multimodal(model, top_k):
    with open("./test_new2.json", "r") as f:
        test_data = json.load(f)
    f.close()
    
    image_id = np.load("./multimodal_ver10/bge/images_id.npy")
    embedding_image_gpt = np.load("./multimodal_ver10/bge/embedding_image_gpt.npy")
    embedding_image_blip = np.load("./multimodal_ver10/bge/embedding_image_blip.npy")
    embedding_image_best = np.load("./multimodal_ver10/bge/embedding_image_best.npy")
    embedidng_image_only = np.load("./multimodal_ver10/bge/embedding_image_only.npy")
    
    # image_id = np.load("./multimodal_ver7/bge-m3/images_id.npy")
    # embedding_image_gpt = np.load("./multimodal_ver7/bge-m3/embedding_image_gpt.npy")
    # embedding_image_blip = np.load("./multimodal_ver7/bge-m3/embedding_image_blip.npy")
    # embedding_image_best = np.load("./multimodal_ver7/bge-m3/embedding_image_best.npy")
    # embedidng_image_only = np.load("./multimodal_ver7/bge-m3/embedding_image_only.npy")

    results = []
    with torch.no_grad():
        for d in tqdm(test_data):
            query_embedding = model.encode(text=d['query'].strip()).cpu().detach().numpy()
        
            # print(query_embedding.shape)
            # print(embedding_image_gpt.shape)
            # lst_candidate_gpt = query_embedding @ embedding_image_gpt.T
            # lst_candidate_blip = query_embedding @ embedding_image_blip.T 
            # lst_candidate_best = query_embedding @ embedding_image_best.T 
            # lst_candidate_image = query_embedding @ embedidng_image_only.T 
        
            lst_candidate_gpt = (query_embedding @ embedding_image_gpt.T) / (norm(query_embedding)*norm(embedding_image_gpt))
            lst_candidate_blip = (query_embedding @ embedding_image_blip.T) / (norm(query_embedding)*norm(embedding_image_blip)) 
            lst_candidate_best = (query_embedding @ embedding_image_best.T) / (norm(query_embedding)*norm(embedding_image_best)) 
            lst_candidate_image = (query_embedding @ embedidng_image_only.T) /  (norm(query_embedding)*norm(embedidng_image_only))
            
            new_candidate_gpt = get_top_k(lst_candidate_gpt.T.flatten(), top_k=top_k)
            new_candidate_blip = get_top_k(lst_candidate_blip.T.flatten(), top_k=top_k)
            new_candidate_best = get_top_k(lst_candidate_best.T.flatten(), top_k=top_k)
            new_candidate_image = get_top_k(lst_candidate_image.T.flatten(), top_k=top_k)
            
            results.append({
                "query": d['query'],
                "ground_truth": d['meta_data']['images'],
                "gpt_results": get_result(new_candidate_gpt, image_id),
                "blip_results": get_result(new_candidate_blip, image_id),
                "best_caption_results": get_result(new_candidate_best, image_id),
                "image_results": get_result(new_candidate_image, image_id)
            })
    
    return results


def eval_results(result, key=0):
    f2_gpt = 0
    f2_blip = 0
    f2_best = 0
    f2_image = 0
    for d in result:
        f2_gpt += eval_retrieval(d['gpt_results'], d['ground_truth'])[key]
        f2_blip += eval_retrieval(d['blip_results'], d['ground_truth'])[key]
        f2_best += eval_retrieval(d['best_caption_results'], d['ground_truth'])[key]
        f2_image += eval_retrieval(d['image_results'], d['ground_truth'])[key] 
    
    return f2_gpt / len(result), f2_blip / len(result), f2_best / len(result), f2_image / len(result)


# def eval_win_rate(result, multimodal=True, key=0):
#     win_rate = 0
#     for d in result:
#         recall1 = 
    
#     return f2_gpt / len(result), f2_blip / len(result), f2_best / len(result), f2_image / len(result)

if __name__ == '__main__':
    # make_new_image_db_with_score()
    
    # model_multimodal = Visualized_BGE(model_name_bge = "/scr/huggingface_models/bge-base-en-v1.5", model_weight="./Visualized_base_en_v1.5.pth")
    # model_multimodal.eval()
    model_multimodal = Visualized_BGE(model_name_bge = "/scr/huggingface_models/bge-m3", model_weight="./Visualized_m3.pth")
    model_multimodal.eval()
    # model_multimodal.eval()
    make_image_db_embedding_multimodal(model_multimodal)
    
    print("--multimodal--")
    predicts = retrieva_data_multimodal(model_multimodal, top_k=1)
    # with open('./prediction_top1.json', 'w', encoding='utf-8') as f:
    #     json.dump(predicts, f, ensure_ascii=False, indent=4)
    # f.close()
    print("F2:")
    print(eval_results(predicts, key=0))
    print("Prec:")
    print(eval_results(predicts, key=1))
    print("Recall:")
    print(eval_results(predicts, key=2))
    print("MAP:")
    print(eval_results(predicts, key=3))