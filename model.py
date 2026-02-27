import torch
from transformers import AutoTokenizer, AutoModel
import math
from PIL import Image
from datasets import load_dataset
import random
import torchvision.transforms as T
import requests
from torchvision.transforms.functional import InterpolationMode
import json
from tqdm import trange, tqdm
from dcs_score import DCScore

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def generate_description(query, image, model, tokenizer):
    pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=2048, do_sample=True, temperature=0.5)

    question = '<image>\n{}'.format(query)
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response


def main_generate_caption_sharegpt4v():
    ## PREPARE Model 
    path = 'OpenGVLab/InternVL2_5-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    
    ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V")['train']

    sample = ds[0:15]

    PATH = "./data/version/3/generated_caption.txt"

    with open(PATH, "r") as f:
        data = f.read()
    f.close()

    # PREPARE INSTRUCTION
    total_instruct = data.split("==================================")
    total_instruct = total_instruct[0:len(total_instruct)-1]
    print(len(total_instruct))

    free_text_instruct = []
    structured_instruct = []
    for ins in total_instruct:
        if "```json" in ins:
            structured_instruct.append(ins)
        else:
            free_text_instruct.append(ins)
    print("Total free text instruction: {}".format(len(free_text_instruct)))
    print("Total structured instruction: {}".format(len(structured_instruct)))

    # GENERATE CAPTION
    results = []
    for i in trange(0, len(sample['image'])):
        seed_free_text_instruct = random.sample(free_text_instruct, 100)
        seed_structured_instruct = random.sample(structured_instruct, 100)

        seed_sample_img = sample['image'][i]
        url_image = "http://images.cocodataset.org/train2017/" + seed_sample_img.split("/")[-1]
        im = Image.open(requests.get(url_image, stream=True).raw)

        sharegpt4v_res = sample['conversations'][i][1]['value']

        caption_free_text = []
        caption_structured = []

        for r_ins in tqdm(seed_free_text_instruct):
            response = generate_description(r_ins, im, model, tokenizer)
            caption_free_text.append(response)
        
        for r_ins in tqdm(seed_structured_instruct):
            response = generate_description(r_ins, im, model, tokenizer)
            caption_structured.append(response)
        
        results.append({
            "free_text_instruction": seed_free_text_instruct,
            "structured_instruction": seed_structured_instruct,
            "free_text_caption": caption_free_text,
            "structured_caption": caption_structured,
            "image_url": url_image,
            "response": response,
            "gpt": sharegpt4v_res
        })

    with open("sample_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    f.close()



if __name__ == '__main__':
    ## PREPARE Model 
    path = 'OpenGVLab/InternVL2_5-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    
    ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V")['train']

    sample = ds[0:15]

    PATH = "./data/version/3/generated_caption.txt"

    with open(PATH, "r") as f:
        data = f.read()
    f.close()

    # PREPARE INSTRUCTION
    total_instruct = data.split("==================================")
    total_instruct = total_instruct[0:len(total_instruct)-1]
    print(len(total_instruct))

    free_text_instruct = []
    structured_instruct = []
    for ins in total_instruct:
        if "```json" in ins:
            structured_instruct.append(ins)
        else:
            free_text_instruct.append(ins)
    print("Total free text instruction: {}".format(len(free_text_instruct)))
    print("Total structured instruction: {}".format(len(structured_instruct)))

    ## demo 1 samples 
    # seed_instruct = random.sample(total_instruct, 1)[0]
    # seed_sample_img = sample['image'][0]
    # url_image = "http://images.cocodataset.org/train2017/" + seed_sample_img.split("/")[-1]
    # im = Image.open(requests.get(url_image, stream=True).raw)

    # pixel_values = load_image(im, max_num=12).to(torch.bfloat16).cuda()
    # generation_config = dict(max_new_tokens=2048, do_sample=True, temperature=0.7)

    # question = '<image>\n{}'.format(seed_instruct)
    # # print(question)
    # print(url_image)
    # response = model.chat(tokenizer, pixel_values, question, generation_config)
    # print(f'User: {question}\nAssistant: {response}')

    ## Demo 
    # results = []
    # for i in trange(0, len(sample['image'])):
    #     seed_instruct = random.sample(total_instruct, 1)[0]
    #     seed_sample_img = sample['image'][i]
    #     url_image = "http://images.cocodataset.org/train2017/" + seed_sample_img.split("/")[-1]
    #     im = Image.open(requests.get(url_image, stream=True).raw)

    #     sharegpt4v_res = sample['conversations'][i][1]['value']
    #     response = generate_description(seed_instruct, im)

    #     results.append({
    #         "instruction": seed_instruct,
    #         "image_url": url_image,
    #         "response": response,
    #         "gpt": sharegpt4v_res
    #     })

    # with open("sample_results.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)
    # f.close()
    dcscore_evaluator = DCScore('sentence-transformers/all-mpnet-base-v2')

    results = []
    for i in trange(0, len(sample['image'])):
        seed_free_text_instruct = random.sample(free_text_instruct, 100)
        seed_structured_instruct = random.sample(structured_instruct, 100)

        seed_sample_img = sample['image'][i]
        url_image = "http://images.cocodataset.org/train2017/" + seed_sample_img.split("/")[-1]
        im = Image.open(requests.get(url_image, stream=True).raw)

        sharegpt4v_res = sample['conversations'][i][1]['value']

        caption_free_text = []
        caption_structured = []

        for r_ins in tqdm(seed_free_text_instruct):
            response = generate_description(r_ins, im, model, tokenizer)
            caption_free_text.append(response)
        
        for r_ins in tqdm(seed_structured_instruct):
            response = generate_description(r_ins, im, model, tokenizer)
            caption_structured.append(response)
        
        # embeddings_free_text_inst = dcscore_evaluator.get_embedding(seed_free_text_instruct, batch_size=4)[0]
        # dcs_score_free_text_inst = dcscore_evaluator.calculate_dcscore_by_embedding(embeddings_free_text_inst, kernel_type="cs", tau=1)

        # embeddings_structured_inst = dcscore_evaluator.get_embedding(seed_structured_instruct, batch_size=4)[0]
        # dcs_score_structured_inst = dcscore_evaluator.calculate_dcscore_by_embedding(embeddings_structured_inst, kernel_type="cs", tau=1)

        # embeddings_free_text_caption = dcscore_evaluator.get_embedding(caption_free_text, batch_size=4)[0]
        # dcs_score_free_text_caption = dcscore_evaluator.calculate_dcscore_by_embedding(embeddings_free_text_caption, kernel_type="cs", tau=1)

        # embeddings_structured_caption = dcscore_evaluator.get_embedding(caption_structured, batch_size=4)[0]
        # dcs_score_structured_caption = dcscore_evaluator.calculate_dcscore_by_embedding(embeddings_structured_caption, kernel_type="cs", tau=1)

        results.append({
            "free_text_instruction": seed_free_text_instruct,
            # "dcs_score_free_text_instruction": dcs_score_free_text_inst,
            "structured_instruction": seed_structured_instruct,
            # "dcs_score_structured_instruction": dcs_score_structured_inst,
            "free_text_caption": caption_free_text,
            # "dcs_score_free_text_caption": dcs_score_free_text_caption,
            "structured_caption": caption_structured,
            # "dcs_score_structured_caption": dcs_score_structured_caption,
            "image_url": url_image,
            "response": response,
            "gpt": sharegpt4v_res
        })

    with open("sample_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    f.close()
