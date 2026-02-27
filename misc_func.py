import openai
import json
import os
import numpy as npy
from dotenv import load_dotenv
from tqdm import tqdm
import nltk
import stanza
import torch
# from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BlipForConditionalGeneration
from PIL import Image
import base64
import io
        
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
import clip
from torchmetrics.functional.multimodal import clip_score
from transformers import BlipProcessor, BlipModel, BlipForImageTextRetrieval

load_dotenv()
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
nltk.download('wordnet')  # This will download the WordNet data

def parse_chatgpt_json(response_text):
    try:
        # Attempt to parse the entire response as JSON
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, attempt to extract JSON from markdown or text
        try:
            start_index = response_text.index('{')
            end_index = response_text.rindex('}') + 1
            json_string = response_text[start_index:end_index]
            data = json.loads(json_string)
        except (ValueError, json.JSONDecodeError) as e:
             return f"Error: Could not parse JSON: {e}"
        
    return data

client = openai.AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("API_VERSION")
)

def query(system_prompt, user_prompt, temp=0.3, max_token=512):
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=messages,
        max_completion_tokens=max_token,
        temperature=temp,
    )
    text = response.choices[0].message.content
    return text


def query2(user_prompt, temp=0.3, max_token=512):
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=messages,
        max_completion_tokens=max_token,
        temperature=temp,
    )
    text = response.choices[0].message.content
    return text


def convert_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    data_uri = f"data:image/png;base64,{img_str}"
    
    return data_uri


def query3(user_prompt, pil_image, max_token=512):
    image_str = convert_image_to_base64(pil_image)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_str
                }
            }
        ]},
    ]
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=messages,
        max_completion_tokens=max_token,
        temperature=0.0,
    )
    text = response.choices[0].message.content
    return text


def query4(messages, max_token=512, top_p=0.0, temperature=1):
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=messages,
        max_completion_tokens=max_token,
        temperature=0.0,
    )
    text = response.choices[0].message.content
    return text


def load_instruct_blip():
    # model_id = "/scr/huggingface_models/instructblip-vicuna-7b"
    model_id = "Salesforce/instructblip-vicuna-7b"
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_id,
    )
    processor = InstructBlipProcessor.from_pretrained(model_id)
    
    device = "cuda"
    
    model = model.to(device)
    return processor, model, device


def query_instruct_blip(prompt, pil_image, processor, model, device):
    model_inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)
    outputs = model.generate(
        **model_inputs,
        do_sample=False,
        max_length=512,
        min_length=1,
        # temperature=0.5,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    return generated_text


def query_image_gen(caption, image_name):
    result = client.images.generate(
        model=os.getenv("GPT_MODEL"),   # you can also use "gpt-4o"
        prompt="Create an image based on this description: {}".format(caption),
        size="1024x1024"
    )
    
    if hasattr(result.data[0], "b64_json") and result.data[0].b64_json:
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # Save to local file
        with open("{}.png".format(image_name), "wb") as f:
            f.write(image_bytes)
        print("✅ Image saved as output.png")
    else:
        # If only URL is returned
        image_url = result.data[0].url
        print("⚠️ No base64 data returned. Use this URL to view the image:")
        print(image_url)


def load_stable_diffusion():
    model_id = "/scr/huggingface_models/stable-diffusion-3.5-medium"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id, 
        transformer=model_nf4,
        torch_dtype=torch.bfloat16
    )
    pipeline.enable_model_cpu_offload()
    
    return pipeline

def query_image_gen_diffusion(pipeline, caption, image_name):
    prompt="Create an image based on this caption: {}".format(caption)
    
    image = pipeline(
        prompt=prompt,
        num_inference_steps=40,
        guidance_scale=4.5,
        max_sequence_length=512,
    ).images[0]
    image.save("{}.png".format(image_name))


def load_clip(device):
    # CLIP_MODEL = "ViT-L/14"
    CLIP_MODEL = "ViT-L/14@336px"
    print("CLIP model: {}".format(CLIP_MODEL))
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()
    return model, preprocess

def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.eval()
    
    return model, processor