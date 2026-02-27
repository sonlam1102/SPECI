import json 
from tqdm import tqdm
from misc import *

def convert_retrieval_data():
    with open("./Dataset/sub_test_sen_to_ids_300.json", "r") as f:
        data = json.load(f)
    f.close()
    
    new_json_data = []
    for k, v in data.items():
        if v['type'] == 'rawSentence':
            new_json_data.append({
                "query": k,
                "meta_data": v
            })
    
    new_data = new_json_data
    # print(json.dumps(new_data, indent=4))
    
    with open('./test_new2.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    f.close()


def make_image_db():
    processor, model, device = load_instruct_blip()
    
    dense_captioning_instruction = """
        You are an AI model that performs dense image captioning. 
        Your task is to generate a detailed, information-rich caption for a given image.

        Guidelines for the caption:
        1. Go beyond a short description—provide fine-grained details about objects, their attributes, relationships, spatial arrangement, colors, textures, and activities.
        2. Describe both foreground and background elements.
        3. Include contextual inferences when reasonable (e.g., likely setting, activity, or purpose) without speculation beyond what the image supports.
        4. Write in complete sentences at the paragraph level (not bullet points).
        5. Maintain a neutral, descriptive tone, avoiding subjective judgments.

        Instruction to Model:
        “Given the input image, generate a dense caption that thoroughly describes the visual scene, highlighting as many distinct and meaningful details as possible in one single paragraph with no more than 100 words.”
    """
    # image_caption_query = "Generate a dense caption describing all objects, attributes, relationships, and context in this image. <RESPONSE:>"
    image_caption_query = "Given the input image, generate a dense caption that thoroughly describes the visual scene, highlighting as many distinct and meaningful details as possible in one single paragraph with no more than 100 words. <RESPONSE:>"
    # image_caption_query = dense_captioning_instruction + "<RESPONSE:>"
    
    with open("./test.json", "r") as f:
        data = json.load(f)
    f.close()
    
    image_db = []
    for d in data:
        image_db = image_db + d['meta_data']['images']
        
    image_db = list(set(image_db))
    
    new_image_data = []
    for im in tqdm(image_db):
        image_path = "./Dataset/flickr30k-images/{}.jpg".format(im)
        pil_image = Image.open(image_path)
        
        gpt_4o_caption = query3(dense_captioning_instruction, pil_image)
        blip_caption = query_instruct_blip(image_caption_query, pil_image, processor, model, device).split("<RESPONSE:>")[-1]
        
        # print(gpt_4o_caption)
        # print("------")
        # print(pali_caption)
        # raise Exception
        new_image_data.append({
            im : {
                "gpt4o": gpt_4o_caption,
                "instruct_blip": blip_caption
            }
        })
    
    return new_image_data
    
if __name__ == "__main__":
    image_db = make_image_db()
    with open('./image_db_ver10.json', 'w', encoding='utf-8') as f:
        json.dump(image_db, f, ensure_ascii=False, indent=4)
    f.close()
    
    # convert_retrieval_data()
    