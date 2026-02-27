
import openai
import time
from tqdm import tqdm
import argparse
import os
import re
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.multi_modal import OfaPreprocessor
# from faithscore.llava15 import LLaVA
# from faithscore.llama_pre import load_llama, stage1_llama
# from faithscore.utils import llava15, ofa
from dotenv import load_dotenv
import nltk
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from sklearn.metrics import cohen_kappa_score, accuracy_score

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
from scipy.stats import spearmanr, kendalltau, pearsonr

import logging
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

path = os.path.dirname(__file__)
cur_path = os.path.dirname(path)
cur_path = os.path.join(cur_path, "faithscore")

load_dotenv()

def llava15(image_path, prompt, model):
    return model.eval(image_path, prompt)

client = openai.AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("API_VERSION")
)

def query2(user_prompt, temp=0.3, max_token=512):    
    response = client.chat.completions.create(
        model=os.getenv("GPT_MODEL"),
        messages=user_prompt,
        max_completion_tokens=max_token,
        temperature=temp,
    )
    text = response.choices[0].message.content
    return text

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class LLaVA:
    def __init__(self, model_path="/home/lxj220018/llava15/llava/eval/checkpoints/llava-v1.5-13b"):
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)

    def eval(self, image_file, qs):
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(image_file)
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        # input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        # outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # print(outputs)
        # raise Exception
        # outputs = outputs.strip()
        # if outputs.endswith(stop_str):
        #     outputs = outputs[:-len(stop_str)]
        # outputs = outputs.strip()
        return outputs
    

class FaithScore():
    def __init__(self, vem_type, api_key=None, llava_path=None, tokenzier_path=None, use_llama=False, llama_path=None):
        openai.api_key = api_key
        max_seq_len = 500
        max_batch_size = 1
        self.use_llama = use_llama

        # self.vem_path = model_path
        self.model_type = vem_type ### [ofa_ve, ofa, mplug, blip2, llava]
        model_list = ["ofa_ve", "ofa", "mplug", "blip2", "llava"]
        if vem_type not in model_list:
            print(f"Error: the model type {vem_type} not in {str(model_list)}")
            exit()
        self.llava_path = llava_path
        
        if use_llama:
            if llama_path:
                self.llama, self.tokenizer = load_llama(llama_path)
            else:
                print(f"Error: please input the model path for llama")
                exit()

    def call_openai(self, pts):
        while True:
            try:
                # response = openai.ChatCompletion.create(
                #     model="gpt-3.5-turbo",
                #     messages=[
                #         {"role": "user",
                #          "content": pts},
                #     ],
                #     temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                # )
                # return response['choices'][0]['message']['content']
                messages=[
                    {"role": "user",
                        "content": pts},
                ]
                return query2(messages, temp=0.3)
            except Exception as e:
                print(e)
                print("Continue......")
                time.sleep(10)

    def stage1(self, answers):
        with open("./faith_score_prompts/prompt_label_des_ana.txt", "r") as f:
            prompt_label_des_ana = f.read() + "\n\n"
        des_ana = []
        print("Stage 1: Sub-sentence Identification")
        # if self.use_llama:
        #     des_ana = stage1_llama(self.llama, self.tokenizer, [a.replace("\n", " ") for a in answers])
        # else:
        for id in tqdm(range(len(answers))):
            if not self.use_llama:
                pts = prompt_label_des_ana + answers[id].replace("\n", " ") + "\n" + "Labeled text: "
                des_ana.append(self.call_openai(pts).replace("\n", ""))
            # else:
            #     pts = stage1_llama(self.llama, self.tokenizer, answers[id].replace("\n", " "))
            #     # print(pts)
            #     des_ana.append(pts)
            # exit()
        return des_ana

    def stage2(self, labeld_sub_sen, ):
        # print(labeld_sub_sen)
        all_texts = []
        labeld_sub_sen = labeld_sub_sen[0].split(".")
        lens = [len(subs) for subs in labeld_sub_sen]
        # labeld_sub_sen = [s for subs in labeld_sub_sen for s in subs]
        
        for ss in labeld_sub_sen:
            desc = ""
            pos_des = [substr.start() for substr in re.finditer("[D]", ss)]
            pos_ana = [substr.start() for substr in re.finditer("[A]", ss)]
            pos_seg = pos_des + pos_ana
            pos_seg.sort()
            for i in range(len(pos_seg)):
                if pos_seg[i] in pos_des:
                    if i == 0:
                        desc += ss[:pos_seg[i] - 1]
                    else:
                        desc += ss[pos_seg[i - 1] + 3:pos_seg[i] - 1]
            all_texts.append(desc.replace("\n", " "))

        # print(all_texts)
        # raise Exception
        with open("./faith_score_prompts/prompt_de_atomic.txt", 'r') as f:
            prompt_de_atomic = f.read()
        Entities = []
        Relations = []
        Colors = []
        Counting = []
        Others = []

        results = []
        nons = "Entities:\nRelations:\nColors:\nCounting:\nOther attributes:"
        print("Stage 2: Atomic Fact Generation")
        inputs = []
        for ans in tqdm(all_texts):
            ans = ans.replace("\n", " ")
            pts = prompt_de_atomic + "\nAnswer: " + ans
            inputs.append(pts)
        # if self.use_llama:
        #     response = stage2_llama(self.llama, self.tokenizer, inputs)
        # else:
        # response = self.call_openai(inputs)
        response = [self.call_openai(inp) for inp in inputs]
        for idx, r in enumerate(response):
            if all_texts[idx] == "" or "Entities" not in r:
                response[idx] = nons

        results = response 
        results = [r.replace("**", "") for r in results]
        # print(results)
        # raise Exception
        for i,facts in enumerate(results):
            lines = facts.split("\n")
            # print(lines)
            entity_seen = False
            for line in lines:
                if line == "":
                    continue
                if line[:9] == "Entities:":
                    if entity_seen:
                        break
                    entity_seen=True
                    entity = line.strip().replace("Entities: ", "").split(". ")
                    if line.strip() == "Entities:":
                        entity = []
                    # print(line)
                    Entities.append(entity)
                    # print(entity)
                    # raise Exception
                elif line[:10] == "Relations:":
                    # print(line.strip().replace("Relations: ","").replace("],","]],").split("], "))
                    relation = line.strip().replace("Relations: ", "").split(". ")
                    if line.strip() == "Relations:":
                        relation = []
                    Relations.append(relation)
                elif line[:7] == "Colors:":
                    color = line.strip().replace("Colors: ", "").split(". ")
                    if line.strip() == "Colors:":
                        color = []
                    Colors.append(color)
                elif line[:9] == "Counting:":
                    count = line.strip().replace("Counting: ", "").split(". ")
                    if line.strip() == "Counting:":
                        count = []
                    Counting.append(count)
                elif line[:17] == "Other attributes:":
                    other = line.strip().replace("Other attributes: ", "").split(". ")
                    if line.strip() == "Other attributes:":
                        other = []
                    Others.append(other)
            for l in [Entities, Relations, Colors, Counting, Others]:
                if len(l) < i+1:
                    for i in range(i-len(l)+1):       
                        l.append([])
        
        unflattened_Entities = []
        unflattened_Relations = []
        unflattened_Colors = []
        unflattened_Counting = []
        unflattened_Others = []
        index = 0
        for length in lens:
            unflattened_Entities.append(Entities[index:index+length])
            unflattened_Relations.append(Relations[index:index+length])
            unflattened_Colors.append(Colors[index:index+length])
            unflattened_Counting.append(Counting[index:index+length])
            unflattened_Others.append(Others[index:index+length])
            index += length
        Entities = unflattened_Entities
        Relations = unflattened_Relations
        Colors = unflattened_Colors
        Counting = unflattened_Counting
        Others = unflattened_Others
        # print(Entities)
        # print(Relations)
        # print(Colors)
        # print(Counting)
        # print(Others)
        # raise Exception
        hallucinations = [Entities[i] + Relations[i] + Colors[i] + Counting[i] + Others[i] for i in range(len(Entities))]
        return hallucinations, Entities, Relations, Colors, Counting, Others

    def stage3(self, atomic_facts, images, model, img_path=None):
        org_img_path = images
        # ofa_pipe = pipeline(Tasks.visual_entailment, model='damo/ofa_visual-entailment_snli-ve_large_en')
        # model = pipeline(Tasks.visual_entailment, model=self.vem_path)
        # print(atomic_facts)
        # raise Exception
        print("Stage 3: Verification")
        if self.model_type == "ofa_ve":
            model = pipeline(Tasks.visual_entailment, model='damo/ofa_visual-entailment_snli-ve_large_en')

        # if self.model_type == "ofa":
        #     preprocessor = OfaPreprocessor(model_dir="damo/ofa_visual-question-answering_pretrain_large_en")
        #     model = pipeline(
        #         Tasks.visual_question_answering,
        #         model="damo/ofa_visual-question-answering_pretrain_large_en",
        #         model_revision='v1.0.1',
        #         preprocessor=preprocessor, 
        #         batch_size=8)

        if self.model_type == "llava":
            if not self.llava_path:
                print("Please input path for LLaVA model.")
                exit()
            # model = LLaVA(model_path="liuhaotian/llava-v1.5-13b")
        # if self.model_type == "mplug":
        #     output = mplug(image, prompt, model)
        # if self.model_type == "blip2":
        #     output = blip_2(image, prompt, model, vis_processors_blip_2)
        
        fact_scores = []
        atomic_facts = [[f for f in sublist if f != []] for sublist in atomic_facts]
        lengths = [len([s for s in sublist if s != []]) for sublist in atomic_facts]
        lengths_2 = [[len(s) for s in sublist if s != []] for sublist in atomic_facts]
        flatten_attomic_facts = [item for sublist in atomic_facts for item in sublist if item != []]
        flatten_attomic_facts = [item for sublist in flatten_attomic_facts for item in sublist]
        images = [[images[i]]*sum(lengths_2[i]) for i in range(len(images))]
        flattened_images = [item for sublist in images for item in sublist]
        BS = 16
        for idx in tqdm(range(0,len(flatten_attomic_facts), BS)):
            facts = flatten_attomic_facts[idx:idx+BS]
            if img_path:
                image = [os.path.join(img_path, flattened_images[j]) for j in range(idx, min(idx+BS, len(flattened_images)))]
            else:
                image = [flattened_images[j] for j in range(idx, min(idx+BS, len(flattened_images)))]
                
            prompts = []
            for element in facts:
                # input = {'image': image, 'text': element}
                prompts.append('Statement: ' + element + ' Is this statement is right according to the image? Please answer yes or no.')
            # if self.model_type == "ofa_ve":
            #     output = ofa(True, model, prompts, image)
            # if self.model_type == "ofa":
            #     output = ofa(False, model, prompts, image)
            if self.model_type == "llava":
                # print(prompts)
                # print(org_img_path)
                # raise Exception
                output = [llava15(org_img_path[0], prompt, model) for prompt in prompts]
                # print(output)
                # raise Exception
                # if self.model_type == "mplug":
                #     output = mplug(image, prompt, model)
                # if self.model_type == "blip2":
                #     output = blip_2(image, prompt, model, vis_processors_blip_2)
            for out in output:
                if "yes" in out.lower():
                    fact_scores.append(1)
                else:
                    fact_scores.append(0)
                # fact_scores.append(fact_score)
                # results[id] = sum(fact_score)/len(fact_score) if len(fact_score) > 0 else 0
            # print(output)
            # print(fact_scores)
            # # output = ofa_pipe(input)[0]
            # if "yes" in output.lower():
            #     fact_score.append(1)
            # else:
            #     fact_score.append(0)

            #     # if output.lower() == "yes" or output== "Yes":
            #     #     fact_score.append(1)
            #     # else:
            #     #     fact_score.append(0)
            # fact_scores.append(fact_score)
            # results[id] = sum(fact_score)/len(fact_score) if len(fact_score) > 0 else 0
                # result.append(output[OutputKeys.LABELS])
            # results.append({"image": images_id[id], "facts": elements, "result": str(result)})
            # checking_results.append(result)
        unflattented_fact_scores = []
        results = {}
        index = 0
        for i,length in enumerate(lengths_2):
            unflattented_fact_scores.append(fact_scores[index:index+sum(length)])
            results[i] = sum(unflattented_fact_scores[i])/sum(length) if sum(length) > 0 else 0

            index += sum(length)
            
        # unflattented_fact_scores = []
        # index = 0
        # for i, length in enumerate(lengths):
        #     unflattented_fact_scores.append(unflattented_fact_scores_2[index:index+length])
        #     index += length
            instance_score = [sum([iii for iii in ii]) / len([iii for iii in ii]) if len([iii for iii in ii]) > 0 else 0 for ii in unflattented_fact_scores]

        # instance_score = [sum([iiii for iii in ii for iiii in iii]) / len([iiii for iii in ii for iiii in iii]) if len([iiii for iii in ii for iiii in iii]) > 0 else 0 for ii in unflattented_fact_scores]
        # print("Overall score: ", sum(instance_score) / len(instance_score))

        return sum(instance_score) / len(instance_score), unflattented_fact_scores, results
    '''
    answers: a list of strings, each element in this list is an answer
    '''

    def faithscore(self, answers, images, model_llava):
        ## Stage 1: Sub-setence Identification
        labeld_sub_sen = self.stage1(answers)
        ### Stage 2: Atomic Fact Generation
        atomic_facts, Entities, Relations, Colors, Counting, Others = self.stage2(labeld_sub_sen)
        ### Stage 3: Verification
        # print(atomic_facts)
        score, fact_scores, results = self.stage3(atomic_facts, images, model_llava)
        # sentence_score, results_sentence = self.sentence_faithscore(Entities, Relations, Colors, Counting, Others, self.labeled_sub(labeld_sub_sen), fact_scores)
        return score, 0, results, 0

    def sentence_faithscore(self, Entities, Relations, Colors, Counting, Others, all_texts, fact_scores):
        Entities_recog = []
        for ents in Entities:
            entities = []
            ents = [ee for e in ents for ee in e ]
            for e in ents:
                ent4sen = []
                sentence = nltk.sent_tokenize(e)
                tags = nltk.pos_tag(nltk.word_tokenize(sentence[0]))
                for tag in tags:
                    if tag[1] in ['NN', 'NNS', 'JJ', 'NNP', 'VBG', 'JJR', 'NNPS', 'RB', 'DT']:
                        # print(tag)
                        ent4sen.append(tag[0])
                    else:
                        ent4sen.append("")
                    # tags.append(chunk.label())

                if len(ent4sen) < 1:
                    print(tags)
                    ent4sen.append("")
                    # exit()
                
                entities.append(ent4sen[-1])


            if len(entities) != len(ents):
                print("error")
                exit()
            Entities_recog.append(entities)

        entity_scores = []
        relation_scores = []
        color_scores = []
        count_scores = []
        other_scores = []
        
        clean_empty = lambda x: [ii for i in x for ii in i if i]
        Entities = [clean_empty(e) for e in Entities]
        Relations = [clean_empty(r) for r in Relations]
        Colors = [clean_empty(c) for c in Colors]
        Counting = [clean_empty(c) for c in Counting]
        Others = [clean_empty(o) for o in Others]
        

        for i in range(len(fact_scores)):
            entity_scores.append(fact_scores[i][:len(Entities[i])])
            relation_scores.append(fact_scores[i][len(Entities[i]): len(Entities[i]) + len(Relations[i])])
            color_scores.append(fact_scores[i][
                                len(Entities[i]) + len(Relations[i]):  len(Entities[i]) + len(Relations[i]) + len(
                                    Colors[i])])
            count_scores.append(fact_scores[i][
                                len(Entities[i]) + len(Relations[i]) + len(Colors[i]):  len(Entities[i]) + len(
                                    Relations[i]) + len(Colors[i]) + len(Counting[i])])
            other_scores.append(
                fact_scores[i][len(Entities[i]) + len(Relations[i]) + len(Colors[i]) + len(Counting[i]):])

        sentence_scores = []
        results = {}
        
        for id1, ins in enumerate(all_texts):
            sentence_score = []
            for id2, sub_sen in enumerate(all_texts[id1]):
                flag = True
                for id3, ee in enumerate(Entities_recog[id1]):
                    if ee in sub_sen and entity_scores[id1][id3] != 1:
                        flag = False
                    for id4, rel in enumerate(relation_scores[id1]):
                        if ee in sub_sen and ee in Relations[id1][id4] and rel != 1:
                            flag = False
                    for id4, rel in enumerate(color_scores[id1]):
                        if ee in sub_sen and ee in Colors[id1][id4] and rel != 1:
                            flag = False
                    for id4, rel in enumerate(count_scores[id1]):
                        if ee in sub_sen and ee in Counting[id1][id4] and rel != 1:
                            flag = False
                    for id4, rel in enumerate(other_scores[id1]):
                        if ee in sub_sen and ee in Others[id1][id4] and rel != 1:
                            flag = False

                sentence_score.append(flag)
            sentence_scores.append(sentence_score)
            results[id1] = sum(sentence_score)/len(sentence_score) if len(sentence_score) > 1 else sentence_score


        score4sen = [sum(ss)/len(ss) if len(ss) > 0 else 1 for ss in sentence_scores]
        sentence_level_score = score4sen
        # print(score4sen)
        # print(sum(score4sen)/len(score4sen))
        return sum(score4sen)/len(score4sen), results

    def labeled_sub(self, des_ana):
        all_texts = []
        for ss in des_ana:
            desc = []
            for sss in ss:
                pos_des = [substr.start() for substr in re.finditer("[D]", sss)]
                pos_ana = [substr.start() for substr in re.finditer("[A]", sss)]
                pos_seg = pos_des + pos_ana
                pos_seg.sort()
                for i in range(len(pos_seg)):
                    if pos_seg[i] in pos_des:
                        if i == 0:
                            desc.append(sss[:pos_seg[i] - 1])
                        else:
                            desc.append(sss[pos_seg[i - 1] + 3:pos_seg[i] - 1])
            all_texts.append(desc)
        return all_texts
    
    
if __name__ == '__main__':
    # # images = ["./Dataset/flickr30k-images/4016499639.jpg"]
    # # answers = ["The image depicts a food cart situated on a city street, prominently displaying the sign \"DISABLED COMBAT VETERAN\" above a colorful array of food options. The cart is adorned with two yellow umbrellas, each featuring the words \"FRANKFURTERS & ROLLS\" and \"SABRET,\" indicating the type of food served. In the foreground, a woman with a black backpack and a pink strap stands facing the cart, while a man in a dark jacket appears to be ordering food. The cart showcases a variety of items, including hot dogs, drinks, and snacks, with glass containers filled with beverages visible behind the vendor. The background features a blurred view of a bus and trees, suggesting a bustling urban environment."]


    # # with open(args.answer_path) as f:
    # #     for line in f:
    # #         line = eval(line)
    # #         answers.append(line["answer"])
    # #         images.append(line["image"])

    # score = FaithScore(vem_type="llava", api_key=os.getenv("API_KEY"),
    #                    llava_path="liuhaotian/llava-v1.5-13b", use_llama=False,
    #                    llama_path=None)
    # model = LLaVA(model_path="liuhaotian/llava-v1.5-13b")
    # # f, sentence_f, results, _ = score.faithscore(answers, images)
    # # print(f"Faithscore is {f}. Sentence-level faithscore is {sentence_f}.")
    
    # # print("-----")
    # # print(results)
    
    # with open("./image_db_spec_score_new_ver10.json", "r") as f:
    #     data = json.load(f)
    # f.close()
    
    # new_data = data[0:100]
    # new_results = []
    # for d in tqdm(new_data):
    #     for k, v in d.items():
    #         faith_score_gpt = score.faithscore([v['gpt4o']], ["./Dataset/flickr30k-images/{}.jpg".format(k)], model)[0]
    #         faith_score_blip = score.faithscore([v['instruct_blip']], ["./Dataset/flickr30k-images/{}.jpg".format(k)], model)[0]
    
    #         new_results.append({
    #             "img": k,
    #             "gpt4": v['gpt4o'],
    #             "blip": v['instruct_blip'],
    #             "speci_gpt4": v['score_gpt4'],
    #             "spceci_blip": v['score_blip'],
    #             "faith_score_gpt": faith_score_gpt,
    #             "faith_score_blip": faith_score_blip
    #         })
    
    # with open('./faith_score.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_results, f, ensure_ascii=False, indent=4)
    # f.close()
    
    
    with open("./faith_score.json", "r") as f:
        data = json.load(f)
    f.close()
    
    speci_data = []
    faith_data = []
    
    diff_spec = []
    diff_cap = []
    
    win_count = 0
    
    for d in data:
        if d['speci_gpt4'] >= d['spceci_blip'] and d['faith_score_gpt'] >= d['faith_score_blip']:
            win_count = win_count + 1
        
        if d['faith_score_gpt'] >= d['faith_score_blip']:
            diff_cap.append(1)
        else:
            diff_cap.append(0)
    
        if d['speci_gpt4'] >= d['spceci_blip']:
            diff_spec.append(1)
        else:
            diff_spec.append(0)
            
        speci_data.append(d['speci_gpt4'] - d['spceci_blip'])
        faith_data.append(d['faith_score_gpt'] - d['faith_score_blip'])
    
    print("Win rate: {}".format((win_count / len(data))))
    
    acc = accuracy_score(diff_spec, diff_cap)
    print(f"Acc: {acc}")
    
    kappa = cohen_kappa_score(diff_spec, diff_cap)
    print(f"Kappa: {kappa}")
    
    print("Spearman Corr")
    correlation, p_value = spearmanr(speci_data, faith_data)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(speci_data, faith_data)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(speci_data, faith_data)
    print(correlation)
    # print(p_value)
    print("===========")
            
    
        