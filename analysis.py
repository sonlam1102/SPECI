from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import logging
import seaborn as snb
from nltk.tokenize import sent_tokenize, word_tokenize
from datasets import load_dataset
import nltk
# import torch
import json
# from dcs_score import DCScore
from scipy.stats import spearmanr, kendalltau, pearsonr

def make_wordcloud(data, name="wordcloud"):
    stopwords = set(STOPWORDS)
    stopwords.update(["object", "image", "description", "detailed", "e.g.", "etc", "subject", "JSON"])

    full_text = ' '.join(ins for ins in data)
    wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white', stopwords=stopwords)
    wordcloud.generate(full_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # plt.show()
    plt.savefig('{}.png'.format(name))


def make_len_distribution(data):
    total_instruct_len = [len(word_tokenize(ins)) for ins in data]
    sns_plot = snb.displot(x=total_instruct_len, bins=4)
    # sns_plot = snb.countplot(x=total_instruct_len)
    fig = sns_plot.figure
    fig.savefig("len_dist.png")


def analyze_postag(data):
    lst_result = []
    for d in data:
        text = word_tokenize(d)
        pos_tagged = nltk.pos_tag(text)
        if len(lst_result) == 0:
            lst_result = pos_tagged
        else:
            lst_result = lst_result + pos_tagged
    
    return lst_result


def analyze_score_corr(data, dname="iiw"):
    spec_score = []
    hallu_score = []
    
    if dname == "iiw":
        mapping_dict = {
            "IIW-Human is substantially better": 2,
            "IIW-Human is marginally better": 1,
            "Neutral": 0,
            "IIW-P5B is marginally better": -1,
            "IIW-P5B is substantially better": -2
        }
        for d in data:
            spec_score.append(mapping_dict[d['iiw-human-sxs-iiw-p5b']['metrics/Specificity']])
            hallu_score.append(mapping_dict[d['iiw-human-sxs-iiw-p5b']['metrics/Hallucination']])
        
    elif dname == "dci":
        mapping_dict = {
            "IIW is substantially better": 2,
            "IIW is marginally better": 1,
            "Neutral": 0,
            "DCI is marginally better": -1,
            "DCI is substantially better": -2
        }
        for d in data:
            spec_score.append(mapping_dict[d['metrics/Specificity']])
            hallu_score.append(mapping_dict[d['metrics/Hallucination']])
    elif dname == "docci":
        mapping_dict = {
            "IIW is substantially better": 2,
            "IIW is marginally better": 1,
            "Neutral": 0,
            "DOCCI is marginally better": -1,
            "DOCCI is substantially better": -2
        }
        for d in data:
            spec_score.append(mapping_dict[d['metrics/Specificity']])
            hallu_score.append(mapping_dict[d['metrics/Hallucination']])
    else:
        return -1
    
    print("Spearman Corr")
    correlation, p_value = spearmanr(spec_score, hallu_score)
    print(correlation)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(spec_score, hallu_score)
    print(correlation)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(spec_score, hallu_score)
    print(correlation)
    print("===========")


if __name__ == '__main__':
    # PATH = "./data/version/3/generated_caption.txt"

    # with open(PATH, "r") as f:
    #     data = f.read()

    # total_instruct = data.split("==================================")
    # total_instruct = total_instruct[0:len(total_instruct)-1]

    # print(len(total_instruct))

    # # ## Analyze the token in instruction
    # free_text_instruct = []
    # structured_instruct = []
    # for ins in total_instruct:
    #     if "```json" in ins:
    #         structured_instruct.append(ins)
    #     else:
    #         free_text_instruct.append(ins)

    # print("Total free text instruction: {}".format(len(free_text_instruct)))
    # print("Total structured instruction: {}".format(len(structured_instruct)))

    # make_wordcloud(total_instruct, name="wordcloud")

    # pos_tags = analyze_postag(total_instruct)

    # lst_noun = [p[0] for p in pos_tags if p[1] in ['NN', 'NNS', 'NNP', 'PRP']]
    # lst_verb = [p[0] for p in pos_tags if p[1] == 'VB']

    # make_wordcloud(lst_noun, name="wordcloud_noun")
    # make_wordcloud(lst_verb, name="wordcloud_verb")

    # ## Computing diversity 
    # model_path_dict = {
    #                 'bert':'google-bert/bert-base-uncased',
    #                 'simcse':'princeton-nlp/unsup-simcse-bert-base-uncased',
    #                 'llama2-7b':'meta-llama/Llama-2-7b-hf',
    #                 'gpt2':'openai-community/gpt2',
    #                 'bge': 'BAAI/bge-large-en-v1.5',
    #                 'sen_bert': 'sentence-transformers/all-mpnet-base-v2'
    #               }
    
    # model_name = 'sen_bert'
    # model_path = model_path_dict[model_name]
    # device = "cuda" if torch.cuda.is_available() else "CPU"
    # batch_size = 128
    # tau = 1
    # kernel_type = 'cs'

    # # evaluated dataset
    # text_list = free_text_instruct

    # # dcscore class
    # dcscore_evaluator = DCScore(model_path)

    # # get embedding
    # embeddings, n, d = dcscore_evaluator.get_embedding(text_list, batch_size=batch_size)

    # # calculate dcscore based on embedding
    # dataset_dcscore = dcscore_evaluator.calculate_dcscore_by_embedding(embeddings, kernel_type=kernel_type, tau=tau)

    # print(dataset_dcscore)
    
    iiw = []
    with open("./dataset/iiw_new.jsonl", 'r') as f:
        for line in f:
            try:
                iiw.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()
    print("===============")
    analyze_score_corr(iiw, dname="iiw")
    dci = []
    with open("./dataset/dci_new.jsonl", 'r') as f:
        
        for line in f:
            try:
                dci.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()
    print("===============")
    analyze_score_corr(dci, dname="dci")
    docci = []
    with open("./dataset/docci.jsonl", 'r') as f:
        
        for line in f:
            try:
                docci.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()
    print("===============")
    analyze_score_corr(docci, dname="docci")
