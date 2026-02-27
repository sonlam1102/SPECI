import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.ticker as ticker
import pandas as pd

def make_scatter_len_spec(data, dtype="docci"):
    if dtype == "dci":
        source = 'IIW'
        target = 'DCI_extra_caption'
        label = {
            "{} is substantially better".format(source): 2,
            "{} is marginally better".format(source): 1,
            "Neutral": 0,
            "{} is marginally better".format("DCI"): -1,
            "{} is substantially better".format("DCI"): -2
        }
    elif dtype == 'docci':
        source = 'IIW'
        target = 'DOCCI'
        label = {
            "{} is substantially better".format(source): 2,
            "{} is marginally better".format(source): 1,
            "Neutral": 0,
            "{} is marginally better".format(target): -1,
            "{} is substantially better".format(target): -2
        }
    elif dtype == 'iiw':
        source = 'IIW'
        target = 'IIW-P5B'
        label = {
            "{} is substantially better".format("IIW-Human"): 2,
            "{} is marginally better".format("IIW-Human"): 1,
            "Neutral": 0,
            "{} is marginally better".format("IIW-P5B"): -1,
            "{} is substantially better".format("IIW-P5B"): -2
        }
    else:
        return -1
    
    color_map = {
        2: 'red',
        1: 'blue',
        0: 'green',
        -1: 'yellow',
        -2: 'purple'
    }

    len_source = []
    len_target = []
    len_label = []
    len_ratio = []
    for d in data:
        len_source.append(len(word_tokenize(d[source])))
        len_target.append(len(word_tokenize(d[target])))
        len_label.append(label[d['iiw-human-sxs-iiw-p5b']['metrics/Specificity']] if dtype == "iiw" else label[d['metrics/Specificity']])
        len_ratio.append(float(len(word_tokenize(d[source]))/len(word_tokenize(d[target]))))
    
    colors = [color_map[cat] for cat in len_label]
    
    plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(len_source, len_target, c=colors, s=50, alpha=0.7)
    # legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
    #                          markerfacecolor=color, markersize=10)
    #               for label, color in color_map.items()]
    # plt.legend(handles=legend_handles, title="Specificity by Humans")

    # plt.xlabel("Source caption length by tokens")
    # plt.ylabel("Target caption length by tokens")
    # plt.grid(True)
    # plt.savefig('{}_scatter.png'.format(dtype))

    # print("Max length source: {}".format(max(len_source)))
    # print("Mean length source: {}".format(statistics.mean(len_source)))
    # print("Min length source: {}".format(min(len_source)))
    
    # print("Max length target: {}".format(max(len_target)))
    # print("Mean length target: {}".format(statistics.mean(len_target)))
    # print("Min length target: {}".format(min(len_target)))
    plot_data = {
        'len_ratio': len_ratio,
        'label': len_label
    }
    df_plot = pd.DataFrame(plot_data)
    grouped_data = [df_plot['len_ratio'][df_plot['label'] == group] for group in df_plot['label'].unique()]
    group_labels = ["2", "1", "0", "-1", "-2"]
    
    # print(grouped_data)
    
    plt.figure(figsize=(8, 6))
    
    plt.rcParams.update({'font.size': 22})
    plt.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    bp = plt.boxplot(grouped_data, patch_artist=True, medianprops={'color': 'black'})

    colors = ['blue', 'red', 'black', 'orange' , 'green']
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])

    plt.xticks(np.arange(1, len(group_labels) + 1), group_labels)
    plt.yticks([f"{label:.2f}" for label in df_plot['len_ratio']])
    plt.ylabel('Length ratio of caption A and B')
    plt.xlabel('Human specificity label')
    plt.grid(axis='y', linestyle='--')
    plt.savefig('{}_boxplot.png'.format(dtype), bbox_inches='tight')
    
    
def make_scatter_len_hallu(data, dtype="docci"):
    if dtype == "dci":
        source = 'IIW'
        target = 'DCI_extra_caption'
        label = {
            "{} is substantially better".format(source): 2,
            "{} is marginally better".format(source): 1,
            "Neutral": 0,
            "{} is marginally better".format("DCI"): -1,
            "{} is substantially better".format("DCI"): -2
        }
    elif dtype == 'docci':
        source = 'IIW'
        target = 'DOCCI'
        label = {
            "{} is substantially better".format(source): 2,
            "{} is marginally better".format(source): 1,
            "Neutral": 0,
            "{} is marginally better".format(target): -1,
            "{} is substantially better".format(target): -2
        }
    elif dtype == 'iiw':
        source = 'IIW'
        target = 'IIW-P5B'
        label = {
            "{} is substantially better".format("IIW-Human"): 2,
            "{} is marginally better".format("IIW-Human"): 1,
            "Neutral": 0,
            "{} is marginally better".format("IIW-P5B"): -1,
            "{} is substantially better".format("IIW-P5B"): -2
        }
    else:
        return -1
    
    color_map = {
        2: 'red',
        1: 'blue',
        0: 'green',
        -1: 'yellow',
        -2: 'purple'
    }

    len_source = []
    len_target = []
    len_label = []
    len_ratio = []
    for d in data:
        len_source.append(len(word_tokenize(d[source])))
        len_target.append(len(word_tokenize(d[target])))
        len_label.append(label[d['iiw-human-sxs-iiw-p5b']['metrics/Specificity']] if dtype == "iiw" else label[d['metrics/Specificity']])
        len_ratio.append(float(len(word_tokenize(d[source]))/len(word_tokenize(d[target]))))
    
    colors = [color_map[cat] for cat in len_label]
    
    plt.figure(figsize=(8, 6))
    plot_data = {
        'len_ratio': len_ratio,
        'label': len_label
    }
    df_plot = pd.DataFrame(plot_data)
    grouped_data = [df_plot['len_ratio'][df_plot['label'] == group] for group in df_plot['label'].unique()]
    group_labels = ["2", "1", "0", "-1", "-2"]
    
    # print(grouped_data)
    
    plt.figure(figsize=(8, 6))
    
    plt.rcParams.update({'font.size': 22})
    # plt.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    bp = plt.boxplot(grouped_data, patch_artist=True, medianprops={'color': 'black'})

    colors = ['blue', 'red', 'black', 'orange' , 'green']
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])

    plt.xticks(np.arange(1, len(group_labels) + 1), group_labels)
    plt.yticks([f"{label:.2f}" for label in df_plot['len_ratio']])
    plt.ylabel('Length ratio of caption A and B')
    plt.xlabel('Human hallucination label')
    plt.grid(axis='y', linestyle='--')
    plt.savefig('{}_boxplot.png'.format(dtype), bbox_inches='tight')


def make_compare(data, caption_a, caption_b, score_det):
    for d in data:
        # d['length_a'] = len(d[caption_a].split())
        d['length_a'] = len(word_tokenize(d[caption_a]))
        # d['length_b'] = len(d[caption_b].split())
        d['length_b'] = len(word_tokenize(d[caption_b]))
        
        d['length_diff'] = d['length_a'] - d['length_b']
        d['score_diff'] = d['score_iiw'] - d[score_det]
    
    return data

def make_plot(data):
    len_value = []
    score_value = []
    for d in data: 
        len_value.append(d['length_diff'])
        score_value.append(d['score_diff'])

    scaler = MinMaxScaler(feature_range=(1, 10), clip=True)
    new_len_value = scaler.fit_transform(np.array(len_value).reshape(-1, 1)).flatten().tolist()
    new_score_value = scaler.fit_transform(np.array(score_value).reshape(-1, 1)).flatten().tolist()
    
    
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_len_value, new_score_value)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_len_value, new_score_value)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_len_value, new_score_value)
    print(correlation)
    # print(p_value)
    print("===========")
    
    
    # x = np.array([i for i in range (0, len(data))])
    # # Y-axis values for the first line
    # y1 = np.array(new_len_value)
    # # Y-axis values for the second line
    # y2 = np.array(new_score_value)
    
    # plt.scatter(x, y1, label='Length of caption', zorder=2) # Add the first line with a label and marker
    # plt.plot(x, y2, label='SPECI score value', linestyle='--', color='red')
    # plt.legend()
    
    # plt.show()


def compare_hallucination_len(data, dtype):
    if dtype == "dci":
        source = 'IIW'
        target = 'DCI_extra'
        speci_name = "score_extra_dci"
        label = {
            "{} is substantially better".format(source): 2,
            "{} is marginally better".format(source): 1,
            "Neutral": 0,
            "{} is marginally better".format("DCI"): -1,
            "{} is substantially better".format("DCI"): -2
        }
    elif dtype == 'docci':
        source = 'IIW'
        target = 'DOCCI'
        speci_name = "score_docci"
        label = {
            "{} is substantially better".format(source): 2,
            "{} is marginally better".format(source): 1,
            "Neutral": 0,
            "{} is marginally better".format(target): -1,
            "{} is substantially better".format(target): -2
        }
    elif dtype == 'iiw':
        source = 'IIW'
        target = 'IIW-P5B'
        speci_name = "score_p5b"
        label = {
            "{} is substantially better".format("IIW-Human"): 2,
            "{} is marginally better".format("IIW-Human"): 1,
            "Neutral": 0,
            "{} is marginally better".format("IIW-P5B"): -1,
            "{} is substantially better".format("IIW-P5B"): -2
        }
    else:
        return -1
    
    diff_length = []
    diff_hallu = []
    for d in data:
        # d['length_a'] = len(d[source].split())
        # d['length_b'] = len(d[target].split())
        
        d['length_a'] = len(word_tokenize(d[source]))
        d['length_b'] = len(word_tokenize(d[target]))

        diff_length.append(d['length_a'] / d['length_b'])
        diff_hallu.append(label[d['human_hallucination']])
    
    scaler = MinMaxScaler(feature_range=(-2, 2))
    new_diff_length = scaler.fit_transform(np.array(diff_length).reshape(-1, 1)).flatten().tolist()
    
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff_length, diff_hallu)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff_length, diff_hallu)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff_length, diff_hallu)
    print(correlation)
    # print(p_value)
    print("===========")
    
    
def compare_specific_len(data, dtype):
    if dtype == "dci":
        source = 'IIW'
        target = 'DCI_extra'
        speci_name = "score_extra_dci"
        label = {
            "{} is substantially better".format(source): 2,
            "{} is marginally better".format(source): 1,
            "Neutral": 0,
            "{} is marginally better".format("DCI"): -1,
            "{} is substantially better".format("DCI"): -2
        }
    elif dtype == 'docci':
        source = 'IIW'
        target = 'DOCCI'
        speci_name = "score_docci"
        label = {
            "{} is substantially better".format(source): 2,
            "{} is marginally better".format(source): 1,
            "Neutral": 0,
            "{} is marginally better".format(target): -1,
            "{} is substantially better".format(target): -2
        }
    elif dtype == 'iiw':
        source = 'IIW'
        target = 'IIW-P5B'
        speci_name = "score_p5b"
        label = {
            "{} is substantially better".format("IIW-Human"): 2,
            "{} is marginally better".format("IIW-Human"): 1,
            "Neutral": 0,
            "{} is marginally better".format("IIW-P5B"): -1,
            "{} is substantially better".format("IIW-P5B"): -2
        }
    else:
        return -1
    
    diff_length = []
    diff_hallu = []
    for d in data:
        # d['length_a'] = len(d[source].split())
        # d['length_b'] = len(d[target].split())
        
        d['length_a'] = len(word_tokenize(d[source]))
        d['length_b'] = len(word_tokenize(d[target]))

        diff_length.append(d['length_a'] / d['length_b'])
        diff_hallu.append(label[d['human_specific']])
    
    scaler = MinMaxScaler(feature_range=(-2, 2))
    new_diff_length = scaler.fit_transform(np.array(diff_length).reshape(-1, 1)).flatten().tolist()
    
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff_length, diff_hallu)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff_length, diff_hallu)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff_length, diff_hallu)
    print(correlation)
    # print(p_value)
    print("===========")
    
    
if __name__ == '__main__':
    # iiw = []
    # with open("./original/iiw_new.jsonl", 'r') as f:
    #     for line in f:
    #         try:
    #             iiw.append(json.loads(line))
    #         except json.JSONDecodeError as e:
    #             print(f"Error decoding JSON: {e}")
    # f.close()
    # print("===============")
    
    # dci = []
    # with open("./original/dci_new.jsonl", 'r') as f:
        
    #     for line in f:
    #         try:
    #             dci.append(json.loads(line))
    #         except json.JSONDecodeError as e:
    #             print(f"Error decoding JSON: {e}")
    # f.close()
    # print("===============")
    
    # docci = []
    # with open("./original/docci.jsonl", 'r') as f:
    #     for line in f:
    #         try:
    #             docci.append(json.loads(line))
    #         except json.JSONDecodeError as e:
    #             print(f"Error decoding JSON: {e}")
    # f.close()
    
    with open("./dci_len_new.json", 'r') as f:
        data_dci = json.load(f)
    f.close()
    data_new_dci = make_compare(data_dci, 'IIW', 'DCI_extra', 'score_extra_dci')
    
    with open("./docci_len_new.json", 'r') as f:
        data_docci = json.load(f)
    f.close()
    data_new_docci = make_compare(data_docci, 'IIW', 'DOCCI', 'score_docci')

    with open("./iiw_len_new.json", 'r') as f:
        data_iiw = json.load(f)
    f.close()
    data_new_iiw = make_compare(data_iiw, 'IIW', 'IIW-P5B', 'score_p5b')
    
    
    
    # with open('./dci_test.json', 'w', encoding='utf-8') as f:
    #     json.dump(data_new_dci, f, ensure_ascii=False, indent=4)
    # f.close()
    
    # with open('./docci_test.json', 'w', encoding='utf-8') as f:
    #     json.dump(data_new_docci, f, ensure_ascii=False, indent=4)
    # f.close()
    
    # with open('./iiw_test.json', 'w', encoding='utf-8') as f:
    #     json.dump(data_new_iiw, f, ensure_ascii=False, indent=4)
    # f.close()
    
    # make_plot(data_new_iiw)
    compare_hallucination_len(data_new_dci, 'dci')
    print("-----")
    compare_specific_len(data_new_dci, 'dci')
    