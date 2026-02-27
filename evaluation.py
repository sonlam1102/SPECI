import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
import math


def round_up_to_half(number):
    return math.ceil(number * 2) / 2

def normalized_score(number):
    if -2 <= number < -1.5:
        return -2
    elif -1.5 <= number <= -1:
        return -1
    elif -1 < number <= -0.5:
        return 0
    elif -0.5 < number <= 0:
        return 0
    elif 0 < number <= 1.5:
        return 1
    else:
        return 2

def corr_dci():
    with open("./dci_specific.json", 'r') as f:
        iiw_data = json.load(f)
    f.close()


    diff = []
    human = []

    main_obj = "IIW"
    target_obj = "DCI"

    mapping_dict = {
        "{} is substantially better".format(main_obj): 2,
        "{} is marginally better".format(main_obj): 1,
        "Neutral": 0,
        "{} is marginally better".format(target_obj): -1,
        "{} is substantially better".format(target_obj): -2
    }

    for d in iiw_data:
        # diff.append(d['score_iiw'] - d['score_new'])
        diff.append(d['score_iiw'] - d['score_extra_dci'])
        # diff.append(int(d['score_iiw']) - int(d['score_new']))
        # diff.append(int(d['score_iiw']) - int(d['score_extra_dci']))
        human.append(mapping_dict[d['human_specific']])

    scaler = MinMaxScaler(feature_range=(-2, 2), clip=True)
    new_diff = scaler.fit_transform(np.array(diff).reshape(-1, 1)).flatten().tolist()
    
    # new_diff = [normalized_score(k) for k in new_diff]
    # new_diff = diff
    # print(new_diff)
    # print(human)

    print("DCI")
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")
    # print("Accuracy")
    # acc = accuracy_score(new_diff, human)
    # print(acc)
    # print("===========")


def corr_docci():
    with open("./docci_specific.json", 'r') as f:
        iiw_data = json.load(f)
    f.close()


    diff = []
    human = []

    main_obj = "IIW"
    target_obj = "DOCCI"

    mapping_dict = {
        "{} is substantially better".format(main_obj): 2,
        "{} is marginally better".format(main_obj): 1,
        "Neutral": 0,
        "{} is marginally better".format(target_obj): -1,
        "{} is substantially better".format(target_obj): -2
    }

    for d in iiw_data:
        # diff.append(d['score_iiw'] - d['score_new'])
        diff.append(d['score_iiw'] - d['score_docci'])
        # diff.append(int(d['score_iiw']) - int(d['score_new']))
        # diff.append(int(d['score_iiw']) - int(d['score_docci']))
        human.append(mapping_dict[d['human_specific']])

    scaler = MinMaxScaler(feature_range=(-2, 2), clip=True)
    new_diff = scaler.fit_transform(np.array(diff).reshape(-1, 1)).flatten().tolist()
    
    # new_diff = [normalized_score(k) for k in new_diff]
    # new_diff = diff
    # print(new_diff)
    # print(human)

    print("DOCCI")
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")
    
    # print("Accuracy")
    # acc = accuracy_score(new_diff, human)
    # print(acc)
    # print("===========")
    
def corr_iiw():
    with open("./iiw_specific.json", 'r') as f:
        iiw_data = json.load(f)
    f.close()


    diff = []
    human = []

    main_obj = "IIW-Human"
    target_obj = "IIW-P5B"

    mapping_dict = {
        "{} is substantially better".format(main_obj): 2,
        "{} is marginally better".format(main_obj): 1,
        "Neutral": 0,
        "{} is marginally better".format(target_obj): -1,
        "{} is substantially better".format(target_obj): -2
    }

    for d in iiw_data:
        # diff.append(d['score_iiw'] - d['score_new'])
        diff.append(d['score_iiw'] - d['score_p5b'])
        # diff.append(int(d['score_iiw']) - int(d['score_new']))
        # diff.append(int(d['score_iiw']) - int(d['score_p5b']))
        human.append(mapping_dict[d['human_specific']])

    scaler = MinMaxScaler(feature_range=(-2, 2))
    new_diff = scaler.fit_transform(np.array(diff).reshape(-1, 1)).flatten().tolist()
    
    # new_diff = [normalized_score(k) for k in new_diff]
    # new_diff = diff
    # print(new_diff)
    # print(human)

    print("IIW")
    print("Spearman Corr")
    correlation, p_value = spearmanr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Kendau Corr")
    correlation, p_value = kendalltau(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")

    print("Pearson Corr")
    correlation, p_value = pearsonr(new_diff, human)
    print(correlation)
    # print(p_value)
    print("===========")
    
    # print("Accuracy")
    # acc = accuracy_score(new_diff, human)
    # print(acc)
    # print("===========")
    

if __name__ == '__main__':
    corr_dci()
    corr_docci()
    corr_iiw()