import json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
import math
import pandas as pd

def make_corr(data, dtype="full"):
    diff = []
    human = []
    
    for index, d in data.iterrows():
        if d['Name'] == 'dci':
            main_obj = "IIW"
            target_obj = "DCI"

            mapping_dict = {
                "{} is substantially better".format(main_obj): 2,
                "{} is marginally better".format(main_obj): 1,
                "Neutral": 0,
                "{} is marginally better".format(target_obj): -1,
                "{} is substantially better".format(target_obj): -2
            }
        elif d['Name'] == 'dooci':
            main_obj = "IIW"
            target_obj = "DOCCI"

            mapping_dict = {
                "{} is substantially better".format(main_obj): 2,
                "{} is marginally better".format(main_obj): 1,
                "Neutral": 0,
                "{} is marginally better".format(target_obj): -1,
                "{} is substantially better".format(target_obj): -2
            }
        elif d['Name'] == 'iiw':
            main_obj = "IIW-Human"
            target_obj = "IIW-P5B"

            mapping_dict = {
                "{} is substantially better".format(main_obj): 2,
                "{} is marginally better".format(main_obj): 1,
                "Neutral": 0,
                "{} is marginally better".format(target_obj): -1,
                "{} is substantially better".format(target_obj): -2
            }
        else:
            exit(1)
        
        human.append(mapping_dict[d['Human']])
                
        diff.append(d['Source Total Visual'] - d['Target Total Visual'])
        
        # diff.append(d['Source Visual'] - d['Target Visual'])
        # diff.append(d['Source Spatial'] - d['Target Spatial'])
        # diff.append(d['Source Linguistic'] - d['Target Linguistic'])
        
        # comb_source = d['Source Color'] + d['Source Visual'] +  d['Source Spatial'] + d['Source Linguistic']
        # comb_target = d['Target Color'] + d['Target Visual'] +  d['Target Spatial'] + d['Target Linguistic']
        # diff.append(comb_source - comb_target)
        
        # comb_source = d['Source Color'] + d['Source Visual'] +  d['Source Spatial'] + d['Source Linguistic']
        # comb_target = d['Target Color'] + d['Target Visual'] +  d['Target Spatial'] + d['Target Linguistic']
        # diff.append(comb_source - comb_target)
        
    
    scaler = MinMaxScaler(feature_range=(-2, 2))
    new_diff = scaler.fit_transform(np.array(diff).reshape(-1, 1)).flatten().tolist()
    # new_diff = diff
    
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
    
    
if __name__ == '__main__':
    data = pd.read_excel("sample_results_ablation.xlsx")
    
    print(len(data))
    make_corr(data)