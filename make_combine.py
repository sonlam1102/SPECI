import json 
from copy import deepcopy

def read_iiw():
    data_gveal = []
    with open("./g_veal/iiw_specific.json", 'r') as f:
        data_gveal = json.load(f)
    f.close()
    
    data_polos = []
    with open("./polos/iiw_specific.json", 'r') as f:
        data_polos = json.load(f)
    f.close()
    
    data_speci = []
    with open("./speci/iiw_specific.json", 'r') as f:
        data_speci = json.load(f)
    f.close()
    
    assert len(data_gveal) == len(data_speci)
    assert len(data_gveal) == len(data_polos)
    
    return data_gveal, data_polos, data_speci
    

def read_docci():
    data_gveal = []
    with open("./g_veal/docci_specific.json", 'r') as f:
        data_gveal = json.load(f)
    f.close()
    
    data_polos = []
    with open("./polos/docci_specific.json", 'r') as f:
        data_polos = json.load(f)
    f.close()
    
    data_speci = []
    with open("./speci/docci_specific.json", 'r') as f:
        data_speci = json.load(f)
    f.close()
    
    assert len(data_gveal) == len(data_speci)
    assert len(data_gveal) == len(data_polos)
    
    return data_gveal, data_polos, data_speci


def read_dci():
    data_gveal = []
    with open("./g_veal/dci_specific.json", 'r') as f:
        data_gveal = json.load(f)
    f.close()
    
    data_polos = []
    with open("./polos/dci_specific.json", 'r') as f:
        data_polos = json.load(f)
    f.close()
    
    data_speci = []
    with open("./speci/dci_specific.json", 'r') as f:
        data_speci = json.load(f)
    f.close()
    
    assert len(data_gveal) == len(data_speci)
    assert len(data_gveal) == len(data_polos)
    
    return data_gveal, data_polos, data_speci

def combine1(data1, type1, data2, type2):
    def set_type(dtype):
        if dtype == 'iiw':
            return "score_p5b"
        elif dtype == "dci":
            return "score_extra_dci"
        elif dtype == "docci":
            return "score_docci"
        else:
            raise Exception
    
    new_data = deepcopy(data1)
    for i in range(0, len(new_data)):
        new_data[i]['score_new'] = (data1[i][set_type(type1)] + data2[i][set_type(type2)])/2
        new_data[i]['score_iiw'] = (data1[i]['score_iiw'] + data2[i]['score_iiw']) / 2
    
    return new_data

def combine2(data1, type1, data2, type2, data3, type3):
    def set_type(dtype):
        if dtype == 'iiw':
            return "score_p5b"
        elif dtype == "dci":
            return "score_extra_dci"
        elif dtype == "docci":
            return "score_docci"
        else:
            raise Exception
    
    new_data = deepcopy(data1)
    for i in range(0, len(data1)):
        new_data[i]['score_new'] = (data1[i][set_type(type1)] + data2[i][set_type(type2)] + data3[i][set_type(type3)])/3
        new_data[i]['score_iiw'] = (data1[i]['score_iiw'] + data2[i]['score_iiw'] + data3[i]['score_iiw']) / 3
    
    return new_data


if __name__ == "__main__":
    gveal_dci, polos_dci, speci_dci = read_dci()
    gveal_docci, polos_docci, speci_docci = read_docci()
    gveal_iiw, polos_iiw, speci_iiw = read_iiw()
    
    gveal_speci_dci = combine1(gveal_dci, "dci", speci_dci, "dci")
    gveal_speci_docci = combine1(gveal_docci, "docci", speci_docci, "docci")
    gveal_speci_iiw = combine1(gveal_iiw, "iiw", speci_iiw, "iiw")
    
    polos_speci_dci = combine1(polos_dci, "dci", speci_dci, "dci")
    polos_speci_docci = combine1(polos_docci, "docci", speci_docci, "docci")
    polos_speci_iiw = combine1(polos_iiw, "iiw", speci_iiw, "iiw")
    
    polos_gveal_dci = combine1(polos_dci, "dci", gveal_dci, "dci")
    polos_gveal_docci = combine1(polos_docci, "docci", gveal_docci, "docci")
    polos_gveal_iiw = combine1(polos_iiw, "iiw", gveal_iiw, "iiw")
    
    gveal_polos_speci_dci = combine2(gveal_dci, "dci", polos_dci, "dci", speci_dci, "dci")
    gveal_polos_speci_docci = combine2(gveal_docci, "docci", polos_docci, "docci", speci_docci, "docci")
    gveal_polos_speci_iiw = combine2(gveal_iiw, "iiw", polos_iiw, "iiw", speci_iiw, "iiw")
    
    with open('./dci_specific.json', 'w', encoding='utf-8') as f:
        json.dump(gveal_polos_speci_dci, f, ensure_ascii=False, indent=4)
    f.close()
    
    with open('./docci_specific.json', 'w', encoding='utf-8') as f:
        json.dump(gveal_polos_speci_docci, f, ensure_ascii=False, indent=4)
    f.close()
    
    with open('./iiw_specific.json', 'w', encoding='utf-8') as f:
        json.dump(gveal_polos_speci_iiw, f, ensure_ascii=False, indent=4)
    f.close()
