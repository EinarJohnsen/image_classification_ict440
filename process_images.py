import os
import numpy as np
from PIL import Image
import weed_utils as wd

NO_WEED, NO_WEED_STRING     =   0, 'n'
DICO, DICO_STRING           =   1, 'dico'
MIX, MIX_STRING             =   2, 'mix'
MONO, MONO_STRING           =   3, 'mono'
ERROR                       =   -1
DS_Store                    =   '.DS_Store'

def transformLabel(a, e):
    if (e == NO_WEED_STRING):
        value = NO_WEED
    else:
        if (a == DICO_STRING):
            value = DICO
        elif(a == MIX_STRING):
            value = MIX
        elif(a == MONO_STRING):
            value = MONO
        else:
            value = ERROR
    return value

def transformLabel2(a, e):

    if (a == DICO_STRING):
        value = 0
    elif(a == MIX_STRING):
        value = 1
    elif(a == MONO_STRING):
        value = 2
    return value

def transformLabel3(a, e):

    if (a == DICO_STRING):
        value = 0
    elif(a == MONO_STRING):
        value = 1
    return value


def load_data(path):
    cwd = os.getcwd()
    data_label = []
    data = []

    for file in os.listdir(path):
        if (file == DS_Store):
            continue

        weed_type = file.split('_')[0]
        #print(weed_type)
        containts_weed = file.split('.')[0][-1:]
        value = transformLabel(weed_type, containts_weed)
        data.append(path + "/" + file)
        data_label.append(value)
    return data, data_label

def load_data2(path):
    cwd = os.getcwd()
    data_label = []
    data = []

    for file in os.listdir(path):
        if (file == DS_Store):
            continue

        weed_type = file.split('_')[0]
        #print(weed_type)
        containts_weed = file.split('.')[0][-1:]
        #if(containts_weed == 'y'):
        value = transformLabel2(weed_type, containts_weed)
        data.append(path + "/" + file)
        data_label.append(value)
    return data, data_label

def load_data3(path):
    cwd = os.getcwd()
    data_label = []
    data = []

    for file in os.listdir(path):
        if (file == DS_Store):
            continue

        weed_type = file.split('_')[0]
        #print(weed_type)
        containts_weed = file.split('.')[0][-1:]
        if(containts_weed == 'y' and weed_type != 'mix'):
            print(path)
            value = transformLabel3(weed_type, containts_weed)
            data.append(path + "/" + file)
            data_label.append(value)
    return data, data_label

def run_it():
    data, data_label = load_data2("all_classes_few_images/3_classes_train")
    test_data, test_data_label = load_data2("all_classes_few_images/3_classes_test")
    verification_data, verification_data_label = load_data2('all_classes_few_images/3_classes_verification')
    return data, data_label, test_data, test_data_label, verification_data, verification_data_label