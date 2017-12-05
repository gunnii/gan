
# coding: utf-8

# In[2]:

import os
import scipy.io as si
import json
import datetime
import numpy as np


# In[22]:

# crop datasets, you should install autocrop
# pip install autocrop
# https://github.com/leblancfg/autocrop

def crop():
    for x in range(0, 100):
        IMG_PATH = 'dataset/wiki_crop/' + str(x).zfill(2)
        crop_path = IMG_PATH + '/crop'
        command = 'autocrop -p ' + IMG_PATH + ' -w 200 -H 200' # crop images to 200*200 size
        if not os.path.exists(crop_path):
            os.makedirs(crop_path)
        os.system(command)
    print('[Preprocessing] image crop completed!!!')

def load_mat():
    MAT_PATH = 'dataset/wiki_crop/wiki.mat'
    load = si.loadmat(MAT_PATH)
    # total : 62328
    result = dict()
    params = ['dob', 'photo_taken', 'gender', 'name', 'age']
    size = load['wiki']['full_path'][0][0][0].size   # shape: (62328,)
    for wiki in load['wiki']:
        for i in range(0, size):  # 0 to size
            values = dict()
            dob = datetime.date.fromordinal(np.max([wiki['dob'][0][0][i] - 366, 1]))
            dob_date = dob.strftime('%Y-%m-%d')
            photo_taken = wiki['photo_taken'][0][0][i]
            full_path = wiki['full_path'][0][0][i][0]
            gender = wiki['gender'][0][0][i]
            name = wiki['name'][0][0][i]
            if not name:  # has no name tag
                name = ''
            values['dob'] = dob_date
            values['photo_taken'] = str(photo_taken)
            values['gender'] = str(gender)
            values['name'] = str(name)
            values['age'] = str(photo_taken - dob.year + 1)
            result[full_path] = values
    return result

def make_info(total):
    result = dict()
    for x in range(0, 100):  # 0 to 100
        cropped_PATH = 'dataset/wiki_crop/' + str(x).zfill(2) + '/crop'
        files = os.listdir(cropped_PATH)
        for y in files:
            path = str(x).zfill(2) +'/'+ y
            if total[path]:
                result[path] = total[path]
        print(cropped_PATH + " -> "+ str(len(files)))
    return result

def save_info(data):
    PATH = 'dataset/wiki_crop/wiki.json'
    with open(PATH, 'w') as fp:
        json.dump(data, fp, indent=4)
    print("Save JSON file.... completed")

def load_info():
    PATH = 'dataset/wiki_crop/wiki.json'
    with open(PATH) as fp:
        data = json.load(fp)
    print("Load JSON file.... completed")
    return data

def get_category(age):
    # 0~19:A, 20~29:B, 30~39:C, 40~49:D, 50~59:E, 60~:F
    if age < 20:  
        return "A"
    elif age >= 20 and age < 30:
        return "B"
    elif age >= 30 and age < 40:
        return "C"
    elif age >= 40 and age < 50:
        return "D"
    elif age >= 50 and age < 60:
        return "E"
    elif age >= 60:
        return "F"
    
    
# 99/crop folder가 있으면, 전처리 완료된 것으로 간주
# json file이 없으면, make info, save info


