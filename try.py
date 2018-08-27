# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:02:38 2018

@author: Frank_Jin
"""

import pickle
import json
import numpy as np
import os

data_path = 'C:/Users/Administrator/Desktop/train/stuttgart/'
data = dict()
num_classes = 1

def to_one_hot(num_classes, name):
    one_hot_vector = [0] * num_classes
    if name == 'pole' or name == 'rod':
        one_hot_vector[0] = 1
    else:
        one_hot_vector[0] = 0

    return one_hot_vector

filenames = os.listdir(data_path)
for filename in filenames:
    one_hot_classes = []
    bounding_boxes = []
    full_path = data_path + filename
    f = open(full_path,'r')
    raw = json.loads(f.read())
    width = float(raw['imgWidth'])
    height = float(raw['imgHeight'])
    for i in range(len(raw['objects'])):
        temp = raw['objects'][i]
        class_name = temp['label']
        one_hot_class = to_one_hot(num_classes, class_name)
        if one_hot_class == [1]:
            raw_bbox = np.array(temp['polygon'])
            xmin = float(np.min(raw_bbox[:,0])) / width
            xmax = float(np.max(raw_bbox[:,0])) / width
            ymin = float(np.min(raw_bbox[:,1])) / height
            ymax = float(np.max(raw_bbox[:,1])) / height
            bounding_box = [xmin,ymin,xmax,ymax]
            
            one_hot_classes.append(one_hot_class)
            bounding_boxes.append(bounding_box)

    bounding_boxes = np.asarray(bounding_boxes)
    one_hot_classes = np.asarray(one_hot_classes)
    image_name = filename.rstrip('polygons.json') + 'color.png'
    image_data = np.hstack((bounding_boxes, one_hot_classes))
    data[image_name] = image_data
    f.close()
    
pickle.dump(data,open('anno_CTSCP_stug.pkl','wb'))
