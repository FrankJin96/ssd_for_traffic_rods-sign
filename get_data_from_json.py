# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 22:03:20 2018

TO DO : get city-scapes dataset annotation data
        from city-scapes json file into pickle

@author: Frank-Jin
"""

import pickle
import json
import numpy as np
import os

class json_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 1
        self.data = dict()
        self._preprocess_json()

    def _preprocess_json(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            one_hot_classes = []
            bounding_boxes = []
            full_path = self.path_prefix + filename
            f = open(full_path,'r')
            raw = json.loads(f.read())
            width = float(raw['imgWidth'])
            height = float(raw['imgHeight'])
            for i in range(len(raw['objects'])):
                temp = raw['objects'][i]
                class_name = temp['label']
                one_hot_class = self._to_one_hot(class_name)
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
            image_name = filename.rstrip('gtFine_polygons.json') + '_leftImg8bit.png'
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data
            f.close()
# =============================================================================
#             tree = ElementTree.parse(self.path_prefix + filename)
#             root = tree.getroot()
#             bounding_boxes = []
#             one_hot_classes = []
#             size_tree = root.find('size')
#             width = float(size_tree.find('width').text)
#             height = float(size_tree.find('height').text)
#             for object_tree in root.findall('object'):
#                 for bounding_box in object_tree.iter('bndbox'):
#                     xmin = float(bounding_box.find('xmin').text)/width
#                     ymin = float(bounding_box.find('ymin').text)/height
#                     xmax = float(bounding_box.find('xmax').text)/width
#                     ymax = float(bounding_box.find('ymax').text)/height
#                 bounding_box = [xmin,ymin,xmax,ymax]
#                 bounding_boxes.append(bounding_box)
#                 class_name = object_tree.find('name').text
#                 one_hot_class = self._to_one_hot(class_name)
#                 one_hot_classes.append(one_hot_class)
#             image_name = root.find('filename').text
#             bounding_boxes = np.asarray(bounding_boxes)
#             one_hot_classes = np.asarray(one_hot_classes)
#             image_data = np.hstack((bounding_boxes, one_hot_classes))
#             self.data[image_name] = image_data
# =============================================================================

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        if name == 'pole' or name == 'rod':
            one_hot_vector[0] = 1
        else:
            one_hot_vector[0] = 0
#            print('unknown label: %s' %name)

        return one_hot_vector
    
#################################
## !!!!!!!!!!!!!!!!!!!!!!!!!!! ##
## example on how to use it!!! ##
## !!!!!!!!!!!!!!!!!!!!!!!!!!! ##
# import pickle
# data = XML_preprocessor('VOC2007/Annotations/').data
# pickle.dump(data,open('VOC2007.pkl','wb'))
        
if __name__ == "__main__":
    # prefix must end with /
    data = json_preprocessor('C:/Users/Administrator/Desktop/train/stuttgart/').data
    pickle.dump(data,open('anno_CTSCP_stgt.pkl','wb'))
    