import argparse
import random
import time
import cv2
import pickle
import util as io
import os
import numpy as np

train_file = io.load_json_object('/home/michal/pcl.pytorch/train.json')
proposal_dict = {'boxes':[],'indexes':[],'scores':[]}
for cat in train_file:
    print(f'cat is {cat} ')
    for i,img in enumerate(train_file[cat]):
        if i%100 == 0:
            print(f'On image {i} out of {len(train_file[cat])}')
        #print(os.path.exists(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img}'))
        box_img = cv2.imread(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img}')
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(box_img)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        proposal_dict['boxes'].append(rects)
        proposal_dict['indexes'].append(img)
        proposal_dict['scores'].append(np.ones(len(proposal_dict['boxes'])).tolist())
    with open('train_proposals.pickle', 'wb') as handle:
        pickle.dump(proposal_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# img = cv2.imread('/data/michal5/gpv/learning_phase_data/web_data/images/fb549be38ff3ce688c9c06658311c6f1ff4a01db09ac8d16de2aac4a9e302e37.jpg')
# print(img)
# ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
# print(ss)
# ss.setBaseImage(img)
# ss.switchToSelectiveSearchQuality()

# print(ss)
# rects = ss.process()
# print(rects)

