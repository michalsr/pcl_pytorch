import os 
import shutil 
import util as io 
import numpy as np 
f = []

file_name = io.load_json_object('/data/michal5/web_training_info/category_to_image_id.json')
test_dict = {}
t = io.load_json_object('/home/michal/web_80/test.json')
#print(len(os.listdir('/data/michal5/gpv/learning_phase_data/web_data/images/')))
for entry in t:
    img_id = entry['image']['image_id']
    # if not os.path.exists(f'/data/michal5/gpv/learning_phase_data/web_data/{img_id}'):
    #     print('problem')
    if os.access(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img_id}',os.R_OK):
        c = set()
        for e in entry['coco_categories']['seen']:
            c.add(e)
        for e in entry['coco_categories']['unseen']:
            c.add(e)
   
        if len(c) != 0:
         
            if len(c) <=1:
                if list(c)[0] not in test_dict:
                    test_dict[list(c)[0]] = []
                if os.access(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img_id}',os.R_OK):
                    test_dict[list(c)[0]].append(entry['image']['image_id'])
too_small = 0
smallest_num = 1000
actual_classes = {}
print(len(test_dict),'test dict')
for a in test_dict:
    if len(test_dict[a])>=20:
        actual_classes[a] = test_dict[a]
        if len(test_dict[a]) <smallest_num:
            smallest_num = len(test_dict[a])
# print(len(actual_classes))
# print(list(actual_classes.keys()))
#io.dump_json_object(actual_classes)
print(smallest_num)
final_list = {}
for a in actual_classes:
    if a not in final_list:
        final_list[a] = []
    for img in actual_classes[a]:
  
        if os.access(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img}',os.R_OK):
            exists = os.path.isfile(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img}')
            print(exists,'exists')
            if exists == False:
                print('bad')
           
                # if not os.path.exists(f'/home/michal/pcl.pytorch/test_val_imgs/{a}/'):
                #     os.mkdir(f'/home/michal/pcl.pytorch/test_val_imgs/{a}/')
                shutil.move(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img}',f'/home/michal/pcl.pytorch/test_val_imgs/{a}/{img}')

                final_list[a].append(img)
for cat in os.listdir('/home/michal/pcl.pytorch/test_val_imgs/'):
    if len(os.listdir(f'/home/michal/pcl.pytorch/test_val_imgs/{cat}/')) <20:
        shutil.rmtree(f'/home/michal/pcl.pytorch/test_val_imgs/{cat}/')
#for f in os.listdir(f'/home/michal/pcl.pytorch/test_val_imgs/{a}/{img}


# for entry in final_list:
#     if not os.path.exists(f'/home/michal/pcl.pytorch/test_val_imgs/{entry}/'):
#         os.mkdir(f'/home/michal/pcl.pytorch/test_val_imgs/{entry}/')
#     for img in final_list[entry]:
#         shutil.move(f'/data/michal5/gpv/learning_phase_data/web_data/images/{img}',f'/home/michal/pcl.pytorch/test_val_imgs/{entry}/{img}')
for f in os.listdir('/home/michal/pcl.pytorch/test_val_imgs/'):
    for img in os.listdir(f'/home/michal/pcl.pytorch/test_val_imgs/{f}/'):
        shutil.move(f'/home/michal/pcl.pytorch/test_val_imgs/{f}/{img}',f'/home/michal/pcl.pytorch/test_val_imgs/{f}/{img}.jpg')





