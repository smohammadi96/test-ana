from PIL import Image
import os
import cv2
import shutil
import fnmatch
import os,json
import numpy as np

path_to_json = r'./splits/test.json'
directory = r'./OriginalFrames'
with open(path_to_json) as json_file:
        data = json.load(json_file)
        data=np.array(data)
        for i in data:
            pattern1=i[0]+'*.png'
            pattern2=i[1]+'*.png'
            c=1
            for file in os.listdir(directory):
                name=file
                src=os.path.join(directory, file)
                dst=r'./Deepfakes/test/real'
                if fnmatch.fnmatch(name,pattern1):
                    shutil.copy(src, dst)
                if fnmatch.fnmatch(name,pattern2):
                    shutil.copy(src, dst)
                c+=1
            continue

path_to_json = r'./splits/train.json'
directory = r'./OriginalFrames'
with open(path_to_json) as json_file:
        data = json.load(json_file)
        data=np.array(data)
        for i in data:
            pattern1=i[0]+'*.png'
            pattern2=i[1]+'*.png'
            c=1
            for file in os.listdir(directory):
                name=file
                src=os.path.join(directory, file)
                dst=r'./Deepfakes/train/real'
                if fnmatch.fnmatch(name,pattern1):
                    shutil.copy(src, dst)
                if fnmatch.fnmatch(name,pattern2):
                    shutil.copy(src, dst)
                c+=1
            continue

path_to_json = r'./splits/val.json'
directory = r'./OriginalFrames'
with open(path_to_json) as json_file:
        data = json.load(json_file)
        data=np.array(data)
        for i in data:
            pattern1=i[0]+'*.png'
            pattern2=i[1]+'*.png'
            c=1
            for file in os.listdir(directory):
                name=file
                src=os.path.join(directory, file)
                dst=r'./Deepfakes/validation/real'
                if fnmatch.fnmatch(name,pattern1):
                    shutil.copy(src, dst)
                if fnmatch.fnmatch(name,pattern2):
                    shutil.copy(src, dst)
                c+=1
            continue

import shutil
#shutil.rmtree('../OriginalFrames', ignore_errors=True)

"""# ***fake below: ***"""

path_to_json = r'./splits/test.json'
directory = r'./DeepfakesFrames'
with open(path_to_json) as json_file:
        data = json.load(json_file)
        data=np.array(data)
        for i in data:
            pattern=i[0] + '_' + i[1] +'*.png'
            c=1
            for file in os.listdir(directory):
                name=file
                src=os.path.join(directory, file)
                dst=r'./Deepfakes/test/fake'
                if fnmatch.fnmatch(name,pattern):
                    shutil.copy(src, dst)
                c+=1
            continue

path_to_json = r'./splits/train.json'
directory = r'./DeepfakesFrames'
with open(path_to_json) as json_file:
        data = json.load(json_file)
        data=np.array(data)
        for i in data:
            pattern=i[0] + '_' + i[1] +'*.png'
            c=1
            for file in os.listdir(directory):
                name=file
                src=os.path.join(directory, file)
                dst=r'./Deepfakes/train/fake'
                if fnmatch.fnmatch(name,pattern):
                    shutil.copy(src, dst)
                c+=1
            continue

path_to_json = r'./splits/val.json'
directory = r'./DeepfakesFrames'
with open(path_to_json) as json_file:
        data = json.load(json_file)
        data=np.array(data)
        for i in data:
            pattern=i[0] + '_' + i[1] +'*.png'
            c=1
            for file in os.listdir(directory):
                name=file
                src=os.path.join(directory, file)
                dst=r'./Deepfakes/validation/fake'
                if fnmatch.fnmatch(name,pattern):
                    shutil.copy(src, dst)
                c+=1
            continue

import shutil
#shutil.rmtree('../DeepfakesFrames', ignore_errors=True)


'''
!pip install GPUtil
# !pwd

!pip install einops

!python /kaggle/working/ResViT/cvit_train_model2.py -e 50 -s 'g' -l 0.0001 -w 0.0000001 -d /kaggle/working/Deepfakes/ -b 32

#!python /kaggle/working/ResViT/cvit_prediction_model2.py

#!pip install facenet_pytorch

#!pip install face_recognition

'''
