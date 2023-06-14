import os
import pandas as pd
import pydicom
import torch
import numpy as np
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import random


class midrc_challenge_dataset(Dataset):
    '''
        Implements the MIDRC Challenge Dataset using pytorch utils.data.Dataset

        transform: 

        image_fps: a list of image file_names associated with the dataset
    '''
    def __init__(self, root_dir,annotation_path, transform=None, fp_list=None):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir,'data')

        self.transform = transform

        self.annotations = pd.read_csv(annotation_path)

        self.file_paths = []
        for image_name in os.listdir(self.data_dir):
            if image_name in fp_list:
                self.file_paths.append(os.path.join(self.data_dir, image_name))
            
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        fp = self.file_paths[idx]

        fn, ext = os.path.splitext(os.path.basename(fp))
        pid, studyid, seriesid, sopid  = fn.split('_')
        score = self.annotations[self.annotations['StudyInstanceUID'] == studyid]['mRALE Score'].iloc[-1]

        image = Image.open(fp)

        if self.transform:
            image = self.transform(image)

        return image, score

    

class midrc_challenge_dicom(Dataset):
    '''
        Implements the MIDRC Challenge Dataset using pytorch utils.data.Dataset
    '''
    def __init__(self, root_dir, annotations_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = pd.read_csv(annotations_file)

        self.file_paths = []
        for image_name in os.listdir(root_dir):
            self.file_paths.append(os.path.join(root_dir, image_name))
        

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        fp = self.file_paths[idx]

        self.labels = []
        ds = pydicom.dcmread(fp)
        image = Image.fromarray(ds.pixel_array)
        suid = ds.StudyInstanceUID
        score = self.annotations[self.annotations['StudyInstanceUID'] == suid]['mRALE Score']
        score = 0
        
        if self.transform:
            image = self.transform(image)

        return image, score
    
class midrc_SIMCLR_dataset():
    # Implementing a class for a json defined version of a simclr pretraining input scheme

    def __init__(self, root_dir, annotations_file,json_file, transform = None):

        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir,'data').replace("\\","/")
        self.transform = transform
        self.annotations = pd.read_csv(annotations_file)

        with open(json_file,"r") as label_info:
            self.label_info=json.load(label_info)

        if "patient_wise" in json_file:
            self.list_key = 'same_patient_images'
        else:
            self.list_key = 'same_study_images'
        # TODO: Instead load the files from the jsons
        #self.file_dict = []

    def __len__(self):
        len(self.label_info)

    def __getitem__(self, idx):
        # each item in the dictionary contains a list of dictionaries
        dict = self.label_info[idx]

        img_fp = os.path.join(self.data_dir, dict['full_image_name']).replace("\\","/")
        rand_select = random.choice(dict[self.list_key])
        img2_fp = os.path.join(self.data_dir, rand_select).replace("\\","/")

        fn, ext = os.path.splitext(os.path.basename(img_fp))
        pid, studyid, seriesid, sopid  = fn.split('_')
        score_1 = self.annotations[self.annotations['StudyInstanceUID'] == studyid]['mRALE Score'].iloc[-1]

        fn, ext = os.path.splitext(os.path.basename(img2_fp))
        pid, studyid, seriesid, sopid  = fn.split('_')
        score_2 = self.annotations[self.annotations['StudyInstanceUID'] == studyid]['mRALE Score'].iloc[-1]

        img1 = Image.open(img_fp)
        img2 = Image.open(img_fp)

        # TODO: Apply same transform to both images in case of random transformation
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2) # Will need to atleast apply PILToTensor()

        return img1, img2, score_1, score_2
