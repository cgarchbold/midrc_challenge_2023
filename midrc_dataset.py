import os
import pandas as pd
import pydicom
import torch
import numpy as np
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader


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
        self
        for image_name in os.listdir(self.data_dir):
            if image_name in fp_list:
                self.file_paths.append(os.path.join(self.data_dir, image_name))
            
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        fp = self.file_paths[idx]

        #print(os.path.basename(fp))
        fn, ext = os.path.splitext(os.path.basename(fp))
        pid, studyid, seriesid, sopid  = fn.split('_')
        score = self.annotations[self.annotations['StudyInstanceUID'] == studyid]['mRALE Score'].iloc[-1]

        image = Image.open(fp) #Convert to float!

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
        image = Image.fromarray(ds.pixel_array.astype(np.float32) / 255.0) #Convert to float!
        suid = ds.StudyInstanceUID
        score = self.annotations[self.annotations['StudyInstanceUID'] == suid]['mRALE Score']
        score = 0
        
        if self.transform:
            image = self.transform(image)

        return image, score

