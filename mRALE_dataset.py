import os
import pandas as pd
import pydicom
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class mRALEDataset(Dataset):
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

