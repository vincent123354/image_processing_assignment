import torch
import pandas as pd
import numpy as np
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.imgs = df.imgs.values
        self.labels = df.labels.values
        self.transform = transform
        
    def __getitem__(self, idx):
        image = np.array(Image.open(self.imgs[idx]).convert('RGB'))
        if self.transform != None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
    def __len__(self):
        return len(self.imgs)