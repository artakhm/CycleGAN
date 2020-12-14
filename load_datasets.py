import torch
import numpy  as np
import pandas as pd
import datetime
import itertools
import os
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join

class edges2shoes(Dataset) :
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,2])
        image = torchvision.transforms.ToTensor()(torchvision.transforms.Resize([128,256], interpolation=2)(Image.open(img_path)))
        edges, item = torch.split(image,128,2)

        if self.transform:
            edges = self.transform(edges)
            item = self.transform(item)
            
        return edges, item
    
class horses2zebras(Dataset) :
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.filenames_a = listdir(rootdir+'/trainA')
        self.filenames_b = listdir(rootdir+'/trainB')
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames_a)
    
    def __getitem__(self, index):
        a = os.path.join(self.rootdir+'/trainA', self.filenames_a[index])
        b = os.path.join(self.rootdir+'/trainB', self.filenames_b[index])
        image_a = torchvision.transforms.ToTensor()(torchvision.transforms.Resize([128,128], interpolation=2)(Image.open(a).convert('RGB')))
        image_b = torchvision.transforms.ToTensor()(torchvision.transforms.Resize([128,128], interpolation=2)(Image.open(b).convert('RGB')))

        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
            
        return image_a, image_b