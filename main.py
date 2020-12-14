import torch
import numpy  as np
import pandas as pd
import datetime
import itertools
import os
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join
from load_datasets import edges2shoes, horses2zebras
from model import Discriminator, Generator_small, Generator
from train import train, train_usual, visualize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



os.environ['KAGGLE_USERNAME'] = "xxxxxx"
os.environ['KAGGLE_KEY'] = "xxxxxxxxxxxxxxxxxxxxx"

get_ipython().system('kaggle datasets download -d arnaud58/horse2zebra')

import zipfile
with zipfile.ZipFile('horse2zebra.zip', 'r') as zip_ref:
    zip_ref.extractall('/datasets')

get_ipython().system('kaggle datasets download -d balraj98/edges2shoes-dataset')


with zipfile.ZipFile('edges2shoes-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/datasets')


transform = torchvision.transforms.Compose([
#                                    torchvision.transforms.CenterCrop(100)
#                                     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

dataset = edges2shoes('datasets/edges2shoes-dataset/metadata.csv', 'datasets/edges2shoes-dataset', transform)
dataset_ = horses2zebras('datasets/horse2zebra')

BATCH_SIZE = 16
VAL_SIZE = 4000

train_set, test_set = torch.utils.data.random_split(dataset, [dataset.__len__()-VAL_SIZE,VAL_SIZE])
trainloader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)

# #train only generators
# Gx = Generator().to(device)
# Gy = Generator().to(device)

# train_usual(Gx, Gy, lr=0.0002,epochs=5, trainloader batch_size=BATCH_SIZE)

# visualize(Gx, Gy, train_set)

# #save model
# torch.save(Gx.state_dict(), './gx.pth')
# torch.save(Gy.state_dict(), './gy.pth')


# #load model
# modelx = Generator()
# modelx.load_state_dict(torch.load('./gx.pth'))
# modelx.eval()
# modely = Generator()
# modely.load_state_dict(torch.load('./gy.pth'))
# modely.eval()


# visualize(modelx.to(device), modely.to(device), train_set)



# Dx = Discriminator().to(device)
# Dy = Discriminator().to(device)
# Gx = Generator_small().to(device)
# Gy = Generator_small().to(device)


# #train gan
# train(Dx,Dy,Gx, Gy, lr=0.0002,N_EPOCHS=4)

# visualize(Gx, Gy, test_set)

# torch.save(Gx.state_dict(), './gx_adv.pth')
# torch.save(Gy.state_dict(), './gy_adv.pth')
# torch.save(Dx.state_dict(), './dx.pth')
# torch.save(Dy.state_dict(), './dy.pth')