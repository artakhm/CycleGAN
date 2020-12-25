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
from load_datasets import edges2shoes, horses2zebras, UnNormalize
from model import Discriminator, Generator_small, Generator
from train import train, visualize
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


transform1 = transforms.Compose([
                                    transforms.RandomApply([
                                        transforms.RandomCrop(128, padding=28),
                                        transforms.RandomRotation(5),
                                        transforms.RandomRotation(10),
#                                         transforms.RandomRotation(20),
                                    ], p=0.5),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.RandomErasing(0.5, scale=(0.02, 0.2)),
                                    transforms.Normalize((0.527074  , 0.5192849 , 0.45850393), (0.1926276 , 0.19553232, 0.21085525))
                               ])
unnorm_horse = UnNormalize((0.527074  , 0.5192849 , 0.45850393), (0.1926276 , 0.19553232, 0.21085525))


transform2 = torchvision.transforms.Compose([
                                  transforms.RandomApply([
                                        transforms.RandomCrop(128, padding=28),
                                        transforms.RandomRotation(5),
                                        transforms.RandomRotation(10),
#                                         transforms.RandomRotation(20),
                                    ], p=0.5),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.RandomErasing(0.5, scale=(0.02, 0.2)),
                                    transforms.Normalize((0.4834474, 0.4674779, 0.3979212), (0.2078747 , 0.20117787, 0.19642662))
                                ])

unnorm_zebra = UnNormalize((0.4834474, 0.4674779, 0.3979212), (0.2078747 , 0.20117787, 0.19642662))

dataset_ = horses2zebras('../input/horse2zebra/horse2zebra', transform1, transform2)

BATCH_SIZE = 16
VAL_SIZE = 50

train_set, test_set = torch.utils.data.random_split(dataset, [dataset_.__len__()-VAL_SIZE,VAL_SIZE])
trainloader = DataLoader(dataset_ = train_set, batch_size = BATCH_SIZE, shuffle = True)

# Dx = Discriminator().to(device)
# Dy = Discriminator().to(device)
# Gx = Generator_small().to(device)
# Gy = Generator_small().to(device)

# #train gan
# train(Dx,Dy,Gx, Gy, trainloader=trainloader, lr=0.0002,N_EPOCHS=25)

# visualize(Gx, Gy, test_set, unnorm_horse, unnorm_zebra)

# torch.save(Gx.state_dict(), './gx_adv.pth')
# torch.save(Gy.state_dict(), './gy_adv.pth')
# torch.save(Dx.state_dict(), './dx.pth')
# torch.save(Dy.state_dict(), './dy.pth')
