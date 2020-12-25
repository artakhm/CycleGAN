import torch.nn.functional as F
import itertools
import torch
import numpy  as np
import pandas as pd
import datetime
import torchvision
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
            )

    def forward(self, x):
        out = self.main(x)
        return x + out
        
class Generator(nn.Module):
    def __init__(self, norm_type='batch_norm'):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1,stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, padding=1,stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32), 
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
    
class Generator_small(nn.Module):
    def __init__(self, norm_type='batch_norm'):
        super(Generator_small, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1,stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, padding=1,stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32), 
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4,padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.main(x)
