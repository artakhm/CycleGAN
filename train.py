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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(Dx,Dy,G_ytox, G_xtoy, N_EPOCHS, trainloader,lr=0.001, BATCH_SIZE=16,
        LAMBDA=10, LAMBDA_ID = 0.0):

    ITER_LOG = trainloader.__len__()-1
    now = datetime.datetime.now()
    optimizerDx = torch.optim.Adam(Dx.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerDy = torch.optim.Adam(Dy.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(itertools.chain(G_ytox.parameters(), G_xtoy.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizerGx = torch.optim.Adam(G_ytox.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerGy = torch.optim.Adam(G_xtoy.parameters(), lr=lr, betas=(0.5, 0.999))
    
    mse = nn.MSELoss()
    l = nn.L1Loss()
    
    for epoch in range(N_EPOCHS):
        Dy_log = 0.0
        Dx_log = 0.0
        Gan_loss_log = 0.0
        Cycle_loss_log = 0.0
        for i, data in enumerate(trainloader, 0):
            x, y = data[0].to(device), data[1].to(device)
            t_size = Dx(x).size()
            if x.size()[0] != BATCH_SIZE: continue
                
            #FORWARD
            y_fake = G_xtoy(x)
            x_fake = G_ytox(y)
                
            y_rec = G_xtoy(x_fake)
            x_rec = G_ytox(y_fake)
            
            ones = torch.ones(t_size).to(device)
            zeros = torch.zeros(t_size).to(device)
                
            optimizerG.zero_grad()
                
            #CYCLCE LOSS
            cycle_loss_x = l(x_rec, x)*LAMBDA
            cycle_loss_y = l(y_rec, y)*LAMBDA
            Cycle_loss_log += cycle_loss_x.item() + cycle_loss_y.item()
            
            #GAN LOSS
            gan_loss_y =  mse(Dy(y_fake), ones)
            gan_loss_x =  mse(Dx(x_fake), ones)
            Gan_loss_log += gan_loss_x.item() + gan_loss_y.item()
            
            #IDENTITY LOSS
            id_loss = 0
            if LAMBDA_ID:
                id_loss = (l(G_xtoy(x), x) + l(G_ytox(y), y))*LAMBDA_ID*LAMBDA
             
            #STEP
            G_loss = gan_loss_y + gan_loss_x + cycle_loss_y + cycle_loss_x + id_loss
            G_loss.backward()
            optimizerG.step()
            
                
            #DISCRIMINATOR LOSS
            optimizerDx.zero_grad()
            D_loss_X = (mse(Dx(x),ones) + mse(Dx(G_ytox(y)), zeros))*0.5
            Dx_log += D_loss_X.item()
            D_loss_X.backward()
            optimizerDx.step()
                
            optimizerDy.zero_grad()
            D_loss_Y = (mse(Dy(y),ones) + mse(Dy(G_xtoy(x)), zeros))*0.5
            Dy_log += D_loss_Y.item()
            D_loss_Y.backward()
            optimizerDy.step()
                
            if  i%ITER_LOG == ITER_LOG-1:    
                print('[%d, %5d] Discriminator X MSE loss: %.3f' %
                    (epoch + 1, i + 1, Dx_log / ITER_LOG))
                Dx_log = 0.0
            
                print('[%d, %5d] Discriminator Y MSE loss: %.3f' %
                    (epoch + 1, i + 1, Dy_log / ITER_LOG))
                Dy_log = 0.0

                print('[%d, %5d] Generator GAN loss: %.3f' %
                    (epoch + 1, i + 1, Gan_loss_log / ITER_LOG))
                Gan_loss_log = 0.0
                    
                print('[%d, %5d] Generator Cycle loss: %.3f' %
                    (epoch + 1, i + 1, Cycle_loss_log / ITER_LOG))
                Cycle_loss_log = 0.0
                
                print('time:',datetime.datetime.now() - now)
                now = datetime.datetime.now()
            
                print('################################################')

    print('Finished Training')
    
## Train without adversarial loss
def train_usual(Gx, Gy, lr, epochs, batch_size, trainloader):
    ITER_LOG=trainloader.__len__()-1
    optimizerGx = torch.optim.Adam(Gx.parameters(), lr=lr)
    optimizerGy = torch.optim.Adam(Gy.parameters(), lr=lr)
    
    mse = nn.MSELoss()    
    for epoch in range(epochs):
        Gx_loss_log = 0.0
        Gy_loss_log = 0.0
        now = datetime.datetime.now()
        for i, data in enumerate(trainloader, 0):
            x, y = data[0].to(device), data[1].to(device)
            if x.size()[0] != batch_size: continue
            
            
            optimizerGx.zero_grad()
            x_fake = Gx(y)
            Gx_loss = mse(x_fake, x)
            Gx_loss.backward()
            optimizerGx.step()
            Gx_loss_log += Gx_loss.item()
            
            optimizerGy.zero_grad()
            y_fake = Gy(x)
            Gy_loss = mse(y_fake, y)
            Gy_loss.backward()
            optimizerGy.step()
            Gy_loss_log += Gy_loss.item()
                
            if  i%ITER_LOG == ITER_LOG-1:    

                print('[%d, %5d] Generator y to x loss: %.3f' %
                        (epoch + 1, i + 1, Gx_loss_log / ITER_LOG))
                Gx_loss_log = 0.0
                    
                print('[%d, %5d] Generator x to y loss: %.3f' %
                        (epoch + 1, i + 1, Gy_loss_log / ITER_LOG))
                Gy_loss_log = 0.0
                
                print('time:',datetime.datetime.now() - now)
                now = datetime.datetime.now()
            
                print('################################################')
                
    print('Finished Training')
    
    
    
def visualize(G_ytox, G_xtoy, test_set):
    testloader = DataLoader(dataset = test_set, batch_size = 8, shuffle = True)

    for i, data in enumerate(testloader, 0):
        x, y = data[0].to(device), data[1].to(device)
        with torch.no_grad(): x_fake, y_fake = G_ytox(y), G_xtoy(x)
        fig, axs = plt.subplots(2, 2,figsize=(8,8))
        axs[0][0].imshow(x_fake[0].cpu().detach().permute(1, 2, 0))
        axs[0][0].set_title('x_fake')
        axs[1][0].imshow(x[0].cpu().detach().permute(1, 2, 0))
        axs[1][0].set_title('x')
        axs[0][1].imshow(y_fake[0].cpu().detach().permute(1, 2, 0))
        axs[0][1].set_title('y_fake')
        axs[1][1].imshow(y[0].cpu().detach().permute(1, 2, 0))
        axs[1][1].set_title('y')
        
        print('Validation MSE Loss for x:',nn.MSELoss()(x, x_fake))
        print('Validation MSE Loss for y:',nn.MSELoss()(y, y_fake))
        break