

# Import libraries
import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.utils import *




class EnvelopeEncoder(nn.Module):

    def __init__(self,             
            fs = 128, # sampling rate
            T = 5, # lenght of each trial in seconds
            F1 = 4,
            drp_fact = 0.5, # dropout factor

            avgpool1_fact = 2,
            avgpool2_fact = 2,
            avgpool3_fact = 8,
        ):
        super(EnvelopeEncoder, self).__init__()

        self.env_encoder = nn.Sequential(
            Conv2d(1, F1, (1, int(fs//2)), padding='same', bias=True),
            nn.BatchNorm2d(F1, affine=True), ELU(), nn.AvgPool2d(1, avgpool1_fact), nn.Dropout(drp_fact),
            Conv2d(F1, F1, (1, int(fs//4)), padding='same', bias=False, groups=1),
            nn.BatchNorm2d(F1, affine=True), ELU(), nn.AvgPool2d(1, avgpool2_fact), nn.Dropout(drp_fact),
            Conv2d(F1, F1*4, (1, int(fs//8)), padding='same', bias=False, groups=1),
            nn.BatchNorm2d(F1*4, affine=True), ELU(), nn.AvgPool2d(1, avgpool3_fact), nn.Dropout(drp_fact),
            nn.Flatten(),
        ) 

    def forward(self, x):
        x = self.env_encoder(x)
        return x

