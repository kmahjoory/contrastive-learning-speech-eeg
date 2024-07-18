

# Import libraries
import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from src.models.utils import *


from collections import OrderedDict


## EEG Encoder
class EEGEncoder(nn.Module):
    def __init__(self,             
            fs = 128, # sampling rate
            T = 5, # lenght of each trial in seconds
            C = 128, # number of EEG channels
            F1 = 8, # 8 or 4 depth in the 2nd conv layer
            D = 2, # number of spatial filters in the second conv layer
            F2 = None, # number of channels (depth) in the pont-wise conv layer
            drp_fact = 0.25, # dropout factor

            avgpool_spat_fact = 4, # spatial pooling factor
            avgpool_point_fact = 8, # depth pooling factor

        ):
        super(EEGEncoder, self).__init__()

        if F2 is None:
            F2 = D * F1

        self.eeg_encoder = nn.Sequential(OrderedDict({
            'conv_temp': Conv2d(1, F1, (1, int(fs/2)), padding='same', bias=True, groups=1),
            'bn_temp': nn.BatchNorm2d(F1, affine=True),
            'conv_spat': Conv2d(F1, out_channels=D*F1, kernel_size=(C, 1), padding=(0, 0), bias=False, groups=F1),
            'bn_spat': nn.BatchNorm2d(D*F1, affine=True), 
            'elu_spat': ELU(), 
            'avgpool_spat': nn.AvgPool2d(1, avgpool_spat_fact), 
            'dropout_spat': nn.Dropout(drp_fact),

            'conv_depth': Conv2d(F2, F2, (1, int(fs/(2*avgpool_spat_fact))), padding='same', bias=False, groups=D*F1),
            'conv_poit': Conv2d(F2, F2, kernel_size=(1, 1), padding=(0, 0), groups=1, bias=False),
            'bn_point': nn.BatchNorm2d(F2, affine=True), 
            'elu_point': ELU(), 
            'avgpool_point': nn.AvgPool2d(1, avgpool_point_fact), 
            'dropout_point': nn.Dropout(drp_fact),
            'flatten': nn.Flatten(),
        })) 

    def forward(self, x):
        x = self.eeg_encoder(x)
        return x

    def normalize_weights(self):

        for ix, (name, param) in enumerate(self.eeg_encoder.named_parameters()):
            if  name == 'conv_spat.weight': # normalize 2nd conv weights to max norm 1
                #print(param.data[:5, 0, 0, 0])
                param.data = torch.renorm(param.data, 2, 0, maxnorm=1)
                #print(param.data[:5, 0, 0, 0])
            elif name == 'linear.weight' and param.ndim==2: # normalize fc weights to max norm 0.25
                #print(param.data.norm(dim=(1)))
                param.data = torch.renorm(param.data, 2, 0, maxnorm=0.25)
                #print(param.data.norm(dim=(1)))





if __name__ == '__main__':
    # Test the model
    model = EEGEncoder(drp_fact=1)
    fs = 128
    T = 5
    C = 128
    x = torch.randn(1, 1, C, fs*T)
    y = model(x)
    print(y.shape)