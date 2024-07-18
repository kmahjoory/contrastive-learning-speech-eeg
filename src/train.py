
# Import libraries
import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time



def eval_model_cl(dl, model, device=torch.device('cpu'), verbose=True):
    """ 
    This function calculates the loss on data, setting backward gradients and batchnorm
    off. This function is written for contrasting learning where the model takes in two
    inputs.

    Args:

    Returns:
      loss: Mean loss of all test samples (scalar)
      loss_X1: Mean loss of all test samples (scalar)
      loss_X2: Mean loss of all test samples (scalar)

    """
    losses, losses_X1, losses_X2 = [], [], []
    model.to(device)  # inplace for model
    # Set the model in evaluation mode
    model.eval()

    with torch.no_grad():
        for idx_batch, (X1b, X2b) in enumerate(dl):

            X1b = X1b.to(device)
            X2b = X2b[:, :, [0], :].to(device) # Only use the first channel

            X1b_features, X2b_features, logit_sc = model(X1b, X2b)

            # Normalize features
            X1b_f_n = X1b_features / X1b_features.norm(dim=1, keepdim=True)
            X2b_f_n = X2b_features / X2b_features.norm(dim=1, keepdim=True)

            logits_per_X1 = logit_sc * X1b_f_n @ X2b_f_n.t()
            logits_per_X2 = logits_per_X1.t()

            # Number of labels equals to the 1st dimension of X1b
            labels = torch.arange(X1b.shape[0], device=device)

            # Batch Loss 
            loss_X1 = F.cross_entropy(logits_per_X1, labels)
            loss_X2 = F.cross_entropy(logits_per_X2, labels)
            loss_batch   = (loss_X1 + loss_X2) / 2

            losses.append(loss_batch.item())
            losses_X1.append(loss_X1.item())
            losses_X2.append(loss_X2.item())

        # Epoch loss (mean of batch losses)
        loss  = sum(losses) / len(losses)
        loss_X1 = sum(losses_X1) / len(losses_X1)
        loss_X2 = sum(losses_X2) / len(losses_X2)

        if verbose:
          print(f"====> Validation loss: {loss:.4f},  X1 loss: {loss_X1:.4f}   X2 loss: {loss_X2:.4f}")

        return loss, loss_X1, loss_X2
    

def eval_sa(dl, model, device=torch.device('cpu'), verbose=True):
    """
        This function calculates the loss on data, setting backward gradients and batchnorm
        off. This function is written for contrasting learning where the model takes in two
        inputs.

        Args:

        Returns:
        loss_test: Mean loss of all test samples (scalar)
    """

    # Set the model in evaluation mode
    model.to(device)
    model.eval()
    accs = []
    with torch.no_grad():
        for idx_batch, (X1b, X2b) in enumerate(dl):

            X1b = X1b.to(device)
            X2b = X2b.to(device)
            
            X1b_features, X2b_features_att, logit_sc_att = model(X1b, X2b[:, :, [0], :])
            _,            X2b_features_unatt, logit_sc_unatt = model(X1b, X2b[:, :, [1], :])

            # Normalize features
            X1b_f_n         = X1b_features / X1b_features.norm(dim=1, keepdim=True)
            X2b_f_n_att     = X2b_features_att / X2b_features_att.norm(dim=1, keepdim=True)
            X2b_f_n_unatt   = X2b_features_unatt / X2b_features_unatt.norm(dim=1, keepdim=True)

            # logits
            logits_per_eeg_att = logit_sc_att * X1b_f_n @ X2b_f_n_att.t()
            logits_per_eeg_unatt = logit_sc_unatt * X1b_f_n @ X2b_f_n_unatt.t()
            accs.append(sum(logits_per_eeg_att.diag() > logits_per_eeg_unatt.diag()) / X1b.shape[0])

    if verbose:
        print(f"====> SA Accuracy: {100*sum(accs)/len(accs):.3f}%")
    return sum(accs)/len(accs)
