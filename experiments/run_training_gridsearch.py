
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import mne
import time


from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Change working directory to the root of the project
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../")

# Import custom modules
from src.train import eval_model_cl
from src.data.utils import unfold_raw, rm_repeated_annotations
from src.data.load_data import load_data

from src.models.eeg_encoder import EEGEncoder
from src.models.envelope_encoder import EnvelopeEncoder
from src.models.contrastive_eeg_speech import CLEE


# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--num_epochs", default=400, type=int, help="Number of epochs")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--scheduler_step_size", default=200, type=int, help="Scheduler step size")
parser.add_argument("--scheduler_gamma", default=0.5, type=float, help="Scheduler gamma")
parser.add_argument("--save_best", default=False, type=bool, help="Save the best model or not")
parser.add_argument("--save_curves", default=False, type=bool, help="Save the loss curves or not")
parser.add_argument("--num_subjects", default=5, type=int, help="Number of subjects to use")
parser.add_argument("--drp_fact", default=0.5, type=float, help="Dropout factor")
args = vars(parser.parse_args())
 
# Parameters to be tuned
lr = args["lr"]
num_epochs = args["num_epochs"]
batch_size = args["batch_size"]
scheduler_step_size = args["scheduler_step_size"]
scheduler_gamma = args["scheduler_gamma"]
save_best = args["save_best"]
save_curves = args["save_curves"]
num_subjects = args["num_subjects"]
drp_fact = args["drp_fact"]
subj_ids = list(range(1, num_subjects+1))


fs = 128
window_size = int(5 * fs)
stride_size_train, stride_size_val, stride_size_test = int(2.5 * fs), int(5 * fs), int(5 * fs)
#batch_size = int(batch_size)
#lr = float(lr)




# Print RUN info
print('========================================')
print('N_epochs: %d  batch size: %d  N_subjects: %d'% (num_epochs, batch_size, len(subj_ids)))
print('Save best model: %s  Save curves: %s' % (save_best, save_curves))
print(f'window_size: {window_size}  stride_size_test: {stride_size_test}')

print(f"Device: {device}")
# Print GPU info if available
if torch.cuda.is_available():
    from pynvml import *
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total/1e9}')
    print(f'free     : {info.free/1e9}')
    print(f'used     : {info.used/1e9}')
print('========================================')

n_channs = 129 # 128 for eeg, 1 for env
dataset_name = ['roch', 'natural_speech']
path_data = os.path.join('../datasets/preprocessed/', dataset_name[0], dataset_name[1])
print(f'data_path: {path_data}')


X = load_data(subj_ids, path_data, window_size, 
              stride_size_train, stride_size_val, stride_size_test, n_channs)


# Create dataloaders
class MyDataset(Dataset):
    def __init__(self, eeg, env):
        self.eeg = eeg
        self.env = env
    
    def __getitem__(self, index):
        return self.eeg[index], self.env[index]
    
    def __len__(self):
        return len(self.eeg)
    

dataset_train = MyDataset(X['eegs_train'], X['envs_train'])
dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
dl_val = DataLoader(MyDataset(X["eegs_val"], X["envs_val"]), 
                    batch_size=batch_size, shuffle=True, drop_last=True)



# Create model
eeg_encoder = EEGEncoder(drp_fact=drp_fact)
env_encoder = EnvelopeEncoder(drp_fact=drp_fact)
model = CLEE(eeg_encoder, env_encoder)
model.to(device)

# Train model
models_dict = {f'lr_{lr}_bs_{batch_size}': model}
lossi = []
udri = [] # update / data ratio 
ud = []



for name, model in models_dict.items():

    # Reset for the new model in the loop
    print(f"+--------------New model: {name}----------------------+")
    run_time = time.strftime('%m%d_%H%M%S')
    if save_curves:
        writer = SummaryWriter(log_dir=f"../models/logs/{name}_{run_time}")
    model.to(device)
    optimizer = optim.NAdam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.7)
    cnt = 0
    loss_batches = []


    for epoch in range(1, num_epochs+1):

        print(f"====== Epoch: {epoch}")

        model.train()
        for ix_batch, (Xb_eeg, Xb_env) in enumerate(dl_train):

            # send to device
            Xb_eeg = Xb_eeg.to(device)
            Xb_env = Xb_env.to(device)

            # Zero out gradients
            optimizer.zero_grad()

            # forward pass
            eeg_features, env_features, logit_scale = model(Xb_eeg, Xb_env) 


            # normalize features
            eeg_features_n = eeg_features / eeg_features.norm(dim=1, keepdim=True)
            env_features_n = env_features / env_features.norm(dim=1, keepdim=True)

            # logits
            logits_per_eeg = logit_scale * eeg_features_n @ env_features_n.t()
            logits_per_env = logits_per_eeg.t()

            #loss function
            labels = torch.arange(batch_size).to(device)
            loss_eeg = F.cross_entropy(logits_per_eeg, labels)
            loss_env = F.cross_entropy(logits_per_env, labels)
            loss   = (loss_eeg + loss_env)/2

            # backward pass
            loss.backward()
            optimizer.step()

            loss_batches.append(loss.item())
            cnt += 1

            if save_curves:
                with torch.no_grad():
                    #ud = {f"p{ix}":(lr*p.grad.std() / p.data.std()).log10().item() for ix, p in enumerate(model.parameters()) if p.ndim==4 }
                    #writer.add_scalars('UpdateOData/ud', ud, cnt)
                    writer.add_scalar('Loss/train_batch', loss.item(), cnt)

            # normalize weights
            #with torch.no_grad():
            #    model.eeg_encoder.normalize_weights()
            
            #break   

        loss_epoch = loss_batches[-(ix_batch + 1):]  # mean loss across batches
        loss_epoch = sum(loss_epoch) / len(loss_epoch)
        if save_curves:
            writer.add_scalar('Loss/train_epoch', loss_epoch, epoch)
        #for pname, p in model.named_parameters():
        #writer.add_histogram(f'Params/{pname}', p, epoch)
        #writer.add_histogram(f'Grads/{pname}', p.grad, epoch)

        loss_val, *_ = eval_model_cl(dl_val, model, device=device)
        if save_curves:
            writer.add_scalar('Loss/val_epoch', loss_val, epoch)

        # Save the best model
        if save_best:
            
            if epoch == 1:
                best_loss = loss_val    

            if loss_val < best_loss:
                best_loss = loss_val
                best_model_weights = model.state_dict().copy()
                best_epoch = epoch

        model.train()

        # Update learning rate based on epoch
        scheduler.step()

    # save the best model after all epochs
    torch.save(best_model_weights, f"../models/best_models/epoch_{best_epoch}_loss_{best_loss:.3f}_{run_time}.pth")
    #break   





