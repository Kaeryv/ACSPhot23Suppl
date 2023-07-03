#!/usr/bin/env python
# coding: utf-8
import argparse
import time

job_start = time.time()

"""

    Parse arguments

"""
parser = argparse.ArgumentParser()
parser.add_argument("-epochs", default=500, type=int)
parser.add_argument("-nthreads", default=1, type=int)
parser.add_argument("-complexity", default=16, type=int)
parser.add_argument("-data", default="./ds_5kset.npz", type=str)
parser.add_argument("-name", default="u-net.model", type=str)
parser.add_argument("-device", default="cuda", type=str)
parser.add_argument("-lr", default=1e-4, type=float)
parser.add_argument("-bs", default=32, type=int)
parser.add_argument("-hours", default=1, type=float)
parser.add_argument("-augment_angle", default=10., type=float)
parser.add_argument("-wd", default=1e-4, type=float)
parser.add_argument("-do", default=0.0, type=float)
parser.add_argument("-validratio", default=0.2, type=float)
parser.add_argument("-copy_contour", action="store_true")
args = parser.parse_args()

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torchvision.transforms.functional import center_crop

from ml_tools.augment import random_rotate_and_center_crop
from ml_tools.scaler import process
from ml_tools.unet import UNet

torch.set_num_threads(args.nthreads)
"""
    Ready up the dataset

"""
ds = np.load(args.data)
import pickle
X, X_scaler = process(ds["epsilon_map"])
Y, Y_scaler = process(ds["leakage_map"], log=True)

ds = TensorDataset(X, Y)
num_samples = X.shape[0]
valid_samples = round(args.validratio * num_samples)
ds_train, ds_valid = random_split(ds, [len(ds) - valid_samples, valid_samples])
loader_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True)
loader_valid = DataLoader(ds_valid, batch_size=128, shuffle=False)

print(" -> Dataloaders ready")

"""
    Create the model

"""

dev = torch.device(args.device)
model = UNet(complexity=args.complexity)
#model.load_state_dict(torch.load(args.name))
model.to(dev)

optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
mse_criterion = nn.MSELoss()

if args.copy_contour:
    img_size = 172
else:
    img_size = 128

print(" -> Start Training")
profile = []
for e in range(args.epochs):
    model.train()
    epoch_loss = 0.0
    for epsilon, leakage in loader_train:
        optim.zero_grad()
        epsilon_mod, angles = random_rotate_and_center_crop(epsilon, max_angle_deg=args.augment_angle, copy_contour=args.copy_contour)
        leakage_mod, _ = random_rotate_and_center_crop(leakage, angles_deg=angles, copy_contour=args.copy_contour)
        pred_leakage = model(epsilon_mod.to(dev))
        loss = mse_criterion(pred_leakage, leakage_mod.to(dev))
        epoch_loss += loss.cpu().item()
        loss.backward()
        optim.step()
    # Validation
    val_loss = 0.0
    if valid_samples > 0:
        model.eval()
        with torch.no_grad():
            for epsilon, leakage in loader_valid:
                out = model(center_crop(epsilon, img_size).to(dev))
                loss = mse_criterion(out, center_crop(leakage, img_size).to(dev))
                val_loss += loss.item()
        epoch_loss /= len(loader_train)
        val_loss /= len(loader_valid)
    profile.append((epoch_loss, val_loss))
    current_time = time.time()
    spent_time = current_time - job_start 
    print(f"[{e} / {args.epochs}] {epoch_loss:.2e}, {val_loss:.2e}, {spent_time//3600} hours + {(spent_time % 3600)/3600}% elapsed. ")
    if spent_time >= args.hours * 3600:
        break


torch.save({
    "model": model.cpu().state_dict(), 
    "args": vars(args), 
    "profile": profile,
    "complexity": args.complexity,
    "xscaler": X_scaler, 
    "yscaler": Y_scaler 
    }, args.name)
