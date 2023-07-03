import torch
from scipy.ndimage import binary_erosion, binary_dilation
import numpy as np


def process(x, log=False, scaler=None):
    if log:
        x = np.log10(x)
    imshape = x[0].shape
    if scaler:
        x = scaler.transform(x)
        return torch.from_numpy(x.reshape(-1, 1, *imshape)).type(torch.float32)
    else:
        scaler = Scaler()
        x = scaler.fit_transform(x)
    
        return torch.from_numpy(x.reshape(-1, 1, *imshape)).type(torch.float32), scaler

class Scaler():
    def __init__(self):
        self.mean = 0
        self.std = 0
    def fit(self, X):
        self.mean = X.mean()
        self.std = X.std()
    def transform(self, X):
        return (X-self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return self.mean + (X * self.std)
