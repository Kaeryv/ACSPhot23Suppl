from ml_tools.unet import load
from ml_tools.scaler import process
from tools import point2epsilon, unscaleY
from torchvision.transforms.functional import center_crop
import numpy as np
import torch
from functools import cache
# helper function to unscale leakage


img_size = 128
def __run__(pop, model, type, **kwargs):
    x = pop
    model, X_scaler, Y_scaler = load(model)
    """
    evaluate evaluates an array of AGPM parameters as mean leakage.
    :param x The array of configurations.
    :return leakage, eps The predicted mean leakages and the corresponding structures (128x128)
    """
    eps = np.empty((x.shape[0], 172, 172))
    for i in range(x.shape[0]):
        eps[i] = point2epsilon(x[i], type)
    maps = eps.copy()
    with torch.no_grad():
        eps = process(eps, log=False, scaler=X_scaler)
        eps = center_crop(eps, img_size)
        leakage = model(eps)
        leakage = 10**unscaleY(leakage.numpy(), Y_scaler).real
        leakage= np.mean(leakage, axis=(1, 2, 3))
    return {"leakage": leakage, "maps": maps }

def __declares__():
    return []

def __requires__():
    return {"type": "population", "variables":["pop", "model","type"]}

@cache
def load(model_path):
    from ml_tools.unet import load as load_unet
    return load_unet(model_path, "cpu", scalers=True, complexity=3)