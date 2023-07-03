import numpy as np
import torch
from torchvision.transforms.functional import rotate, center_crop

'''
    Rotates each element of the batch by a random angle.
    Then applies center_crop to avoid undetermined corners.
'''
def random_rotate_and_center_crop(X, max_angle_deg=10, out_shape=(128, 128), angles_deg=None, copy_contour=False):
    bs = X.shape[0]
    x = np.linspace(-1, 1, 172)
    if copy_contour:
        out_shape=(172,172)
    XX, YY = np.meshgrid(x,x)
    mask = np.logical_not((XX**2+YY**2) < 0.49)
    output = torch.zeros((bs, 1, *out_shape))
    if angles_deg is None:
        random_angles_deg = np.random.rand((bs)) * max_angle_deg
    else:
        random_angles_deg = angles_deg
        assert len(angles_deg) == bs
    for i in range(bs):
        output[i] = center_crop(rotate(X[i], random_angles_deg[i]), out_shape[0])
    if copy_contour:
        output[:,:,mask] = X[:,:,mask]
    return output, random_angles_deg

