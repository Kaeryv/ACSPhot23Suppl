'''
    This module specifies how the data from the simulation should be loaded.
'''

import numpy as np
from tools import cmpt_leakage
from os.path import join
def __run__(individual_tag, workdir="", tmpdir=""):
    returns = {"epsilon_map": [], "leakage_map": [], "metric": []}
    for tag in individual_tag:
        data_file = join(workdir, tag + ".out.npz") 
        d = np.load(data_file)
        epsilon = d["eps"][0]
        leakage_map = cmpt_leakage(d["ex"], d["ey"])
        returns["epsilon_map"].append(epsilon)
        returns["leakage_map"].append(leakage_map)
        returns["metric"].append(d["leakage"])
    return returns

def __declares__():
    return ["leakage_map", "metric", "epsilon_map"]

def __requires__():
    '''
        Define the arguments of the __run__ function.
        - type: single (one individual at a time) population (whole population)
    '''
    return {"variables":["individual_tag", "workdir"]}
