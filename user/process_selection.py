'''
    We want to select original, performant configurations.
    BOTH. We do not only need performing ones.
    We want to increase the reliability of the model.
'''
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-population', type=str, required=True)
parser.add_argument('-model', type=str, required=True)
parser.add_argument('-optimization-archives', nargs='+', required=True)
parser.add_argument('-nclusters', type=int, required=True)
parser.add_argument('-threshold', type=float, default=0.08)
parser.add_argument('-nselected', type=int, required=True)
parser.add_argument('-subsampling', type=int, default=5)
parser.add_argument('-output', type=str, required=True)
parser.add_argument('-img_size', type=int, default=128)
parser.add_argument('-clustering_alg', type=str, default="kmeans")
args = parser.parse_args()

import os
from os.path import join


import pickle
from ml_tools.scaler import process
from ml_tools.unet import load

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from torchvision.transforms.functional import center_crop
import torch

clustering_algorithms = { "kmeans": KMeans, "agglomerative_clustering": AgglomerativeClustering }




def similarity_metric(X, Y):
    '''
        returns similarity in [0, 1] where @ 1 structures are perfectly
        dissimilar.
    '''
    img_size = X.shape[0]
    max_pxl_diff = np.max(X) - np.min(X) # Guaranteed constant
    num_px = img_size**2
    max_abs_diff = num_px * max_pxl_diff

    return np.sum(np.abs(Y-X)) / max_abs_diff

def sample_not_too_close(dest_array, src_array, sample_id, threshold=0.05):
    smallest_distance = np.inf
    src = src_array[sample_id]
    distances = list()
    for dest in dest_array: # Compute the euc. distance to all elements in dest_array
        distances.append(similarity_metric(dest, src))
    if len(dest_array) > 0:
        smallest_distance = np.min(distances)
    return smallest_distance > threshold

def process_optimizer_archive(name):
    data = np.load(name)
    ss = args.subsampling
    configs = data["configs"][::ss]
    aptitudes = data["aptitudes"][::ss].real
    maps = center_crop(process(data["maps"][::ss].real, log=False, scaler=xscaler), args.img_size).numpy()

    nsamples = configs.shape[0]

    m_clustering = AgglomerativeClustering(args.nclusters)

    clusters = m_clustering.fit_predict(maps.reshape(nsamples, -1))
    nconfs = args.nselected
    # The bigger the deviation in the cluster, the more samples we take with a minimum of 1
    deviations = np.asarray([ np.std(aptitudes[clusters==i]) for i in range(args.nclusters)])
    deviations /= np.sum(deviations)
    nchoices = np.round(deviations * nconfs)
    nchoices[nchoices<1]=1
    indices = np.arange(nsamples)
    choices = list()
    for i in range(args.nclusters):
        probs = np.max(aptitudes)-aptitudes[clusters==i]
        probs /= np.sum(probs)
        choices.append(np.random.choice(indices[clusters==i], size=int(nchoices[i]), replace=False, p=probs))

    return { 
            "clusters": clusters,
            "aptitudes": aptitudes,
            "nchoices": nchoices,
            "choices": choices,
            "configs": configs,
            "maps": maps
    }






if __name__ == "__main__":
    N = args.img_size
    xscaler, _ = load(args.model, scalers=True, model=False)
    current_population = np.load(f"{args.population}")['epsilon_map']
    current_population = process(current_population, log=False, scaler=xscaler)
    current_population = center_crop(current_population, N).numpy()
    current_population = current_population.reshape(-1, N, N)
    
    from types import SimpleNamespace
    pop = [ e for e in current_population ]
    added_configs = list()

    for i, opti_filename in enumerate(args.optimization_archives):
        selection_data = process_optimizer_archive(opti_filename)
        #make_charts(**selection_data, name=args.name)
        selection_data = SimpleNamespace(**selection_data)
        choices = sorted([ e for l in selection_data.choices for e in l ])

        
        to_add = list()
        ito_add= list()
        for c in choices:
            if sample_not_too_close(to_add, selection_data.maps, c, threshold=args.threshold):
                to_add.append(selection_data.maps[c])
                ito_add.append(c)
        for i, c in enumerate(ito_add):
            if sample_not_too_close(pop, to_add, i, threshold=args.threshold):
                pop.append(to_add[i])
                added_configs.append(selection_data.configs[c])

    np.savez(args.output, variables=added_configs)
