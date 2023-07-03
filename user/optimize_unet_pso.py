import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-fom", type=str, required=True)
parser.add_argument("-model", type=str, required=True)
parser.add_argument("-nagents", type=int, default=40)
parser.add_argument("-fevals", type=int, default=40)
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-output", type=str, default="workdir")
parser.add_argument("-nthreads", type=int, default=16)
parser.add_argument("-rules", nargs="+", default=[], type=int)

parser.add_argument("-device", type=str, default="cpu")
parser.add_argument("-max-stagnation", type=int, default=50)
args = parser.parse_args()

print("PSO OPTIMIZATION STARTED")

import sys
sys.path.append("..")
sys.path.append(".")
from hybris.optim import Optimizer

from keever.tools import ensure_file_directory_exists
from keever.algorithm import Algorithm
from keever.database import count_continuous_variables, countinuous_variables_boundaries
import keever
import numpy as np
import torch
torch.set_num_threads(args.nthreads)

ensure_file_directory_exists(args.output)
fom = Algorithm.from_json(keever.load(args.fom))
variables = count_continuous_variables(fom.config["variables"])
opt = Optimizer(num_agents=args.nagents, num_variables=[variables, 0], max_fevals=args.fevals)
vminmax = countinuous_variables_boundaries(fom.config["variables"])
opt.vmin = vminmax[0]
opt.vmax = vminmax[1]
#opt.weights[0,:] = 0.3
opt.reset(args.seed)


if args.rules:
    opt.set_rule(0, args.rules[0])

max_iters = args.fevals // args.nagents

all_configurations = list()
all_maps = list()
all_aptitudes = list()

current_best = np.inf
stagnation = 0
iterations = 0

while not opt.stop():
    x = opt.ask()
    all_configurations.append(x)
    ret = fom.action("evaluate-unet", {"pop": x, "model": args.model })
    y = ret["leakage"]
    all_maps.append(ret["maps"])
    all_aptitudes.append(y)
    iterations += 1
    if np.min(y) >= current_best:
        stagnation += 1
    else:
        current_best = np.min(y)
        stagnation = 0

    opt.tell(y)
    if stagnation > args.max_stagnation:
        break

img_size=172
all_maps = np.asarray(all_maps).reshape(iterations*args.nagents, img_size, img_size)
all_aptitudes = np.asarray(all_aptitudes).reshape(iterations*args.nagents)
all_configurations = np.asarray(all_configurations).reshape(iterations*args.nagents, -1)

np.savez_compressed(args.output, aptitudes=all_aptitudes, configs=all_configurations, maps=all_maps)

print(f"Finished optimizing with leakage {opt.profile[-1]}, saving to {args.output}.")