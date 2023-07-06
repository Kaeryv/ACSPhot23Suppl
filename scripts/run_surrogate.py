from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-project", type=str, required=True)
parser.add_argument("-epochs", type=int, required=True)
args = parser.parse_args()

import os
assert(os.path.isfile(args.project))


import sys
sys.path.append(".")

from keever.algorithm import ModelManager
import yaml

import numpy as np
from tqdm import trange

config = yaml.safe_load(open(args.project, "r"))

mm = ModelManager()
mm.load_state_dict(config)
mm.get("doe").save("doe-no-evaluations")

eval = mm.get("fom").action("evaluate-fdtd", args={"population": mm.get("doe")})
mm.get("doe").update_entries(eval["individual_tag"], {key: eval[key] for key in ["epsilon_map", "leakage_map", "metric"]})
mm.get("main").clear()
mm.get("main").merge(mm.get("doe"))
mm.get("unet").action("train", args={"dataset": mm.get("main")})

for i in trange(args.epochs):
    opt_files = mm.get("pso").reload().action("run", args={"fom": mm.get("fom"), "seed": np.random.randint(999999, size=10).tolist(), "workdir": "wd/"})
    selections = mm.get("sel").reload().action("run", args={"dataset": mm.get("main"), "optimizer_archive": opt_files["output"], "workdir":"wd"})
    mm.get("selected").append_npz_keys(selections["output"], ["variables"])
    eval = mm.get("fom").action("evaluate-fdtd", args={"population": mm.get("selected")})
    mm.get("selected").update_entries(eval["individual_tag"], {key: eval[key] for key in ["epsilon_map", "leakage_map", "metric"]})
    mm.get("main").merge(mm.get("selected"))
    mm.get("unet").action("train", args={"dataset": mm.get("main")})
    mm.get("selected").clear()
    mm.get("main").save(f"wd/population_epoch_{i:03}")

