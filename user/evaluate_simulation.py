# AGPM center simulation
# ======================

from argparse import ArgumentParser

parser = ArgumentParser("Dispatch to AGPM Sims by Lorenzo Koning")
parser.add_argument("-npz", type=str, required=True)
parser.add_argument("-type", type=str, required=True)
parser.add_argument("-res", type=float, default=10.)
args = parser.parse_args()


import numpy as np

config = np.load(args.npz)

gp = 1.21
parameters = {
    "gp": gp,
    "gdc": .6509,
    "gh": 4.5753,
    "swa": 0.,
    "agpm_start": 6.05,        #  use AGPM pattern beyond this radius
    "lp": 2,                   #  topological charge (2=AGPM)
    "size": 5,                 #  size of MS block parameter matrices (l,w,..)
    "resolution": args.res,    #  MEEP resolution (10-15, 5 ok for test)
    "wvl": 3.5,
    "sourc": "RHC",
    "dpml": 8.,
    "dpad": 3.,
    "dsub": 3.,
    "rununtil": 100,
    "sxp": 14*gp,
    "syp": 14*gp
}

if args.type == "metasurface":
    from konig.sim_fins import vortex_AGPM as sim

    parameters.update({
        "pxl": "hexa",             #  cartesian or hexagonal pixels ?
        "l": config['l'],
        "w": config['w'],
        "xc": config['xc'],
        "yc": config['yc'],
        "dpillar": config['dpillar'],
    })
    leakage,eps,ex,ey = sim(parameters) # run meep, save and plot results (if needed), return the leakage
    np.savez_compressed(args.npz.replace(".in.", ".out."), leakage=leakage, eps=eps, ex=ex, ey=ey)

elif args.type == "annular":
    from konig.sim_annular import vortex_annular as sim
    parameters.update({
        "r": config['r'],
        "w": config['w']
    })

    leakage,eps,ex,ey = sim(parameters) # run meep, save and plot results (if needed), return the leakage
    np.savez_compressed(args.npz.replace(".in.", ".out."), leakage=leakage, eps=eps, ex=ex, ey=ey)

elif args.type == "angles":
    from konig.sim_fins import vortex_AGPM as sim
    ww = np.sqrt(config['w'])
    ll = config['l'] * ww
    parameters.update({
        "pxl": "hexa",             #  cartesian or hexagonal pixels ?
        "l": ll,
        "w": ww,
        "xc": config['xc'],
        "yc": config['yc'],
        "dpillar": config['dpillar'],
        "angles": config["angles"]
    })
    leakage,eps,ex,ey = sim(parameters) # run meep, save and plot results (if needed), return the leakage
    np.savez_compressed(args.npz.replace(".in.", ".out."), leakage=leakage, eps=eps, ex=ex, ey=ey)
