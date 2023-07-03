from os.path import join
from user.tools import metasurface_raw_to_npz, angle_raw_to_npz, annular_raw_to_npz

type2converter = {
    "angles": angle_raw_to_npz,
    "annular": annular_raw_to_npz,
    "metasurface": metasurface_raw_to_npz
}

def __run__(population, workdir="", tmpdir="", type=""):
    tags = list()
    for indiv, props in population:
        if type in  type2converter.keys():
            type2converter[type](props["variables"], join(workdir, indiv + ".in.npz"))
        else:
            print(f"Unrecognized type: {type}.")
        
        tags.append(indiv)

    return {"individual_tag": tags}

def __requires__():
    return {"variables":["population", "workdir", "type"]}

def __declares__():
    return ["individual_tag"]
