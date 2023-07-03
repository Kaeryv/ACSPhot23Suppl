# =========================================================================
# FDTD simulation of Vortex Phase Mask, created may20, updated nov21, feb22

# This script propagates a circularly polarized plane wave through a Vortex
# Phase Mask made of full-dielectric diamond subwavelength structures, aka 
# metasurfaces. The results are complex field maps of x and y polarization 
# monitored in slices at certain distance below the subwavelength structure
# inside the diamond substrate. These field maps are saved as npy files and
# can be used as raw data for eventual post processing. From these complex 
# field maps the right and left circular polarizations are then calculated,
# returning the leakage and the phase ramp which can be plotted. Finally,  
# the metric of performance (leakage) is calculated and returned upstream. 

# Note that 'hexa' pxl should be used for now (otherwise overlap with AGPM)

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# define the function and read the parameters

from argparse import Namespace

def vortex_annular(parsdict):
    args = Namespace(**parsdict)

    # -------------------------------------------------------------------------
    # define the simulation cell
    
    #  /________________ /    #    A circular polarized plane wave is incident
    # |  _____PML_____  |     #    from the air side onto the metasurface and
    # |.|....s.r.c....|.|     #    monitored inside the diamond substrate. 
    # |_|____A_I_R____|_|     #
    # |_|_METASURFACE_|_|     #    z   y
    # |.|....m.o.n....|.|     #         
    # | |____S_U_B____| |     #    | /  
    # |_______PML_______|/    #    |/__  x
    
    resolution = 10               #  pxl/um - 20 is the max we can afford
    dpml       = args.dpml        #  PML thickness - 8-12 is good (4 for tests)
    dpad       = args.dpad        #  air thickness above vortex mask
    gh         = args.gh          #  height of MS strucutre
    dsub       = args.dsub        #  substrate thickness outside PML, will be 
                                  #   extended into PML
    
    sx = args.sxp+2*dpml          #  cell size with PML layers
    sy = args.syp+2*dpml          #  sx: cell size, sxp: cell size without PMLs
    sz = dpml+dpad+gh+dsub+dpml   #
    print("Nominal cell size: ",sx,sy,sz)
    
    cell_size = mp.Vector3(sx,sy,sz)  #  define cell size with PMLs
    
    pml_layers = [mp.PML(thickness=dpml)]  #  make PMLs on all sides
    
    # -------------------------------------------------------------------------
    # define the source
    
    wvl = args.wvl  #  monochromatic source wavelength
    
    sourc = args.sourc
    if sourc=='LHC':
        a_x = 1     #  LHC incoming plane wave
        a_y = 0-1j  #
    elif sourc=='RHC':
        a_x = 1     #  RHC incoming plane wave (default)
        a_y = 0+1j  #
    else:
        print("Source 'sourc' must be 'LHC' or 'RHC' ! Stopping script.")
        exit()      # check for source, otherwise quit. 
    
    turnontime = 0  #  nonzero turnontime doesn't change too much
    
    sources = [mp.Source(mp.ContinuousSource(frequency=1/wvl,width=turnontime),
                    component=mp.Ex,
                    center=mp.Vector3(0,0,0.5*sz-dpml-0.5*dpad),
                    size=mp.Vector3(sx,sy,0),
                    amplitude=a_x),
               mp.Source(mp.ContinuousSource(frequency=1/wvl,width=turnontime),
                    component=mp.Ey,
                    center=mp.Vector3(0,0,0.5*sz-dpml-0.5*dpad),
                    size=mp.Vector3(sx,sy,0),
                    amplitude=a_y)]
    
    # -------------------------------------------------------------------------
    # define the geometry
    
    air = mp.Medium(index=1.00)      #  index of refraction is assumed to be..
    diamond = mp.Medium(index=2.38)  #  ..constant over used wavelength range
    
    geometry = [mp.Block(material=diamond,  #  substrate block
                         size=mp.Vector3(mp.inf,mp.inf,dsub+dpml),
                         center=mp.Vector3(0,0,-0.5*sz+0.5*(dpml+dsub)))]
    
    swa = args.swa*np.pi/180         #  sidewall angle was given in degrees
    
    # -------------------------------------------------------------------------
    # define AGPM-like circles around central MS pattern
   
    agpm_start = args.agpm_start     #  start with AGPM beyond this radius
    gdc = args.gdc                   #  filling factor / grating duty cycle
    gp = args.gp                     #  grating period
    jmax = np.int(1+np.ceil(np.sqrt(2)*(sx/2-agpm_start)/gp))     # THERE WAS A BUG HERE
    rmax = agpm_start+np.ceil(np.sqrt(2)*(sx/2-agpm_start)/gp)*gp # SOLVED ON 23/2/2022
    # max radius of grating lines to be used in geometry (must start from outer..
    # ..most lines since we use solid cone objects alternating air and diamond)
    # Note that rmax is used differently as in previous AGPM scripts: now it is..
    # ..in [mu], before it was in [gp] for some reason
    agpm_center = mp.Vector3(0,0,sz/2-dpml-dpad-gh/2)
    for j in range(jmax):
        geometry.append(mp.Cone(material=diamond,
                     center=agpm_center,
                     radius=rmax-j*gp+gdc*gp+gh*np.tan(swa),
                     radius2=rmax-j*gp+gdc*gp,
                     height=gh))
        geometry.append(mp.Cone(material=air,
                     center=agpm_center,
                     radius=rmax-j*gp-gh*np.tan(swa),
                     radius2=rmax-j*gp,
                     height=gh))
    # We start with a cone of air of radius
    # radius=rmax-(jmax-1)*gp
    r0 = rmax - (jmax-1)*gp
    ri = r0 / 6
    w = args.w
    r = args.r
    for i in range(len(r)):
        rri = ri + ri * i + r[i]
        rad1 = r0 - rri + w[i] / 2.0
        if rad1 < 0:
            rad1=1e-4
            break
        geometry.append(mp.Cone(material=diamond, center=agpm_center, radius=rad1, radius2=rad1, height=gh))
        rad2 = r0 - rri - w[i] / 2.0
        if rad2 < 0:
            rad2=1e-4
        geometry.append(mp.Cone(material=air, center=agpm_center, radius=rad2, radius2=rad2, height=gh))
    rad = w[-1]
    geometry.append(mp.Cone(material=diamond, center=agpm_center, radius=rad, radius2=rad, height=gh))




    xo = 0.              #  shift the finished pattern by some scalar
    yo = 0.              #  this is useful for tests of off-axis MS
                         #  it does not shift the AGPM circles
    
    # -------------------------------------------------------------------------
    # define simulation
    
    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        force_complex_fields=True)  # complex fields for phase
    
    # -------------------------------------------------------------------------
    # run simulation and get results
    
    rununtil = args.rununtil  #  simulation runtime, in meep units
    
    zslices = [sz/2-dpml-dpad-gh/2,   #  output only two slices, one for eps
               -sz/2+dpml+dsub*1/4]   #  and one for leakage / phaseramp

    
    eps_data = len(zslices)*[0]
    ex_data = len(zslices)*[0]
    ey_data = len(zslices)*[0]
    def get_zslices(sim):
        for i in range(len(zslices)):
            eps_data[i] = sim.get_array(center=mp.Vector3(0,0,zslices[i]),
                                        size=mp.Vector3(sx-2*dpml,sy-2*dpml,0),
                                        component=mp.Dielectric)
            ex_data[i] = sim.get_array(center=mp.Vector3(0,0,zslices[i]),
                                        size=mp.Vector3(sx-2*dpml,sy-2*dpml,0),
                                        component=mp.Ex)
            ey_data[i] = sim.get_array(center=mp.Vector3(0,0,zslices[i]),
                                        size=mp.Vector3(sx-2*dpml,sy-2*dpml,0),
                                        component=mp.Ey)
    
    sim.run(mp.at_end(get_zslices),until=rununtil)  #  run simulation
    
    # -------------------------------------------------------------------------
    # return the leakage

    er_data = 1./np.sqrt(2)*(np.array(ex_data)-1j*np.array(ey_data))
    el_data = 1./np.sqrt(2)*(np.array(ex_data)+1j*np.array(ey_data))
    # extreme caution with +/- !
    # np.array needed to be able to piecewise multiply by 1j. Was not needed in the plot_vortex.py script...

    er_int = np.abs(er_data)**2  #  calculate intensity of R/L
    el_int = np.abs(el_data)**2  #
    RHC_int = np.sum(er_int[-1,:,:])  #  calculate metric
    LHC_int = np.sum(el_int[-1,:,:])  #
    metric = RHC_int/(LHC_int+RHC_int)

    
    return metric, eps_data, ex_data, ey_data

