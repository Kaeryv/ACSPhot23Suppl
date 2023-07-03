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

def vortex_AGPM(parsdict):
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
    
    resolution = args.resolution  #  pxl/um - 20 is the max we can afford
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
    for j in range(jmax):
        geometry.append(mp.Cone(material=diamond,
                     center=mp.Vector3(0,0,sz/2-dpml-dpad-gh/2),
                     radius=rmax-j*gp+gdc*gp+gh*np.tan(swa),
                     radius2=rmax-j*gp+gdc*gp,
                     height=gh))
        geometry.append(mp.Cone(material=air,
                     center=mp.Vector3(0,0,sz/2-dpml-dpad-gh/2),
                     radius=rmax-j*gp-gh*np.tan(swa),
                     radius2=rmax-j*gp,
                     height=gh))

    # -------------------------------------------------------------------------
    # define central MS pattern
   
    #  The parameters of each block will be passed as np.tril
    #  of 5x5=15 (cart) or 5x5=11 (hexa) float values
    #
    #  Note that cartesian grid doesn't make much sense at the moment, since it
    #  doesn't transition nicely into the AGPM pattern - crop / remove the pxls 
    #  overlapping with the circles would be an option, but no priority now
    #
    #  case cartesian grid     #  case hexagonal grid
    #                          #
    #    :  |  :  :  :  :  :   #   /  \_|_/  \__/  \__/
    #   .:..|..:..:..:..:..:   #   \__/4|0\__/  \__/  \
    #    :  |40:41:42:43:44:   #   /  \_|_/41\__/  \__/
    #   .:..|..:..:..:..:..:   #   \__/3|0\__/42\__/  \
    #    :  |30:31:32:33:  :   #   /  \_|_/31\__/  \__/
    #   .:..|..:..:..:..:..:   #   \__/2|0\__/32\__/  \
    #    :  |20:21:22:  :  :   #   /  \_|_/21\__/  \__/
    #   .:..|..:..:..:..:..:   #   \__/1|0\__/  \__/  \
    #    :  |10:11:  :  :  :   #   /  \_|_/11\__/  \__/
    #   .:..|..:..:..:..:..:   #   \__/0|0\__/  \__/  \
    #    :  |00:  :  :  :  :   #   /  \_|_/  \__/  \__/
    #   _:__|__:__:__:__:__:   #   \__/_|_\__/__\__/__\
    #    :  |  :  :  :  :  :   #   /  \_|_/  \__/  \__/
    #                          #
    #     y      offsets are   #    y     offsets are 
    #     |      along cart-   #    |     defined in 
    #     |__ x  esian axes    #     \ x  polar coords
    #                          #
    #  center of cartesian     #  center of hexagonal grid 
    #  grid is between 4 pxl   #  is inside 1 pxl (sym)
    #                          #  -> use a central pillar

    pxl = args.pxl  #  hexa or cart; hexa should be used for now

    l = args.l      #  length of each MS block
    w = args.w      #  width of each MS block
    xc = args.xc    #  center of each MS block in their pixel
    yc = args.yc    #  (along x,y for cart, along phi,r for hexa!)

    # Use only 10x10 (cart) or 11x11x11 pixels (hexa) for now, since we
    # replicate the segments manually here (symmetry)

    # for cartesian grid, replicate the 1/8 segment so to have a nice 1/4
    # segment that can be rotated without duplications
    if pxl=='cart':
      l[0,1]=l[1,0];w[0,1]=w[1,0];xc[0,1]=yc[1,0];yc[0,1]=xc[1,0]
      l[0,2]=l[2,0];w[0,2]=w[2,0];xc[0,2]=yc[2,0];yc[0,2]=xc[2,0]
      l[0,3]=l[3,0];w[0,3]=w[3,0];xc[0,3]=yc[3,0];yc[0,3]=xc[3,0]
      l[0,4]=l[4,0];w[0,4]=w[4,0];xc[0,4]=yc[4,0];yc[0,4]=xc[4,0]
      l[1,2]=l[2,1];w[1,2]=w[2,1];xc[1,2]=yc[2,1];yc[1,2]=xc[2,1]
      l[1,3]=l[3,1];w[1,3]=w[3,1];xc[1,3]=yc[3,1];yc[1,3]=xc[3,1]
      l[1,4]=l[4,1];w[1,4]=w[4,1];xc[1,4]=yc[4,1];yc[1,4]=xc[4,1]
      l[2,3]=l[3,2];w[2,3]=w[3,2];xc[2,3]=yc[3,2];yc[2,3]=xc[3,2]
      l[2,4]=l[4,2];w[2,4]=w[4,2];xc[2,4]=yc[4,2];yc[2,4]=xc[4,2]
      l[3,4]=l[4,3];w[3,4]=w[4,3];xc[3,4]=yc[4,3];yc[3,4]=xc[4,3]
      # note that xc and yc are mirrored by symmetry
      # xc and yc: make sure, that the diagonals are 0, i.e. not shifted 
      #  (symmetry, and quadrant multiplication would result in overlap)
      for i in range(len(xc)):
        if not xc[i,i]==0:
          print('xc: diagonals are not zero ! setting to zero...')
          xc[i,i]=0
        if not yc[i,i]==0:
          print('yc: diagonals are not zero ! setting to zero...')
          yc[i,i]=0

    # for hexagonal grid, replicate the 1/12 segment so to have a nice 1/6
    # segment that can be rotated without duplications (except center)
    if pxl=='hexa':
      l[2,2]=l[2,1];w[2,2]=w[2,1];xc[2,2]=-xc[2,1];yc[2,2]=yc[2,1]
      l[3,3]=l[3,1];w[3,3]=w[3,1];xc[3,3]=-xc[3,1];yc[3,3]=yc[3,1]
      l[4,3]=l[4,2];w[4,3]=w[4,2];xc[4,3]=-xc[4,2];yc[4,3]=yc[4,2]
      l[4,4]=l[4,1];w[4,4]=w[4,1];xc[4,4]=-xc[4,1];yc[4,4]=yc[4,1]
      # note that xc and yx are now polar coords (x=phi,y=r) but still [mu]
      # xc: make sure, that the phi diagonals are 0, i.e. not shifted
      #  (symmetry, and quadrant multiplication would result in overlap)
      # yc: radial displacement allowed for all pixels, including diagonals
      #  (it doesn't break the symmetry)
      # note that it's not [i,i] because of hexa 120deg axes
      # note that it's also [i,0] for phi because of pixelization symmetry
      for i in range(np.int(len(xc)/2)):
        if not xc[2*i+1,i+1]==0:
          print('xc: diagonals are not zero ! setting to zero...')
          xc[2*i+1,i+1]=0
      for i in range(len(xc)):
        if not xc[i,0]==0:
          print('xc: verticals are not zero ! setting to zero...')
          xc[i,0]=0
    
    xo = 0.              #  shift the finished pattern by some scalar
    yo = 0.              #  this is useful for tests of off-axis MS
                         #  it does not shift the AGPM circles

    # ---------------------------------------------------------------------
    # cartesian grid

    if pxl=='cart':
        print('cartesian grid...')
        print('/!\ Cartesian grid doesn t make too much sense, since it does not transition nicely into AGPM circles...')
        g = args.gp*np.arange(.5,len(l))  #  define the coordinates of the
        h = args.gp*np.arange(.5,len(l))  #  center of each pixel
        gg,hh = np.meshgrid(g,h)          #
        xx,yy = gg+xc,hh+yc               #  add the offsets inside the pxl
        lp = args.lp                      #  topological charge
        alpha = -lp/2*np.arctan(yy/xx)    #  calculate rotation of blocks
        
        listofblocks = np.empty((np.int(len(l)**2),4,2))  # n x n patch - all blocks are defined (although only n*(n+1)/2 blocks are independent)
        
        # rotate the blocks in their pxl according to the topological charge
        # then add pixel coordinates
        block_count = 0
        for pxlcoordx in range(len(xx[0])):
          for pxlcoordy in range(len(yy[0])):
            wxy = w[pxlcoordx,pxlcoordy]
            lxy = l[pxlcoordx,pxlcoordy]
            rot = np.array([[np.cos(alpha[pxlcoordx,pxlcoordy]),np.sin(alpha[pxlcoordx,pxlcoordy])],
                            [-np.sin(alpha[pxlcoordx,pxlcoordy]),np.cos(alpha[pxlcoordx,pxlcoordy])]])
            relvertices = np.array([np.dot(rot,np.array([wxy/2,lxy/2])),
                                    np.dot(rot,np.array([wxy/2,-lxy/2])),
                                    np.dot(rot,np.array([-wxy/2,-lxy/2])),
                                    np.dot(rot,np.array([-w[pxlcoordx,pxlcoordy]/2,l[pxlcoordx,pxlcoordy]/2]))])
            listofblocks[block_count] = np.array([relvertices[i]+np.array([xx[pxlcoordx,pxlcoordy],yy[pxlcoordx,pxlcoordy]]) for i in range(4)])
            block_count+=1
        
        sign_x = [1,1,-1,-1]  #  repeat for 4 quadrants (mirror)
        sign_y = [1,-1,-1,1]  #
        for quadrant in range(4):
            for blocknumber in range(block_count):
                designvertices = listofblocks[blocknumber]
                vertices = []  #  define vertices for 1 quadrant
                for vertexnumber in range(len(designvertices)):
                    x = designvertices[vertexnumber][0]
                    y = designvertices[vertexnumber][1]
                    vertices.append(mp.Vector3(xo+sign_x[quadrant]*x,
                                               yo+sign_y[quadrant]*y,
                                               sz/2-dpml-dpad))
                    #  we define the prism vertices at the air side of the MS..
                print(len(vertices))
                geometry.append(mp.Prism(vertices,
                                         height=gh,
                                         axis=mp.Vector3(0,0,-1),
                                         sidewall_angle=swa,
                                         material=diamond))
                #  ...together with inverted axis (0,0,-1) because of sidewalls
        
    # ---------------------------------------------------------------------
    # hexagonal grid

    elif pxl=='hexa':
        print('hexagonal grid...')
        print('BEWARE of non-orthogonal axes (120deg) ! That is, xc,yc have to be defined in polar coordinates (xc=phi,yc=r).')

        dpillar = args.dpillar  # add central pillar
        geometry.append(mp.Cone(material=diamond,
                     center=mp.Vector3(0,0,sz/2-dpml-dpad-gh/2),
                     radius=dpillar/2+gh*np.tan(swa),
                     radius2=dpillar/2,
                     height=gh))

        g = args.gp*np.arange(0.,len(l))    #  define the coordinates of
        h = args.gp*np.arange(1.,len(l)+1)  #  the center of each pxl
        gg,hh = np.meshgrid(g,h)
        # define the rotated orthogonal coord sys of each pxl center, and 
        # add xc,yc along these directions; but calculate the rotations of
        # the blocks according to their true position after adding xc,yc, 
        # using beta, not beta1.
        gamma1 = np.pi/3   #  60deg hexa axis angle
        delta = np.pi/6    #  30 (!) deg axis angle for coord transformation
        ce1 = np.sqrt(gg**2+hh**2-2*gg*hh*np.cos(gamma1))  # cos/sin laws
        beta1 = np.arcsin(gg/ce1*np.sin(gamma1))  # azimuth angle of pixel
        xx,yy = gg+1/np.cos(delta)*(np.cos(beta1)*xc[:,:]+np.sin(beta1)*yc[:,:]),hh+np.tan(delta)*(np.cos(beta1)*xc[:,:]+np.sin(beta1)*yc[:,:])-np.sin(beta1)*xc[:,:]+np.cos(beta1)*yc[:,:] #  this is: add the offsets in polar coords converted to 120deg basis
        lp = args.lp       #  topological charge
        gamma = np.pi/3    #  60deg hexa axis angle
        ce = np.sqrt(xx**2+yy**2-2*xx*yy*np.cos(gamma))  # cos/sin laws
        beta = np.arcsin(xx/ce*np.sin(gamma))  # azimuth angle of pixel
        alpha = lp/2*beta  #  calculate rotation of blocks. '+', not '-' !
        if "angles" in vars(args):
          alpha += args.angles
        #listofblocks = np.empty((15,4,2))  # 15: manual count for 11x11x11
        listofblocks = list()
        # rotate the blocks in their pixel according to topological charge
        # then add pixel coordinates
        for pxlcoordx in range(len(xx)):
          for pxlcoordy in range(pxlcoordx+1):
            # Get the sizes
            lxy = l[pxlcoordx,pxlcoordy]
            wxy = w[pxlcoordx,pxlcoordy]
            # If any of those is near zero, do not put the block
            if abs(lxy) < 1e-3 or abs(wxy) < 1e-3:
              continue
            alpha_xy = alpha[pxlcoordx,pxlcoordy]
            rot = np.array([[np.cos(alpha_xy),np.sin(alpha_xy)],
                            [-np.sin(alpha_xy),np.cos(alpha_xy)]])
 
            relvertices = np.array([np.dot(rot,np.array([lxy/2,wxy/2])),
                                    np.dot(rot,np.array([lxy/2,-wxy/2])),
                                    np.dot(rot,np.array([-lxy/2,-wxy/2])),
                                    np.dot(rot,np.array([-lxy/2,wxy/2]))])
            listofblocks.append(np.array([relvertices[i]+np.array([np.cos(np.pi/6)*xx[pxlcoordx,pxlcoordy],yy[pxlcoordx,pxlcoordy]-np.sin(np.pi/6)*xx[pxlcoordx,pxlcoordy]]) for i in range(4)]))
        
        qqs = np.arange(0,2*np.pi,np.pi/3)  #  repeat for 6 'hexants'
        for qq in qqs:
            rot = np.array([[np.cos(qq),np.sin(qq)],   #  rotation is needed
                            [-np.sin(qq),np.cos(qq)]]) #  not mirror
            for designvertices in listofblocks:
                vertices = []  #  define vertices for 1 'hexant' (1/6 of mask)
                for vertexnumber in range(len(designvertices)):
                    x = designvertices[vertexnumber][0]
                    y = designvertices[vertexnumber][1]
                    vertices.append(mp.Vector3(np.dot(rot,np.array([x,y]))[0],
                                               np.dot(rot,np.array([x,y]))[1],
                                               sz/2-dpml-dpad))
                    #  we define the prism vertices at the air side of the MS..
                geometry.append(mp.Prism(vertices,
                                         height=gh,
                                         axis=mp.Vector3(0,0,-1),
                                         sidewall_angle=swa,
                                         material=diamond))
                #  ...together with inverted axis (0,0,-1) because of sidewalls
        
        # ---------------------------------------------------------------------
        # plot pattern if requested
        
       
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

    zlabels = ['inside grating',
               '2.25mu below grating']
    
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

