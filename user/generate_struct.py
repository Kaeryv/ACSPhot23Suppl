"""
    The pitch:
    Generate a 2D rasterized structure like the one from meep.
    But faster and without meep's annoying presence.

    use for ML, plotting, etc.
    Based on lorenzo's plotting function
"""

import sys
sys.path.append(".")
import numpy as np
from PIL import Image, ImageDraw

def get_vertices_(dpml, dpad, gh, dsub, sxp, syp, swa, pxl, xc, yc, w, l, dpillar, gp, lp, angles=None):
    sx = sxp+2*dpml          #  cell size with PML layers
    sy = syp+2*dpml          #  sx: cell size, sxp: cell size without PMLs
    sz = dpml+dpad+gh+dsub+dpml   #
    swa = swa*np.pi/180         #  sidewall angle was given in degrees
    
    pxl = pxl  #  hexa or cart; hexa should be used for now

    l = l      #  length of each MS block
    w = w      #  width of each MS block
    xc = xc    #  center of each MS block in their pixel
    yc = yc    #  (along x,y for cart, along phi,r for hexa!)

    # Use only 10x10 (cart) or 11x11x11 pixels (hexa) for now, since we
    # replicate the segments manually here (symmetry)

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

    dpillar = dpillar  # add central pillar
    g = gp*np.arange(0.,len(l))    #  define the coordinates of
    h = gp*np.arange(1.,len(l)+1)  #  the center of each pxl
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
    lp = lp       #  topological charge
    gamma = np.pi/3    #  60deg hexa axis angle
    ce = np.sqrt(xx**2+yy**2-2*xx*yy*np.cos(gamma))  # cos/sin laws
    beta = np.arcsin(xx/ce*np.sin(gamma))  # azimuth angle of pixel
    alpha = lp/2*beta  #  calculate rotation of blocks. '+', not '-' !
    if angles is not None:
        alpha += angles
    listofblocks = np.empty((15,4,2))  # 15: manual count for 11x11x11
    
    # rotate the blocks in their pixel according to topological charge
    # then add pixel coordinates
    blocknumber = 0
    for pxlcoordx in range(len(xx)):
      for pxlcoordy in range(pxlcoordx+1):
        rot = np.array([[np.cos(alpha[pxlcoordx,pxlcoordy]),np.sin(alpha[pxlcoordx,pxlcoordy])],
                        [-np.sin(alpha[pxlcoordx,pxlcoordy]),np.cos(alpha[pxlcoordx,pxlcoordy])]])
        relvertices = np.array([np.dot(rot,np.array([l[pxlcoordx,pxlcoordy]/2,w[pxlcoordx,pxlcoordy]/2])),
                                np.dot(rot,np.array([l[pxlcoordx,pxlcoordy]/2,-w[pxlcoordx,pxlcoordy]/2])),
                                np.dot(rot,np.array([-l[pxlcoordx,pxlcoordy]/2,-w[pxlcoordx,pxlcoordy]/2])),
                                np.dot(rot,np.array([-l[pxlcoordx,pxlcoordy]/2,w[pxlcoordx,pxlcoordy]/2]))])
        listofblocks[blocknumber] = np.array([relvertices[i]+np.array([np.cos(np.pi/6)*xx[pxlcoordx,pxlcoordy],yy[pxlcoordx,pxlcoordy]-np.sin(np.pi/6)*xx[pxlcoordx,pxlcoordy]]) for i in range(4)])
        blocknumber+=1
    
    return listofblocks, (sx, sy, sz)


'''
    New version, the only good version
'''

def get_epsilon_map(raw, type="metasurface"):
    if type == "metasurface":
        return get_epsilon_map_metasurface(raw)
    elif type == "annular":
        return get_epsilon_map_annular(raw)
    elif type == "angles":
        return get_epsilon_map_metasurface(raw, angles=True)


def get_epsilon_map_metasurface(raw, angles=False):
    gp            = 1.21    #  float  #  mu      #  AGPM grating period, and MS pixel period
    gdc           = .6509   #  float  #          #  AGPM filling factor (grating duty cycle)
    gh            = 4.5753  #  float  #  mu      #  AGPM grating depth (grating height)
    swa           = 0.      #  float  #  deg     #  side wall angle (0 for proof-of-concept)
    pxl           = 'hexa'  #  string #          #  cartesian or hexagonal pixels ?
    agpm_start    = 6.05    #  float  #  mu !    #  use AGPM pattern beyond this radius
    lp            = 2       #  int !  #          #  topological charge (2=AGPM)
    dpml          = 8.      #  float  #  mu      #  PML thickness (on all sides)
    dpad          = 3.      #  float  #  mu      #  air padding thickness (above MS)
    dsub          = 3.      #  float  #  mu      #  diamond substrate thickness (below MS)
    sxp           = gp*14   #  float  #  mu      #  size of the simulation cell without PMLs..
    syp           = gp*14   #  float  #  mu      #  ..(5 MS blocks + 2 AGPM periods) * 2

    if angles:
        w, l, xc, yc, dpillar, angles = raw
        vertices, s = get_vertices_(dpml, dpad, gh, dsub, sxp, syp, swa, pxl, xc, yc, w, l, dpillar, gp, lp, angles=angles)

    else:
        w, l, xc, yc, dpillar = raw
        vertices, s = get_vertices_(dpml, dpad, gh, dsub, sxp, syp, swa, pxl, xc, yc, w, l, dpillar, gp, lp)

    s = (34, )
    
    N=1024
    img = Image.new("L", (N, N), 0)
    draw = ImageDraw.Draw(img)
    qqs = np.pi/6 + np.arange(0,2*np.pi,np.pi/3)  #  repeat for 6 'hexants'
    
    sx = sxp+2*dpml          #  cell size with PML layers
    sy = syp+2*dpml          #  sx: cell size, sxp: cell size without PMLs
    rmax = agpm_start+np.ceil(np.sqrt(2)*(sx/2-agpm_start)/gp)*gp # SOLVED ON 23/2/2022
    #rmax*=0.98
    for j in range(5):
        r=(rmax-j*2*gp+2*gdc*gp)/s[0]*N
        draw.ellipse((N/2-r, N/2-r, N/2+r, N/2+r), fill=1)
        r=(rmax-j*2*gp)/s[0]*N
        draw.ellipse((N/2-r, N/2-r, N/2+r, N/2+r), fill=0)
    for qq in qqs:
        rot = np.array([[np.cos(qq),np.sin(qq)], [-np.sin(qq),np.cos(qq)]]) #  not mirror
        for blocknumber in range(len(vertices)):
            vert = np.array([ rot @ point for point in vertices[blocknumber] ])
            vert /= s[0]/2
            vert *= N
            vert += N/2
            draw.polygon(vert.astype(int).flatten().tolist(), fill=1)
    r = 2*dpillar / s[0] / 2 * N
    draw.ellipse((N/2-r, N/2-r, N/2+r, N/2+r), fill=1)
    img = Image.fromarray(np.array(img).astype(np.float32))
    img = img.resize((172, 172), resample=Image.BILINEAR)
    img = np.array(img)
    diamond = 2.38**2
    img = 1.0 + img * (diamond-1.0)
    return img


def get_epsilon_map_annular(raw):
    gp            = 1.21    #  float  #  mu      #  AGPM grating period, and MS pixel period
    gdc           = .6509   #  float  #          #  AGPM filling factor (grating duty cycle)
    agpm_start    = 6.05    #  float  #  mu !    #  use AGPM pattern beyond this radius
    dpml          = 8.      #  float  #  mu      #  PML thickness (on all sides)
    sxp           = gp*14   #  float  #  mu      #  size of the simulation cell without PMLs..
    syp           = gp*14   #  float  #  mu      #  ..(5 MS blocks + 2 AGPM periods) * 2

    radiuses, ring_widths = raw
    #else:
        #w, l, xc, yc, dpillar = ann(X, *ranges)
    sx = sxp+2*dpml          #  cell size with PML layers
    sy = syp+2*dpml          #  sx: cell size, sxp: cell size without PMLs
    s = (34.25, )
    
    N=2048
    img = Image.new("L", (N, N), 0)
    draw = ImageDraw.Draw(img)
    
    
    jmax = np.int(1+np.ceil(np.sqrt(2)*(s[0]/2-agpm_start)/gp))
    # jmax is correct
    
    rmax = agpm_start+np.ceil(np.sqrt(2)*(s[0]/2-agpm_start)/gp)*gp # SOLVED ON 23/2/2022
    #rmax*=0.98
    for j in range(5):
        r=(rmax-j*2*gp+2*gdc*gp)/s[0]*N
        draw.ellipse((N/2-r, N/2-r, N/2+r, N/2+r), fill=1, outline=1)
        r=(rmax-j*2*gp)/s[0]*N
        draw.ellipse((N/2-r, N/2-r, N/2+r, N/2+r), fill=0)
    r0 = rmax - (jmax-1)*gp
    ri = r0 / 6
    for i in range(len(radiuses)):
        rri = ri * (i+1) + radiuses[i] # Radius of the ring
        r = r0 - rri + ring_widths[i] / 2  # outer radius
        r = r / s[0] * N * 2
        draw.ellipse((N/2-r, N/2-r, N/2+r, N/2+r), fill=1)
        
        r = r0 - rri - ring_widths[i] / 2 # outer radius
        r = r / s[0] * N * 2
        draw.ellipse((N/2-r, N/2-r, N/2+r, N/2+r), fill=0)
    # Draw the central pillar    
    dpillar = ring_widths[-1] * 2
    r = 2*dpillar / s[0] / 2 * N
    draw.ellipse((N/2-r, N/2-r, N/2+r, N/2+r), fill=1)
    img = Image.fromarray(np.array(img).astype(np.float32))
    img = img.resize((172, 172), resample=Image.BILINEAR)
    img = np.array(img)
    diamond = 2.38**2
    img = 1.0 + img * (diamond-1.0)
    return img