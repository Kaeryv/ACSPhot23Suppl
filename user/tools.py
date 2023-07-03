import numpy as np

def cmpt_leakage(ex, ey):
    ex, ey = np.asarray(ex[1]), np.asarray(ey[1])
    er = 1. / np.sqrt(2) * (ex - 1j * ey)
    el = 1. / np.sqrt(2) * (ex + 1j * ey)
    er_int = np.abs(er)**2
    el_int = np.abs(el)**2
    return  er_int / (np.mean(el_int) + np.mean(er_int))


# Masks for the metasurface definition
w_l_yc_mask = np.tril(np.ones((5, 5))) > 0
w_l_yc_mask[2, 2] = False; w_l_yc_mask[3, 3] = False; w_l_yc_mask[4, 4] = False; w_l_yc_mask[4, 3] = False; 
xc_mask = np.zeros((5, 5)) > 0
xc_mask[2:5, 1] = True; xc_mask[4, 2] = True
tril_mask = np.tril(np.ones((5, 5))) > 0

def trilify(X, mask=tril_mask):
    ''' Create tril matric with elements from X placed at positions mask. '''
    assert np.count_nonzero(mask) == len(X), "There must be len(mask) Trues in the mask."
    output = np.zeros((5, 5))
    output[mask] = X
    return output

def metasurface_vec2mat(X, angles=False):
    w =  trilify( X[11:22], mask=w_l_yc_mask)
    l =  trilify( X[ 0:11], mask=w_l_yc_mask)
    xc = trilify( X[22:26], mask=xc_mask    )
    yc = trilify( X[26:37], mask=w_l_yc_mask)
    dpillar = X[37]
    if angles:
        angles = trilify(X[38:], mask=w_l_yc_mask)
        return w, l, xc, yc, dpillar, angles
    else:
        return w, l, xc, yc, dpillar
    
def annular_raw_to_npz(X, filename):
    assert X.shape[0] == 9
    r = X[0:4]
    w = X[4:]
    np.savez_compressed(filename, r=r, w=w)

def angle_raw_to_npz(X, filename):
    assert X.shape[0] == 38+11
    w, l, xc, yc, dpillar, angles = metasurface_vec2mat(X, angles=True)
    np.savez_compressed(filename, l=l, w=w, xc=xc, yc=yc, dpillar=dpillar, angles=angles)

def metasurface_raw_to_npz(X, filename):
    assert X.shape[0] == 38
    w, l, xc, yc, dpillar = metasurface_vec2mat(X)
    np.savez_compressed(filename, l=l, w=w, xc=xc, yc=yc, dpillar=dpillar)


img_size =  172
def unscaleY(x, scaler):
    xshape = x.shape
    return scaler.inverse_transform(x.reshape(x.shape[0], -1)).reshape(xshape)

def point2epsilon(X, type):
    from generate_struct import get_epsilon_map
    if type == "metasurface":
        w, l, xc, yc, dpillar = metasurface_vec2mat(X)
        eps = get_epsilon_map(raw=[w, l, xc, yc, dpillar], type="metasurface").reshape(1, 172,172)
    elif type == "angles":
        w, l, xc, yc, dpillar, angles = metasurface_vec2mat(X, angles=True)
        eps = get_epsilon_map(raw=[w, l, xc, yc, dpillar, angles], type="angles").reshape(1, 172,172)
    elif type == "annular":
        assert X.shape[0] == 9
        r = X[0:4]
        w = X[4:]
        eps = get_epsilon_map(raw=[r, w], type="annular").reshape(1, 172,172)
    return eps