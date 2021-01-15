import numpy as np

def pixcorr(x, y, var='row'):
    shape = x.shape
    x_flat = x.reshape(shape[0], -1)
    y_flat = y.reshape(shape[0], -1)
    if var == 'col':
        x_flat = x_flat.T
        y_flat = y_flat.T

    nvar = x_flat.shape[0]
    rmat = np.corrcoef(x_flat, y_flat, rowvar=1)

    r = np.diag(rmat[:nvar, nvar:])
    return r

def rms(x,y):
    raise NotImplementedError

def pixcorrME(x, y , ME_param=None):
    raise NotImplementedError

def rmsME(x,y, ME_param=None):
    raise Not