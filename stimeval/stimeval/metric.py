import numpy as np
from numpy.matlib import repmat

def pixcorr(x, y, var='col'):
    """[summary]

    Args:
        x (np.array): soruce (true) values [batch_size, *shape]
        y (np.array): target (decoded/recon) values [batch_size, *shape]
        var (str, optional): Specifying whther rows or colmuns represent variables.
                             Defaults to 'row'.

    Return:
        r
         Correlation coefficient
    """
    
    batch_size = x.shape[0]
    x_flat = x.reshape(batch_size, -1)
    y_flat = y.reshape(batch_size, -1)
    # Normalize x and y to row-var format
    if var =='row':
        if x_flat.shape[1] ==1:
            x_flat = x_flat.T
        if y_flat.shape[1] == 1:
            y_flat = y_flat.T
    elif var == 'col':
        x_flat = x_flat.T
        y_flat = y_flat.T
    else:
        raise ValueError('Unknonn var parameter specified')

    # Match size of x and y
    if x_flat.shape[0] ==1 and y_flat.shape[0] != 1:
        x_flat = repmat(x_flat, y_flat.shape[0], 1)
    elif x_flat.shape[0] !=1 and y_flat.shape[0] == 1:
        y_flat = repmat(y_flat, x_flat.shape[0], 1)

    # Check size of normalized x and y
    if x_flat.shape != y_flat.shape:
        raise TypeError('Input matrixes size mismatch')

    #Get num variables
    nvar = x_flat.shape[0]

    #Get Correlation
    rmat = np.corrcoef(x_flat, y_flat, rowvar=1)
    r = np.diag(rmat[:nvar, nvar:])
    return r

    

def squarederror(x,y):
    """Calculate squared error

    Args:
        x (np.array): source (true) values 
        y (np.array): target (decoded/recon) values


    Returns:
        np.array: (x - y) ** 2 
    """
    diff_squared = (x - y)**2
    return np.sqrt(diff_squared)
    

def pixcorrME(x, y , ME_param=None):
    raise NotImplementedError

def rmsME(x,y, ME_param=None):
    raise NotImplementedError



if __name__ == '__main__':
    rand_img = np.random.rand(1, 224,224, 3)

    print(rms(rand_img, rand_img))