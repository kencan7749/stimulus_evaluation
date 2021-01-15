import numpy as np

def pixcorr(x, y, var='row'):
    
    raise NotImplementedError

def rms(x,y):
    diff_squared = (x - y)**2
    return np.mean(np.sqrt(diff_squared))
    

    raise NotImplementedError

def pixcorrME(x, y , ME_param=None):
    raise NotImplementedError

def rmsME(x,y, ME_param=None):
    raise NotImplementedError



if __name__ == '__main__':
    rand_img = np.random.rand(1, 224,224, 3)

    print(rms(rand_img, rand_img))