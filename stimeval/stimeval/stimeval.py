"""
StimEval class
"""
import numpy as np
from .metric import *
metric_dict = {
    'pixel correlation' : pixcorr,
    'root mean square': rms,
    'ME pix corr': pixcorrME,
}

class StimEval():
    """ StimEval class
    Parameters
    ----------
    metric: str (defalt 'pixel correlation')
        string to select metric.

    """

    def __init__(self, metric='pixel correlation'):
        try:
            self.metric = metric_dict[metric]
        except KeyError as e:
            print(metric + ' is not implemented yet' + e)


    def __call__(self,true_stim, recon_stim, **ops):
        """
        Quantitative evaluation between true_stim and recon stim
        Input:
            true_stim: numpy array (expected np.float32 or np.float), shape is either 
                        [batch_size, fr, height, width, channel] (video) or 
                        [batch_size, height, width, channel] (image) 
            recon_stim: the same of true stim
        """ 

        # check for mismatch between true and recon
        try:
            true_stim.shape == recon_stim.shape
        except:
            print('The shape is not matched between inputs')
        return self.metric(true_stim, recon_stim)


