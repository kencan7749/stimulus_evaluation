#!/usr/bin/python
#-*- coding: utf-8 -*- 
"""
StimEval class
    python package for evaluating 
    the reconstructed/decoded image[video, other domin] with targets.
"""
import numpy as np
from .metric import *
metric_dict = {
    'profile correlation' : pixcorr,
    'spatial correlation' : pixcorr,
    'squared error': squarederror,
    'pairwise identification': pairwise_identification,
}

opts_dict= {'spatial correlation': {'var': 'row'}}

class StimEvaluater():
    """ Abstract 'stim evaluator class common for all types of stim evaluators/

    """
    def __init__(self, metric, **opts):
        """ Initialize evaluator class

        Args:
            metric (string): specify the metric 
            opt_dict (dict): optional dict for calculate metrics (this will be not well written)
        """
        self.metric = metric_dict[metric]
        self.opts = opts
        # select option if opts (dict) is empty
        if len(self.opts)==0 and metric in opts_dict:
            self.opts = opts_dict[metric]
        
        
    def calc_metric(self, true_stim, recon_stim):
        """[summary]

        Args:
            true_stim ([type]): [description]
            recon_stim ([type]): [description]

        Returns:
            np.array (batch_size, *shape): the first element is the batch_size of the inputs
                                        For the second each elements is the metric calculated. 
        """
        if type(true_stim) == list:
            true_stim = np.array(true_stim)
            recon_stim = np.array(recon_stim)

        # check for mismatch between true and recon
        try:
            true_stim.shape == recon_stim.shape
        except:
            raise('The shape is not matched between inputs')


        #return the caluculated values for every sample
        calculated_list = self.metric(true_stim, recon_stim, **self.opts)
        #print(np.mean(calculated_list))
        return calculated_list
   
    def __call__(self, true_stim, recon_stim):
        calculated_list = self.calc_metric(true_stim, recon_stim)
        return calculated_list


class ImageEvaluator(StimEvaluater):
    """ calculate specified metric between two input images

    Args:
        StimEvaluater ([object]): Abstract class 

    Returns:
        
        np.array (batch_size, height, width, channel): the first element is the batch_size of the inputs
                                        For the second each elements is the metric calculated. 
    """

    def __init__(self, img_metric='pixel correlation'):
        super().__init__(img_metric)

    def __call__(self, true_img, recon_img):
        """[summary]

        Args:
            true_stim ([type]): [description]
            recon_stim ([type]): [description]

        Returns:
            np.array (batch_size, *shape): the first element is the batch_size of the inputs
                                        For the second each elements is the metric calculated. 
        """

        #check image shape (bs, h, w, ch)
        
        if len(true_img[0].shape) != 3:
            raise('The shape is not matched for image')
        return super().calc_metric(true_img, recon_img)


class FeatEvaluator(StimEvaluater):
    """ calculate specified metric between two input DNN features
 
    Args:
        StimEvaluater ([object]): Abstract class 

    Returns:
        
        np.array (batch_size, *shape): the first element is the batch_size of the inputs
                                        For the second each elements is the metric calculated. 
    """
    def __init__(self, img_metric='pixel correlation'):
        super().__init__(img_metric)

    def __call__(self, true_feat, decoded_feat):
        return super().__call__(true_feat, decoded_feat)

class VideoEvaluator(StimEvaluater):
    """ calculate specified metric between two input videos
 
    Args:
        StimEvaluater ([object]): Abstract class 

    Returns:
        
        np.array (batch_size, *shape): the first element is the batch_size of the inputs
                                        For the second each elements is the metric calculated. 
    """

    def __init__(self, vid_metric='pixel correlation'):
        super().__init__(vid_metric)

    def __call__(self, true_vid, recon_vid):
        """[summary]

        Args:
            true_vid ([type]): [description]
            recon_vid ([type]): [description]

        Returns:
            np.array (batch_size, *shape): the first element is the batch_size of the inputs
                                        For the second each elements is the metric calculated. 
        """

        #check image shape (bs, fr, h, w, ch)
        
        if len(true_vid[0].shape) != 4:
            raise('The shape is not matched for video')
        return super().__call__(true_vid, recon_vid)

