#!/usr/bin/python
#-*- coding: utf-8 -*- 

import unittest
import os 
import sys
import numpy as np
from PIL import Image

sys.path.append('./stimeval')
from stimeval import ImageEvaluator

def _load_img_with_resize(path, size=(224,224)):
    #load image 
    img = Image.open(path)
    #resize
    img = img.resize(size)
    #return as np.array
    return np.asarray(img).astype(np.float32)


#setting path
image_path = './test_data/img'
true_img_path = os.path.join(image_path, 'true')
recon_img_path = os.path.join(image_path, 'recon')

true_img_file_names = os.listdir(true_img_path)
recon_img_file_names = os.listdir(recon_img_path)

#load numpy img
true_img_list = [_load_img_with_resize(os.path.join(true_img_path,true_img), (224,224))
                    for true_img in true_img_file_names]
recon_img_list = [_load_img_with_resize(os.path.join(recon_img_path,true_img), (224,224))
                    for true_img in recon_img_file_names]

true_img_list_min = [_load_img_with_resize(os.path.join(true_img_path,true_img), (32,32))
                    for true_img in true_img_file_names]
recon_img_list_min = [_load_img_with_resize(os.path.join(recon_img_path,true_img), (32,32))
                    for true_img in recon_img_file_names]

class TestImageEvaluator(unittest.TestCase):

    def test_se_rand(self):
        """Test squared error evalution in image evaluator
        """
        #test rms return 0 for the same input
        rand_img = np.random.rand(1,224,224,3)
        eval = ImageEvaluator(img_metric='squared error')
        self.assertAlmostEqual(np.sum(eval(rand_img, rand_img)), 0)


    def test_se_actual_image(self):
        """Test squared error evaluation for true and reconstructed image
        """
    
        # Set evaluatator
        eval = ImageEvaluator(img_metric='squared error')
        # evaluate true image and recon image
        val = eval(recon_img_list,true_img_list)

        self.assertEqual(len(val), len(true_img_file_names))
        

    def test_profile_corr_actual_image(self):
        """Test profile correlation evaluation for true and reconstructed image
        """
        eval = ImageEvaluator(img_metric='profile correlation')
        # evaluate true image and recon image
        #val = eval(true_img_list_min, recon_img_list_min)
        val = eval(recon_img_list,true_img_list ,)
        self.assertEqual(val.shape, (224,224,3))

    def test_spatial_corr_actual_image(self):
        """Test profile correlation evaluation for true and reconstructed image
        """
        eval = ImageEvaluator(img_metric='pattern correlation')
        # evaluate true image and recon image
        #val = eval(true_img_list_min, recon_img_list_min)
        val = eval(recon_img_list,true_img_list )
        self.assertEqual(len(val), len(true_img_list))

    def test_pariwise_identification(self):
        """Test identification analysis for true and reconstructed image
        """
        eval = ImageEvaluator(img_metric='pairwise identification')
        # evaluate true image and recon image
        #val = eval(true_img_list_min, recon_img_list_min)
        val = eval(recon_img_list,true_img_list )
        self.assertEqual(len(val), len(true_img_list))




if __name__ == '__main__':
    unittest.main()