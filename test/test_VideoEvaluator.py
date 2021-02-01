import unittest
import os 
import sys
import numpy as np
from PIL import Image

sys.path.append('./stimeval')
from stimeval import VideoEvaluator
from utils import load_video

#setting path
video_path = './test_data/vid'
true_vid_path = os.path.join(video_path, 'true')
recon_vid_path = os.path.join(video_path, 'recon')

true_vid_file_names = os.listdir(true_vid_path)
recon_vid_file_names = os.listdir(recon_vid_path)

#load numpy img
true_vid_list = [load_video(os.path.join(true_vid_path,true_vid), 'float', 224,224)[:11]
                    for true_vid in true_vid_file_names]
recon_vid_list = [load_video(os.path.join(recon_vid_path,true_vid), 'float', 224,224)
                    for true_vid in recon_vid_file_names]

true_vid_list_min = [load_video(os.path.join(true_vid_path,true_vid), 'float', 16,16)[:11]
                    for true_vid in true_vid_file_names]
recon_vid_list_min = [load_video(os.path.join(recon_vid_path,true_vid), 'float', 16,16)
                    for true_vid in recon_vid_file_names]

class TestVideoEvaluator(unittest.TestCase):

    def test_se_rand(self):
        """Test squared error evalution in video evaluator
        """
        #test rms return 0 for the same input
        rand_vid = np.random.rand(1, 16,224,224,3)
        eval = VideoEvaluator(vid_metric='squared error')
        self.assertAlmostEqual(np.sum(eval(rand_vid, rand_vid)), 0)


    def test_se_actual_video(self):
        """Test squared error evaluation for true and reconstructed video
        """
    
        # Set evaluatator
        eval = VideoEvaluator(vid_metric='squared error')
        # evaluate true image and recon image
        val = eval(recon_vid_list,true_vid_list)

        self.assertEqual(len(val), len(true_vid_file_names))
        

    def test_corr_actual_video(self):
        """Test profile correlation evaluation for true and reconstructed image
        """
        eval = VideoEvaluator(vid_metric='profile correlation')
        # evaluate true image and recon image
        val = eval(recon_vid_list,true_vid_list)
        
        self.assertEqual(len(val), 11 * 224 * 224 * 3)




if __name__ == '__main__':
    unittest.main()