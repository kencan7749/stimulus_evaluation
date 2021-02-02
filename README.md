# stim_eval: python package for evaluating the reconstructed/decoded image[video, other domin] with targets.
[Example (the goal)]
```
from stimeval import FeatEvaluator, ImageEvaluator, VideoEvaluator


true_img_arr = ... # np.array: shape (sample, height, width, channel (3))
recin_img_arr = ... # np.array: shape (same as true_img_arr)

evaluator = FeatEvaluator(metric = 'profile correlation') # like 'pairwise identification', 'squared error'...  
# (see metric.py in detail)
score = eval(recon_img_arr, true_img_arr) 
score.shape 
(-> hetight * width * channel if 'profile correltion')
(-> sample if 'pairwise identification')


true_feat_arr = ... # np.array: shape (sample, *feat_shape)
decoded_feat_arr = ... # np.array: shape (same as true_stim_arr)

evaluator = FeatEvaluator(metric = 'profile correlation') # like 'pairwise identification', 'squared error'...  
# (see metric.py in detail)
score = eval(decoded_feat_arr, true_feat_arr) 
score.shape 
(-> feat_shape if 'profile correltion')
(-> sample if 'pairwise identification')

```
# Requirements 
Confirm in python3
- numpy
- tqdm

# Plan

- Evaluate metric in feature space from image inputs

