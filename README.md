# video_evaluation
[Example (the goal)]

eval = VideoEval(name  = 'pixel correlation')

score = eval(true_vid, recon_vid) #both shape should be (batch, fr, height, width, channel).

