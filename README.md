# stim_eval: python package for evaluating the reconstructed/decoded image[video, other domin] with targets.
[Example (the goal)]

eval = StimEval(name  = 'pixel correlation')

score = eval(true_stim, recon_stim) #both shape should be (batch, stim_shape).

