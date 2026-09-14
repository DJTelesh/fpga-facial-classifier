import numpy as np
#importing numpy


rng = np.random.default_rng(0)
#making the random list (confirm 1 yes/no)


x = rng.normal(size=2304)
#making list x of size 2304 rows, I assume all the pixels in the image (confirm 2 yes/no)

W1 = rng.normal(size=(64, 2304))
#making list of 64 neurons each with 2304 weights (confirm 3 yes/no)

b1 = rng.normal(size=64)
#shifting the relu function as much as I need/want (not fully sure how I will use this, my guess is a positive b would allow more values through) (confirm 4 yes/no)


h = W1 @ x + b1
#dot product of the image pixels with the weights (confirm 5 yes/no)

h = np.maximum(0, h)
#this is the relu function, if h is negative or 0 then 0 is selected otherwise h is selected, this will be performed 64 times for all the neurons, and be assigned to h (confirm 6 yes/no)

W2 = rng.normal(size=(3, 64))
#3 I assume is the facial expressions with 64 relu neurons (confirm 7 yes/no)

b2 = rng.normal(size=3)
#3 It's the relu shift again (confirm 8 yes/no)

logits = W2 @ h + b2
#I guess this performs the 2nd layer on the output of the first which summarized the pixels into one value (confirm 9 yes/no)
print(h.min(), h.max())
print(np.isnan(h).sum(), np.isinf(h).sum())
print(h.shape)
print(logits.shape)