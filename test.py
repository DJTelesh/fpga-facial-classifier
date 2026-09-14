import numpy as np
d = np.load("fer3.npz")
print(d.files)                          # what arrays are in it
print(d["x_train"].shape, d["x_train"].dtype)
print(d["y_train"].shape)
print(np.bincount(d["y_train"]))        # count per class
print(d["x_train"].min(), d["x_train"].max())
print(d["class_names"])

import matplotlib.pyplot as plt
img = d["x_train"][5].reshape(48, 48)
plt.imsave("check0.png", img, cmap="gray")
print(d["class_names"][d["y_train"][5]])