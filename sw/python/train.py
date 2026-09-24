import torch
import numpy as np
import torch.nn as nn

d = np.load("fer3.npz")

x_train = torch.from_numpy((d["x_train"].astype(np.float32) - 128) / 128)
y_train = torch.from_numpy(d["y_train"])
x_val = torch.from_numpy((d["x_val"].astype(np.float32) - 128) / 128)
y_val = torch.from_numpy(d["y_val"])
x_test = torch.from_numpy((d["x_test"].astype(np.float32) - 128) / 128)
y_test = torch.from_numpy(d["y_test"])

model = nn.Sequential(
    #first hidden layer I think based on googling
    nn.Linear(2304, 64),
    nn.ReLU(),

    #second hidden layer I think based on googling
    nn.Linear(64, 3),
)
criterion = nn.CrossEntropyLoss() #everyone online uses that name
predictions = model(x_train)
targets = y_train
loss = criterion(predictions, targets)
loss.backward()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001) #does this know to apply the loss somehow? feels like there are a looot of layers of abstraction here




print(loss.item(), "lossss")



print(x_train.dtype, x_train.min(), x_train.max())
print(y_train.dtype, y_train.shape)
print(d["x_train"], "this is x train")

print(d["x_val"], "this is x val")
'''
d = np.load("fer3.npz")
print(d.files)                          # what arrays are in it
print(d["x_train"].shape, d["x_train"].dtype)
print(d["y_train"].shape, d["y_train"].dtype)
print(np.bincount(d["y_train"]))        # count per class
print(d["x_train"].min(), d["x_train"].max())
print(d["class_names"])

import matplotlib.pyplot as plt
img = d["x_train"][5].reshape(48, 48)
plt.imsave("check0.png", img, cmap="gray")
print(d["class_names"][d["y_train"][5]])


import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(size=2304)
W1 = rng.normal(size=(64, 2304))
b1 = rng.normal(size=64)
h = W1 @ x + b1
h = np.maximum(0, h)
W2 = rng.normal(size=(3, 64))
b2 = rng.normal(size=3)
logits = W2 @ h + b2
'''