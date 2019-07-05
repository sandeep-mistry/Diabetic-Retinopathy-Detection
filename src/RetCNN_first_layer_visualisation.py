"""

This file shows a visualisation of the first convolutional layer of RetCNN.

"""

import torch
import matplotlib.pyplot as plt
from RetCNN import RetCNN

load_weights_dir = "CNNDRweightsbalanced5epochs"

# build the model and load state
model = RetCNN()
model.load_state_dict(torch.load(load_weights_dir))

weights = model.conv1.weight.data.cpu()

# plot the first layer features
for i in range(0,30):
	plt.subplot(5,6,i+1)
	plt.imshow(weights[i, 0, :, :], cmap=plt.get_cmap())
plt.show()
