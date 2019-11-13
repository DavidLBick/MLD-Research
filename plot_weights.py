import torch
import matplotlib.pyplot as plt
import numpy as np

model = torch.load("saved_models/time_space_conv_725_epoch7.pt")

#import pdb
#pdb.set_trace()
layers = [child for child in model.children()]
embedding_model = layers[0]
conv_layer = embedding_model[0]
weights = conv_layer.weight
weights = weights.squeeze()

plt.figure(1)
for i in range(16):
	plt.subplot(4, 4, i+1)
	plt.imshow(weights[i,:,:].detach().numpy())
plt.show()