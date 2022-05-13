import torch
from BasicNet import BasicNet
from MaskingNet import BasicMaskingNet
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = BasicNet(0.003, 2, 5000, 1, device)
masking_net = BasicMaskingNet(learning_rate_mask=3, learning_rate_nn=0.01, input_shape=2, hidden_layer_shape=5000,
                              output_shape=1, mask_percent=0.5, device=device)

inputs = torch.tensor([[0., 1.],
                       [1., 0.],
                       [0., 0.],
                       [1., 1.]], device=device, dtype=torch.float)
outputs = torch.tensor([[1.],
                        [1.],
                        [0.],
                        [0.]], device=device, dtype=torch.float)

loss_nn = []
loss_mask = []
for _ in range(100):
    loss_mask.append(masking_net.training_step_mask(inputs, outputs))
    loss_nn.append(net.training_step(inputs, outputs))

plt.plot(loss_mask, label="Mask, lr = 3")
plt.plot(loss_nn, label="Weights, lr = 0.003")
plt.yscale("log")
plt.legend(title="Training")
plt.xlabel("Epoch")
plt.ylabel("Log MSE loss")
plt.title("Mask vs weight training for large mask learning rates")
plt.savefig("xor_result.png")
plt.show()

import os
os.getcwd()

net.forward(inputs)
masking_net.forward(inputs, masking=False)
masking_net.forward(inputs, masking=True)


masking_net.training_step_nn(inputs, outputs)
masking_net.training_step_mask(inputs, outputs)


torch.sum(masking_net.fc1_mask_weight.grad)
torch.sum(masking_net.fc2_mask_weight.grad)
torch.sum(masking_net.fc1_mask_bias.grad)
torch.sum(masking_net.fc2_mask_bias.grad)

