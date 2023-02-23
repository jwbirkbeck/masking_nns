import torch
from BasicNet import BasicNet
from MaskingNet import BasicMaskingNet
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = BasicNet(0.01, 2, 5, 1, device)
masking_net = BasicMaskingNet(learning_rate_mask=1, learning_rate_nn=0.01, input_shape=2, hidden_layer_shape=5,
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

# masking_net.refresh_unused_weights()


for _ in range(100):
    loss_mask.append(masking_net.training_step_mask(inputs, outputs))
    loss_nn.append(net.training_step(inputs, outputs))

plt.plot(loss_mask, label="Mask")
plt.plot(loss_nn, label="Weights")
plt.yscale("log")
plt.legend(title="Training")
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.title("Mask vs weight training for large mask learning rates")
plt.savefig("xor_result.png")
plt.show()

masking_net.forward(inputs, masking=True)
net.forward(inputs)


