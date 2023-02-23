import torch
from BasicNet import BasicNet
from MaskingNet import BasicMaskingNet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inputs, outputs = load_diabetes(return_X_y=True)

inputs = torch.tensor(inputs, device=device, dtype=torch.float)
outputs = torch.tensor(outputs, device=device, dtype=torch.float).reshape(442,1)

net = BasicNet(0.01, 10, 500, 1, device)
masking_net = BasicMaskingNet(learning_rate_mask=0.99, learning_rate_nn=0.01, input_shape=10, hidden_layer_shape=500,
                              output_shape=1, mask_percent=0.1, device=device)

loss_nn = []
loss_mask = []
loss_mask_nn = []

masking_net.refresh_unused_weights()

range_int = 25
for _ in range(range_int):
    loss_mask.append(masking_net.training_step_mask(inputs, outputs))

for _ in range(range_int):
    loss_mask.append(masking_net.training_step_nn(inputs, outputs, masking_only=True))

for _ in range(range_int):
    loss_nn.append(net.training_step(inputs, outputs))
print("done")


plt.plot(loss_mask, label="Mask")
plt.plot(loss_mask_nn, label="Mask_weights")
plt.plot(loss_nn, label="Weights")
plt.yscale("log")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Log MSE loss")
plt.title("Mask vs weight training for large mask learning rates")
plt.savefig("xor_result.png")
plt.show()

masking_net.refresh_unused_weights()
masking_net.mask_percent += 0.1
print(masking_net.mask_percent)

loss_mask = []
for i in range(5):
    for j in range(20):
        loss_mask.append(masking_net.training_step_mask(inputs, outputs))
        # for j in range(5):
        #     loss_mask.append(masking_net.training_step_nn(inputs, outputs, masking_only=True))
        masking_net.refresh_unused_weights()