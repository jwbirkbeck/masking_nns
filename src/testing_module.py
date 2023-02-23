import torch
from MaskingNet_module import MaskingNet
import matplotlib.pyplot as plt

device = "cpu"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputs = torch.tensor([[0., 1.],
                       [1., 0.],
                       [0., 0.],
                       [1., 1.]], device=device, dtype=torch.float)
outputs = torch.tensor([[1.],
                        [1.],
                        [0.],
                        [0.]], device=device, dtype=torch.float)

masking_net = MaskingNet(learning_rate_mask=0.01, input_size=2, hidden_layer_size=50,
                         output_size=1, mask_percent=0.1, device=device)

loss_mask = []

for _ in range(500):
    loss_mask.append(masking_net.training_step_mask(inputs, outputs))

plt.plot(loss_mask, label="Mask")
plt.yscale("log")
plt.legend(title="Training")
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.title("Mask vs weight training for large mask learning rates")
plt.show()

with torch.no_grad():
    print(masking_net.forward(inputs))

masking_net.refresh_all_layers()
masking_net.mask_percent += 0.1
print(masking_net.mask_percent)
