import torch
import torch.nn.functional as funcs
from torch import nn
from torch import optim


@torch.no_grad()
def target_polyak_update(source_model, target_model, initial_update=False):
    polyak = 0 if initial_update else 0.995
    for target_param, param in zip(target_model.fc1.parameters(), source_model.fc1.parameters()):
        target_param.data.copy_(target_param.data * polyak + param.data * (1 - polyak))
    for target_param, param in zip(target_model.fc2.parameters(), source_model.fc2.parameters()):
        target_param.data.copy_(target_param.data * polyak + param.data * (1 - polyak))
    for target_param, param in zip(target_model.fc3.parameters(), source_model.fc3.parameters()):
        target_param.data.copy_(target_param.data * polyak + param.data * (1 - polyak))


class TargetStackingNetwork(nn.Module):
    def __init__(self, layer_sizes, alpha, device):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.alpha = alpha
        self.device = device
        self.fc1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.fc2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.fc3 = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        self.loss_func = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

    def forward(self, x):
        out = self.fc1(x)
        out = funcs.relu(out)
        out = self.fc2(out)
        out = funcs.relu(out)
        out = self.fc3(out)
        return out

    def training_step(self, inputs, outputs):
        self.optimizer.zero_grad(set_to_none=True)
        preds = self.forward(inputs)
        loss = self.loss_func(preds, outputs)
        loss.backward()
        return loss.item()
