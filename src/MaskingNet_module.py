import torch
from torch import nn
from torch import optim
from copy import deepcopy

class MaskingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, mask_weights, masking_percent):
        orig_shape = weight.shape
        _, flat_mask_bottomk = torch.topk(mask_weights.reshape(-1),
                                          k=int((1 - masking_percent) * mask_weights.view(-1).shape[0]),
                                          largest=False)
        masked_weights = weight.clone()
        masked_weights = masked_weights.reshape(-1)
        masked_weights[flat_mask_bottomk] = 0
        masked_weights = masked_weights.reshape(orig_shape)
        return masked_weights

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output, None


class Masking(nn.Module):
    def __init__(self):
            super(Masking, self).__init__()

    def forward(self, weight, mask_weights, masking_percent):
        x = MaskingFunction.apply(weight=weight, mask_weights=mask_weights, masking_percent=masking_percent)
        return x


class Mask(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        weight = torch.rand(size=(self.output_size, self.input_size), requires_grad=True)
        self.weight = nn.Parameter(weight)
        bias = torch.rand(size=[self.output_size], requires_grad=True)
        self.bias = bias

    def forward(self, inputs, nn_layer, masking_percent):
        nn_layer_masked = deepcopy(nn_layer)
        masked_weights = MaskingFunction.apply(nn_layer_masked.weight, self.weight, masking_percent)
        masked_bias = MaskingFunction.apply(nn_layer_masked.bias, self.bias, masking_percent)
        output = torch.matmul(masked_weights, inputs.T).T + masked_bias
        return output

    def get_masks(self, masking_percent):
        output_mask_weight = torch.ones(self.weight.shape, dtype=torch.bool)
        output_mask_bias = torch.ones(self.bias.shape, dtype=torch.bool)
        _, flat_mask_bottomk_weight = torch.topk(self.weight.reshape(-1),
                                                 k=int((1 - masking_percent) * self.weight.view(-1).shape[0]),
                                                 largest=False)
        _, flat_mask_bottomk_bias = torch.topk(self.bias.reshape(-1),
                                               k=int((1 - masking_percent) * self.bias.view(-1).shape[0]),
                                               largest=False)
        output_mask_weight.view(-1)[flat_mask_bottomk_weight] = False
        output_mask_bias.view(-1)[flat_mask_bottomk_bias] = False
        return output_mask_weight, output_mask_bias


class MaskingNet(nn.Module):
    def __init__(self, learning_rate_mask, input_size, hidden_layer_size,
                 output_size, mask_percent, device):
        super().__init__()
        self.device = device
        self.mask_percent = torch.tensor(mask_percent, device=device, dtype=torch.float)
        self.learning_rate_mask = learning_rate_mask
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size).to(device)
        self.fc1_mask = Mask(input_size, hidden_layer_size).to(device)
        self.fc1.requires_grad_(False)
        self.fc1_mask.requires_grad_(True)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size).to(device)
        self.fc2_mask = Mask(hidden_layer_size, hidden_layer_size).to(device)
        self.fc2.requires_grad_(False)
        self.fc2_mask.requires_grad_(True)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size).to(device)
        self.fc3_mask = Mask(hidden_layer_size, hidden_layer_size).to(device)
        self.fc3.requires_grad_(False)
        self.fc3_mask.requires_grad_(True)
        self.fc4 = nn.Linear(hidden_layer_size, output_size).to(device)
        self.fc4_mask = Mask(hidden_layer_size, output_size).to(device)
        self.fc4.requires_grad_(False)
        self.fc4_mask.requires_grad_(True)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate_mask)
        self.to(self.device)

    def forward(self, input, masking=True):
        activation = nn.ReLU()
        if masking:
            output = self.fc1_mask(input, self.fc1, self.mask_percent)
            output = activation(output)
            output = self.fc2_mask(output, self.fc2, self.mask_percent)
            output = activation(output)
            output = self.fc3_mask(output, self.fc3, self.mask_percent)
            output = activation(output)
            output = self.fc4_mask(output, self.fc4, self.mask_percent)
        else:
            output = self.fc1(input)
            output = activation(output)
            output = self.fc2(output)
        return output

    def training_step_mask(self, inputs, outputs):
        self.optimizer.zero_grad(set_to_none=True)
        preds = self.forward(inputs, masking=True)
        loss = self.loss_func(preds, outputs)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def refresh_weights_in_layer(self, layer, mask_layer):
        with torch.no_grad():
            weight_old = layer.weight.clone()
            bias_old = layer.bias.clone()
            shape_old = layer.weight.shape
            # True = useful values, False = not useful values
            weight_mask, bias_mask = mask_layer.get_masks(self.mask_percent)
            layer = nn.Linear(shape_old[1], shape_old[0]).to(self.device)
            layer.weight[weight_mask] = weight_old[weight_mask]
            layer.bias[bias_mask] = bias_old[bias_mask]

    def refresh_all_layers(self):
        self.refresh_weights_in_layer(self.fc1, self.fc1_mask)
        self.refresh_weights_in_layer(self.fc2, self.fc2_mask)
        self.refresh_weights_in_layer(self.fc3, self.fc3_mask)
        self.refresh_weights_in_layer(self.fc4, self.fc4_mask)
        self.fc1_mask = Mask(self.input_size, self.hidden_layer_size).to(self.device)
        self.fc2_mask = Mask(self.hidden_layer_size, self.hidden_layer_size).to(self.device)
        self.fc3_mask = Mask(self.hidden_layer_size, self.hidden_layer_size).to(self.device)
        self.fc4_mask = Mask(self.hidden_layer_size, self.output_size).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate_mask)


