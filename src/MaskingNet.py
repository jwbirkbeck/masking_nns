import torch
from torch import nn
from torch import optim


class MaskingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, masking):
        return input * masking

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Masking(nn.Module):
    def __init__(self):
            super(Masking, self).__init__()

    def forward(self, x, masking):
        x = MaskingFunction.apply(x, masking)
        return x


class BasicMaskingNet(nn.Module):
    def __init__(self, learning_rate_mask, learning_rate_nn, input_shape, hidden_layer_shape, output_shape, mask_percent, device):
        super(BasicMaskingNet, self).__init__()
        self.device = device
        self.mask_percent = mask_percent
        self.learning_rate_mask = learning_rate_mask
        self.learning_rate_nn = learning_rate_nn
        self.input_shape = input_shape
        self.hidden_layer_shape = hidden_layer_shape
        self.output_shape = output_shape
        self.fc1 = nn.Linear(input_shape, hidden_layer_shape)
        self.fc2 = nn.Linear(hidden_layer_shape, output_shape)
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False
        self.fc1_mask_weight = torch.rand(size=self.fc1.weight.shape, device=self.device, requires_grad=True)
        self.fc2_mask_weight = torch.rand(size=self.fc2.weight.shape, device=self.device, requires_grad=True)
        self.fc1_mask_bias = torch.rand(size=self.fc1.bias.shape, device=self.device, requires_grad=True)
        self.fc2_mask_bias = torch.rand(size=self.fc2.bias.shape, device=self.device, requires_grad=True)

        self.fc1.weight = torch.nn.init.kaiming_uniform_(self.fc1.weight)
        self.fc2.weight = torch.nn.init.kaiming_uniform_(self.fc2.weight)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate_nn)
        self.to(self.device)

    def get_masks(self):
        _, fc1_mask_weight_topk = torch.topk(self.fc1_mask_weight, k=int(self.mask_percent * self.fc1_mask_weight.shape[0]), dim=0)
        _, fc2_mask_weight_topk = torch.topk(self.fc2_mask_weight, k=int(self.mask_percent * self.fc2_mask_weight.shape[0] * self.fc2_mask_weight.shape[1]), dim = 1)
        _, fc1_mask_bias_topk = torch.topk(self.fc1_mask_bias, k=int(self.mask_percent * self.fc1_mask_bias.shape[0]), dim=0)
        _, fc2_mask_bias_topk = torch.topk(self.fc2_mask_bias, k=int(self.mask_percent * self.fc2_mask_bias.shape[0]), dim=0)

        fc1_mask_weight_index = torch.zeros(self.fc1_mask_weight.shape, device=self.device, dtype=torch.bool, requires_grad=False)
        fc1_mask_weight_index[fc1_mask_weight_topk[:, 0], 0] = True
        fc1_mask_weight_index[fc1_mask_weight_topk[:, 1], 1] = True

        fc2_mask_weight_index = torch.zeros(self.fc2_mask_weight.shape[1], device=self.device, dtype=torch.bool, requires_grad=False)
        fc2_mask_weight_index[fc2_mask_weight_topk] = True

        fc1_mask_bias_index = torch.zeros(self.fc1_mask_bias.shape, device=self.device, dtype=torch.bool, requires_grad=False)
        fc1_mask_bias_index[fc1_mask_bias_topk] = True

        fc2_mask_bias_index = torch.zeros(self.fc2_mask_bias.shape, device=self.device, dtype=torch.bool, requires_grad=False)
        fc2_mask_bias_index[fc2_mask_bias_topk] = True

        return fc1_mask_weight_index, fc2_mask_weight_index, fc1_mask_bias_index, fc2_mask_bias_index

    def forward(self, input, masking):
        activation = nn.ReLU()
        masker = Masking()
        if masking:
            fc1_mask_weight_index, fc2_mask_weight_index, fc1_mask_bias_index, fc2_mask_bias_index = self.get_masks()
            fc1_mask_weight = masker(self.fc1_mask_weight, fc1_mask_weight_index)
            fc2_mask_weight = masker(self.fc2_mask_weight, fc2_mask_weight_index)
            fc1_mask_bias = masker(self.fc1_mask_bias, fc1_mask_bias_index)
            fc2_mask_bias = masker(self.fc2_mask_bias, fc2_mask_bias_index)

            output = torch.matmul((fc1_mask_weight * self.fc1.weight), input.T) +\
                     (fc1_mask_bias * self.fc1.bias).reshape(self.hidden_layer_shape, 1)

            output = activation(output)
            output = torch.matmul((fc2_mask_weight * self.fc2.weight), output) +\
                     (fc2_mask_bias * self.fc2.bias).reshape(self.output_shape, 1)
        else:
            output = self.fc1(input)
            output = activation(output)
            output = self.fc2(output)
        return output.reshape(4, 1)

    def training_step_mask(self, inputs, outputs):
        self.fc1_mask_weight.grad = None
        self.fc2_mask_weight.grad = None
        self.fc1_mask_bias.grad = None
        self.fc2_mask_bias.grad = None
        preds = self.forward(inputs, masking=True)
        loss = self.loss_func(preds, outputs)
        loss.backward()
        self.fc1_mask_weight.data -= self.learning_rate_mask * self.fc1_mask_weight.grad
        self.fc2_mask_weight.data -= self.learning_rate_mask * self.fc2_mask_weight.grad
        self.fc1_mask_bias.data -= self.learning_rate_mask * self.fc1_mask_bias.grad
        self.fc2_mask_bias.data -= self.learning_rate_mask * self.fc2_mask_bias.grad
        return loss.item()

    def training_step_nn(self, inputs, outputs):
        self.fc1.weight.requires_grad = True
        self.fc1.bias.requires_grad = True
        self.fc2.weight.requires_grad = True
        self.fc2.bias.requires_grad = True
        self.optimizer.zero_grad(set_to_none=True)
        preds = self.forward(inputs, masking=False)
        loss = self.loss_func(preds, outputs)
        loss.backward()
        self.optimizer.step()
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False
        return loss.item()
