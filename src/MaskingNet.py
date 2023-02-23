import torch
from torch import nn
from torch import optim


class MaskingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, mask_weights, masking_percent):
        # _, fc1_mask_weight_topk = torch.topk(self.fc1_mask_weight, k=int(self.mask_percent * self.fc1_mask_weight.shape[0]), dim=0)
        orig_shape = mask_weights.shape
        _, flat_mask_bottomk = torch.topk(mask_weights.view(-1), k=int((1 - masking_percent) * mask_weights.view(-1).shape[0]), largest=False)
        masked_weights = weights.clone()
        masked_weights.view(-1)[flat_mask_bottomk] = 0
        return masked_weights

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output, None


class Masking(nn.Module):
    def __init__(self):
            super(Masking, self).__init__()

    def forward(self, weights, mask_weights, masking_percent):
        x = MaskingFunction.apply(weights, mask_weights, masking_percent)
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
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
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
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate_nn)
        self.to(self.device)

    def get_mask(self, mask_weights):
        with torch.no_grad():
            _, flat_mask_bottomk = torch.topk(mask_weights.view(-1),
                                              k=int((1 - self.mask_percent) * mask_weights.view(-1).shape[0]), largest=False)
            mask = torch.ones( size=mask_weights.shape, dtype=torch.bool)
            mask.view(-1)[flat_mask_bottomk] = False
        return mask

    def forward(self, input, masking):
        activation = nn.ReLU()
        masker = Masking()
        if masking:
            fc1_weights = masker(self.fc1.weight, self.fc1_mask_weight, self.mask_percent)
            fc1_bias = masker(self.fc1.bias, self.fc1_mask_bias, self.mask_percent)
            fc2_weights = masker(self.fc2.weight, self.fc2_mask_weight, self.mask_percent)
            fc2_bias = masker(self.fc2.bias, self.fc2_mask_bias, self.mask_percent)

            output = torch.matmul((fc1_weights), input.T) +\
                     (fc1_bias).reshape(self.hidden_layer_shape, 1)

            output = activation(output)
            output = torch.matmul((fc2_weights), output) +\
                     (fc2_bias).reshape(self.output_shape, 1)
        else:
            output = self.fc1(input)
            output = activation(output)
            output = self.fc2(output)
        output = output.reshape(input.shape[0], 1)
        # output = torch.sigmoid(output)
        # return output.reshape(4, 1)
        return output

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

    def refresh_unused_weights(self):
        # Firstly, generate new weights in the shape of the ones:
        fc1_mask_weight_bool = self.get_mask(self.fc1_mask_weight)
        fc2_mask_weight_bool = self.get_mask(self.fc2_mask_weight)
        fc1_mask_bias_bool = self.get_mask(self.fc1_mask_bias)
        fc2_mask_bias_bool = self.get_mask(self.fc2_mask_bias)

        # Save the current weights temporarily:
        fc1_weights = self.fc1.weight.clone()
        fc2_weights = self.fc2.weight.clone()
        fc1_bias = self.fc1.bias.clone()
        fc2_bias = self.fc2.bias.clone()

        # Refresh all the weights by reinitialising the layer:
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

        # Insert the weights we want to keep, so we've only refreshed the unused weights:
        self.fc1.weight[fc1_mask_weight_bool] =\
            fc1_weights[fc1_mask_weight_bool]
        self.fc2.weight[fc2_mask_weight_bool.reshape(1, self.hidden_layer_shape)] =\
            fc2_weights[fc2_mask_weight_bool.reshape(1, self.hidden_layer_shape)]
        self.fc1.bias[fc1_mask_bias_bool] =\
            fc1_bias[fc1_mask_bias_bool]
        self.fc2.bias[fc2_mask_bias_bool] =\
            fc2_bias[fc2_mask_bias_bool]

        # Finally, randomize the mask scores to restart learning:
        self.fc1_mask_weight = torch.rand(size=self.fc1.weight.shape, device=self.device, requires_grad=True)
        self.fc2_mask_weight = torch.rand(size=self.fc2.weight.shape, device=self.device, requires_grad=True)
        self.fc1_mask_bias = torch.rand(size=self.fc1.bias.shape, device=self.device, requires_grad=True)
        self.fc2_mask_bias = torch.rand(size=self.fc2.bias.shape, device=self.device, requires_grad=True)

    def training_step_nn(self, inputs, outputs, masking_only=False):
        self.fc1.weight.requires_grad = True
        self.fc1.bias.requires_grad = True
        self.fc2.weight.requires_grad = True
        self.fc2.bias.requires_grad = True
        self.optimizer.zero_grad(set_to_none=True)
        preds = self.forward(inputs, masking=False)
        loss = self.loss_func(preds, outputs)
        loss.backward()
        if masking_only:
            fc1_mask_weight_bool = self.get_mask(self.fc1_mask_weight)
            fc2_mask_weight_bool = self.get_mask(self.fc2_mask_weight)
            fc1_mask_bias_bool = self.get_mask(self.fc1_mask_bias)
            fc2_mask_bias_bool = self.get_mask(self.fc2_mask_bias)
            self.fc1.weight.grad[torch.logical_not(fc1_mask_weight_bool)] = 0
            self.fc2.weight.grad[torch.logical_not(fc2_mask_weight_bool).reshape(1, self.hidden_layer_shape)] = 0
            self.fc1.bias.grad[torch.logical_not(fc1_mask_bias_bool)] = 0
            self.fc2.bias.grad[torch.logical_not(fc2_mask_bias_bool)] = 0
        self.optimizer.step()
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False
        return loss.item()
