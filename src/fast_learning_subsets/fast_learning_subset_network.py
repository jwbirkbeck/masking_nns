from torch import nn
from torch import optim

class FastLearningSubsetNetwork(nn.Module):
    def __init__(self, learning_rate, subset_fraction, input_shape, hidden_layer_shape, output_shape, device):
        super(FastLearningSubsetNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.subset_fraction = subset_fraction
        self.input_shape = input_shape
        self.hidden_layer_shape = hidden_layer_shape
        self.output_shape = output_shape
        self.fc1 = nn.Linear(input_shape, hidden_layer_shape)
        self.fc2 = nn.Linear(hidden_layer_shape, output_shape)
        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.to(self.device)

    def forward(self, input):
        activation = nn.ReLU()
        output = self.fc1(input)
        output = activation(output)
        output = self.fc2(output)
        return output

    def training_step(self, inputs, outputs):
        self.optimizer.zero_grad(set_to_none=True)
        preds = self.forward(inputs)
        loss = self.loss_func(preds, outputs)
        # Here, identify the self.subset_fraction amount of neurons to keep the gradients for.
        # Mask all other gradients to zero.
        


        loss.backward()
        self.optimizer.step()
        return loss.item()