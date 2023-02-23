import torch
from torchvision import datasets, transforms
from TargetNetwork import TargetStackingNetwork, target_polyak_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alpha = 0.1
layer_sizes = (784, 64, 128, 10)
network = TargetStackingNetwork(layer_sizes=layer_sizes, alpha=alpha, device=device)
network_t1 = TargetStackingNetwork(layer_sizes=layer_sizes, alpha=0, device=device)
network_t2 = TargetStackingNetwork(layer_sizes=layer_sizes, alpha=0, device=device)
network_t3 = TargetStackingNetwork(layer_sizes=layer_sizes, alpha=0, device=device)

target_polyak_update(source_model=network, target_model=network_t1, initial_update=True)
target_polyak_update(source_model=network_t1, target_model=network_t2, initial_update=True)
target_polyak_update(source_model=network_t2, target_model=network_t3, initial_update=True)

# network t3 <- network t2 <- network t1 <- network

# Load data
# Take a mini-batch and train network
# polyak update networks t1, t2, t3
# assess loss on new batch for each

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, ), (0.5,),)])
trainset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data", download=True, train=True, transform=transform)
testset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data", download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

epochs = 10
network.train()
for epoch in range(epochs):
