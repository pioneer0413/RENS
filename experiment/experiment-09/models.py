# PyTorchs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR #학습률 스케줄링
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import resnet50, resnet101

# SnnTorchs
import snntorch as snn
from snntorch import surrogate
#from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt 

class CNN(nn.Module):
    def __init__(self, dataset_type='cifar10', dropout_rate=0.5):
        super().__init__()

        self.dataset_type = dataset_type

        # Initialize layers
        if dataset_type == 'mnist':
            self.conv1 = nn.Conv2d(1, 12, 5)
        elif dataset_type == 'nmnist':
            self.conv1 = nn.Conv2d(2, 12, 5)
        else:
            self.conv1 = nn.Conv2d(3, 12, 5)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(12, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)

        if dataset_type == 'mnist':
            self.fc1 = nn.Linear(64*4*4, 10)
        else:
            self.fc1 = nn.Linear(64*5*5, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):

        x = F.max_pool2d(self.bn1(self.conv1(x)), 2)
        x = self.relu1(x)
        x = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = self.relu2(x)
        if self.dataset_type == 'mnist':
            x = x.view(-1, 64*4*4)
        else:
            x = x.view(-1, 64*5*5)
        x = self.dropout(x)
        x = self.fc1(x) #  >> [ batch_size, 10 ]
        x = self.bn3(x)

        return x

# Define Network
class SpikingCNN(nn.Module):
    def __init__(self, num_steps, dataset_type='cifar10', beta=0.9, spike_grad=surrogate.atan(), dropout_rate=0.5):
        super().__init__()
        
        self.num_steps = num_steps
        self.dataset_type = dataset_type

        # Initialize layers
        if dataset_type == 'mnist':
            self.conv1 = nn.Conv2d(1, 12, 5)
        elif dataset_type == 'nmnist':
            self.conv1 = nn.Conv2d(2, 12, 5)
        else:
            self.conv1 = nn.Conv2d(3, 12, 5)
        self.bn1 = nn.BatchNorm2d(12)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.dropout = nn.Dropout(dropout_rate)
        
        if dataset_type == 'mnist':
            self.fc1 = nn.Linear(64*4*4, 10)
        else:
            self.fc1 = nn.Linear(64*5*5, 10)
        self.bn3 = nn.BatchNorm1d(10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.bn1(self.conv1(x)), 2)
        spk1, mem1 = self.lif3(cur1)

        cur2 = F.max_pool2d(self.bn2(self.conv2(spk1)), 2)
        spk2, mem2 = self.lif3(cur2)

        cur3 = self.dropout(spk2.view(spk2.shape[0], -1))
        cur3 = self.fc1(cur3)

        cur3 = self.bn3(cur3)
        spk3, mem3 = self.lif3(cur3)

        return spk3

class SpikingResNet50_1(nn.Module):
    def __init__(self, original_model):
        super(SpikingResNet50_1, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)
        
        # Copy the remaining layers
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc

    def forward(self, x, num_steps=10):
        x = self.conv1(x)
        x = self.bn1(x)

        spk_rec = []
        mem = self.lif1.reset_mem()
        for step in range(num_steps):
            spk, mem = self.lif1(x)
            spk_rec.append(spk)

        x = torch.mean(torch.stack(spk_rec, dim=0), dim=0)
        
        # Proceed with the rest of the ResNet
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class SpikingResNet50_2(nn.Module):
    def __init__(self, original_model):
        super(SpikingResNet50_2, self).__init__()

        # Copy the initial layers before Conv2_x
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.maxpool = original_model.maxpool

        # Modify the Conv2_x layer group by replacing ReLU with snn.Leaky
        self.layer1 = self.modify_layer_group(original_model.layer1)

        # Copy the remaining layers without modification
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc

    def modify_layer_group(self, layer_group):
        # Function to replace all ReLU activations with snn.Leaky in a given layer group
        for idx, layer in enumerate(layer_group):
            if isinstance(layer.relu, nn.ReLU):
                layer_group[idx].relu = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True)
        return layer_group

    def reset_mem(self):
        for module in self.modules():
            if isinstance(module, snn.Leaky):
                module.reset_mem()
    
    def forward(self, x, num_steps=10):
        # Forward pass through the modified network
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        # Forward pass through modified layer1 (Conv2_x)
        spk_rec = []
        self.reset_mem()
        for step in range(num_steps):
            spk = self.layer1(x)
            spk_rec.append(spk)

        # Average over time steps
        x = torch.mean(torch.stack(spk_rec, dim=0), dim=0)

        # Forward pass through the remaining layers
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x