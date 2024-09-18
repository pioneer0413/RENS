import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck
import snntorch as snn
import snntorch.utils as utils

class SpikingCNN_v1(nn.Module):
    def __init__(self, beta, spike_grad):
        super().__init__()

        ###
        self.block1 = nn.Sequential(nn.Conv2d(2, 12, 5),
                                    nn.MaxPool2d(2),
                                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                                    nn.Conv2d(12, 32, 5),
                                    nn.MaxPool2d(2),
                                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                                    nn.Flatten(1),
                                    nn.Linear(32*5*5, 10),
                                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
                                    )
        ##*

    def forward(self, x): # << [t, b, c, x, y]
        spk_rec = []
        #mem_rec = []
        utils.reset(self.block1)

        num_steps = x.size(0)
        for step in range(num_steps):
            spk_out = self.block1(x[step]) # << [b, c, x, y] at time step 't'
            spk_rec.append(spk_out)
            #mem_rec.append(mem_out)

        return torch.stack(spk_rec)
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        ###
        self.block1 = nn.Sequential(nn.Conv2d(2, 12, 5),
                                    nn.MaxPool2d(2),
                                    nn.ReLU(),
                                    nn.Conv2d(12, 32, 5),
                                    nn.MaxPool2d(2),
                                    nn.ReLU(),
                                    nn.Flatten(1),
                                    nn.Linear(32*5*5, 10),
                                    nn.Sigmoid()
                                    )
        ##*

    def forward(self, x): # << [t, b, c, x, y]
        output_rec = []
        num_steps = x.size(0)
        for step in range(num_steps):
            output = self.block1(x[step]) # << [b, c, x, y] at time step 't'
            output_rec.append(output)
        return torch.stack(output_rec).sum(dim=0) # >> [t, b, num_classes] >> [b, num_classes]
    
class CustomResNet50(ResNet):
    def __init__(self, num_classes=1000):
        super(CustomResNet50, self).__init__(
            block=Bottleneck,
            layers=[3,4,6,3],
        )
        self.conv1 = nn.Conv2d(
            in_channels=2,  # 입력 채널을 2로 변경
            out_channels=self.conv1.out_channels,
            kernel_size=self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            bias=False
        )
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        output_rec = []
        num_steps = x.size(0)
        for step in range(num_steps):
            output = super(CustomResNet50, self).forward(x[step])
            output_rec.append(output)
        return torch.stack(output_rec).sum(dim=0)
        