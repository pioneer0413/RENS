import torch
import torch.nn as nn
import torch.nn.functional as F

# Single Noise Classification Convolutional Neural Network
class SNC_CNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 첫 번째 합성곱 레이어
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 두 번째 합성곱 레이어
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 세 번째 합성곱 레이어
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 완전 연결 레이어
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # 첫 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv1(x)))
        # 두 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv2(x)))
        # 세 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv3(x)))
        # 텐서를 평탄화
        x = x.view(-1, 1024)
        # 완전 연결 레이어 + ReLU
        x = F.relu(self.fc1(x))
        # 출력 레이어 (이진 분류)
        x = torch.sigmoid(self.fc2(x))
        return x

# Multi Noise Classification Convolutional Neural Network
class MNC_CNN(nn.Module):
    def __init__(self):
        super(MNC_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # assuming input images are 32x32
        self.fc2 = nn.Linear(128, 5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x