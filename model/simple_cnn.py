"""
File name: SimpleCNN.py
Purpose: 단순한 CNN모델의 반복적 재사용

Change log:
  - 2024-08-06: 코드 설명 주석 추가 (v1.0.1)
  - 2024-08-29: CLS_CNN Deprecated Warning 코드 추가 (CNN으로 대체) (v1.1.0)
  
Last update: 2024-08-29 16:38 Thu.
Last author: mwkim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Purpose: 범용 CNN모델
Parameters: __init__ 한정
    - channels (int): 입력이미지의 채널수(RGB: 3, Grayscale: 1)
    - classes  (int): 출력 클래스의 수
Methods: 
    - _create_conv_block: 컨볼루션 블록 생성
Constraints:
    - 이진분류 시, BCELoss함수가 아닌 BCEWithLogitsLoss 손실함수 사용
Notes: "..." (optional)
Last update: 2024-08-29 17:04 Thu.
Last author: mwkim
"""
class CNN(nn.Module):
    def __init__(self, channels=3, classes=10):
        super(BNC_CNN, self).__init__()
        
        # 합성곱 블록 정의
        self.conv_block1 = self._create_conv_block(channels, 16)
        self.conv_block2 = self._create_conv_block(16, 32)
        self.conv_block3 = self._create_conv_block(32, 64)
        
        # Adaptive Pooling을 사용하여 고정 크기 출력 확보
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, classes)

    def _create_conv_block(self, in_channels, out_channels):
        """
        합성곱 블록을 생성하는 헬퍼 메서드
        :param in_channels: 입력 채널 수
        :param out_channels: 출력 채널 수
        :return: 합성곱 블록 (Sequential)
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        """
        순전파 정의
        :param x: 입력 데이터 텐서
        :return: 네트워크의 예측 출력
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Adaptive Pooling을 통해 고정된 크기의 출력으로 변환
        x = self.adaptive_pool(x)
        
        # 텐서를 평탄화
        x = x.view(x.size(0), -1)
        
        # 완전 연결 레이어 및 드롭아웃 적용
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # BCEWithLogitsLoss
        return x


# Binary Noise Classification Convolutional Neural Network
"""
TODO: 클래스 주석 달기 >> (v1.0.2)
"""
class BNC_CNN(nn.Module):
    def __init__(self):
        print("DEPRECATED WARNING: This(BNC_CNN) class will be unable since (v1.2.0).")
        super(BNC_CNN, self).__init__()
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
"""
TODO: 클래스 주석 달기 >> (v1.0.2)
"""
class MNC_CNN(nn.Module):
    def __init__(self):
        print("DEPRECATED WARNING: This(MNC_CNN) class will be unable since (v1.2.0).")
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

"""
Purpose: 다중 출력층을 갖는 단순한 CNN모델
Attributes: 
  - conv1, conv2, conv3 : convolution layer. 부모클래스로부터의 상속
  - pool : pooling layer. 부모클래스로부터의 상속
  - fc1, fc2 : fully connected layer. 부모클래스로부터의 상속
Methods: 
  - __init__ : 모델의 레이어 구성. 부모클래스의 오버라이딩
  - forward : 순전파 함수. 부모클래스의 오버라이딩
Relationships:
  - Inherits:
    - torch.nn.Module
  - Compositions:
    - torch.nn.Conv2d()
    - torch.nn.MaxPool2d()
    - torch.nn.Linear()
    - torch.nn.functional.relu()
Constraints:
  - num_classes : 1 이상 (2 이상일 때 제 기능 수행)
  - input 이미지는 32x32x3(32x32 pixel, 3channel) 크기로 한정
Last update: 2024-08-29 16:38 Thu.
Last author: mwkim
"""
class CLS_CNN(nn.Module):
    def __init__(self, num_classes=5):
        print("DEPRECATED WARNING: This(CLS_CNN) class will be unable since (v1.2.0).")
        # num_classes : 최종 출력층의 수. input을 classify할 class의 수
        super(CLS_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # assuming input images are 32x32
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x