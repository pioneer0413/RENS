import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import snntorch as snn
import snntorch.spikegen as spikegen
from snntorch import surrogate

# 파라미터 설정
batch_size = 64
time_steps = 100  # 총 타임 스텝
#chunk_size = 100   # 한 번에 처리할 타임 스텝의 크기
num_epochs = 5    # 에폭 수
learning_rate = 0.001  # 학습률
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Anomaly Detection 활성화 (문제의 원인을 찾기 위해)
#torch.autograd.set_detect_anomaly(True)

# MNIST 데이터셋 로드
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=True)

time_steps=1
# SNN 모델 정의
class SNNModel(nn.Module):
    def __init__(self):
        super(SNNModel, self).__init__()
        
        # 스파이킹 뉴런 레이어 설정
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  # [16, 24, 24]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # [32, 20, 20]
        self.fc1 = nn.Linear(32 * 20 * 20, 10)  # Fully connected layer, 32 * 20 * 20 = 12800
        
        # snntorch의 Leaky Integrate and Fire (LIF) 뉴런을 사용
        self.lif1 = snn.Leaky(beta=0.9)
        self.lif2 = snn.Leaky(beta=0.9)
        self.lif3 = snn.Leaky(beta=0.9)
    
    #def forward(self, x, chunk_size):
    def forward(self, x):
        # 결과를 누적하기 위한 텐서 초기화
        outputs = torch.zeros(x.size(0), 10, device=x.device)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # 전체 타임 스텝을 한 번에 처리
        x_spikes = x
        #x_spikes = spikegen.rate(x, num_steps=time_steps)
        x_spikes = x_spikes.view(time_steps, x.size(0), 1, 28, 28)
        
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []

        for step in range(time_steps):
            spk1, mem1 = self.lif1(self.conv1(x_spikes[step]), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            spk2 = spk2.view(spk2.size(0), -1)
            spk3, mem3 = self.lif3(self.fc1(spk2), mem3)
            spk3_rec.append(spk3)

        outputs = torch.stack(spk3_rec, dim=1).sum(dim=1)  # 타임 스텝 차원을 합쳐서 최종 출력 계산
        
        '''
        x_spikes = x_spikes.view(-1, 1, 28, 28)
        spk1, mem1 = self.lif1(self.conv1(x_spikes), mem1)
        spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
        spk2 = spk2.view(spk2.size(0), -1)

        spk3, mem3 = self.lif3(self.fc1(spk2), mem3)
        #current_batch_size = spk3.size(0) // time_steps
        outputs = outputs + spk3.view(batch_size, time_steps, -1).sum(1)
        '''
        
        """
        # 타임 스텝을 나누어 처리
        for i in range(0, time_steps, chunk_size):
            # 현재 chunk에 대한 스파이크 트레인 생성 (Rate Encoding)
            x_chunk = spikegen.rate(x, num_steps=chunk_size)
            
            # Time dimension을 batch dimension에 합치기
            x_chunk = x_chunk.view(-1, 1, 28, 28)  # [batch_size * chunk_size, channel, height, width]

            # SNN 레이어를 거쳐 처리
            spk1, mem1 = self.lif1(self.conv1(x_chunk))
            spk2, mem2 = self.lif2(self.conv2(spk1))
            '''
            temp = spk2.view(spk2.size(0), -1)
            #spk2 = temp
            #spk2 = spk2.view(spk2.size(0), -1)  # [batch_size * chunk_size, 32 * 20 * 20]
            try:
                tempfc = self.fc1(temp)
                print(f"tempfc shape after fc1: {tempfc.shape}")  # 디버깅을 위해 출력
            except Exception as e:
                print(f"Error during fc1 processing: {e}")
                raise
            #tempfc = self.fc1(temp)
            spk3, mem3 = self.lif3(tempfc)
            #spk3, mem3 = self.lif3(self.fc1(temp))
            #spk3, mem3 = self.lif3(self.fc1(spk2.clone()))
            '''
            spk2 = spk2.view(spk2.size(0), -1)  # [batch_size * chunk_size, 32 * 20 * 20]
            # 크기 확인
            #print(f"spk2 shape before fc1: {spk2.shape}")  # 디버깅을 위해 출력
            
            spk3, mem3 = self.lif3(self.fc1(spk2))
            #print(f"spk3 shape after fc1: {spk3.shape}")  # 디버깅을 위해 출력

            
            # Adjust the view for the last chunk
            current_batch_size = spk3.size(0) // chunk_size
            # In-place 연산을 피하면서 최종 출력에 대해 타임 스텝의 결과를 누적
            outputs = outputs + spk3.view(current_batch_size, chunk_size, -1).sum(1).clone()
        """

        return outputs

# 모델 초기화
model = SNNModel().to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 훈련 루프
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 모델에 데이터 전달
        output = model(data)
        # 모델에 데이터 전달 (chunk_size 사용)
        #output = model(data, chunk_size)
        
        loss = criterion(output, targets)
        loss.backward()  # retain_graph=False로 설정하여 단일 backward 호출
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 검증 단계
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        
        # 모델에 데이터 전달
        output = model(data)
        # 모델에 데이터 전달 (chunk_size 사용)
        #output = model(data, chunk_size)
        
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

accuracy = 100. * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
