# Experiment no.14

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from generate_noised_dataset import generate_one_noisy_image

import os
#os.environ['CUDA_LAUNCH_BLOCKING']="1"
#os.environ['TORCH_USE_CUDA_DSA']="1"
import re
import argparse
from datetime import datetime

path_root = '/home/hwkang/jupyter/root/'
path_dataset = path_root + 'dataset/'
path_result = path_root + 'result/' + 'exp14/'
path_result_image = path_result + 'image/'
path_result_loss = path_result + 'loss/'
path_result_accuracy = path_result + 'accuracy/'
path_result_meta = path_result + 'meta/'

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")

def current_time_str():
    now = datetime.now()
    formatted_time = now.strftime("%y-%m-%d_%H-%M-%S")
    return formatted_time

def get_next_xid(path):
    max_id = -1
    pattern = re.compile(r'^(\d+)_')
    for filename in os.listdir(path):
        m = pattern.match(filename)
        if m:
            current_id = int(m.group(1))
            if(current_id > max_id):
                max_id = current_id
    return max_id + 1

def write_metadata(path, status):
    with open(path, 'a') as file:
        file.write(f'\nstatus: {status}')

def get_subset(dataset, ratio):
    dataset_size = len(dataset)
    subset_size = int(dataset_size * ratio)
    remainder_size = dataset_size - subset_size
    subset, _ = random_split(dataset, [subset_size, remainder_size])
    return subset

class NoisedDataset(Dataset):
    def __init__(self, data_loader, noise_type='gaussian'):
        self.x = []
        self.y = []
        for image, label in data_loader:
            image = image.squeeze(0)
            if( np.random.rand() >= 0.5 ):
                self.x.append( generate_one_noisy_image(image, intensity=np.random.rand(), noise_type=noise_type) )
                self.y.append( 1 )
            else:
                self.x.append( image )
                self.y.append( 0 )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_data = self.x[idx]
        y_data = self.y[idx]
        return x_data, y_data

def visualize_noisy_sample(loader, noise_type, path):
    # Unzip sample_batch to 10 samples
    x, y = next(iter(loader)) # [n, 64, 1, 28, 28] -> [64, 1, 28, 28]
    
    samples = [(x[i], y[i]) for i in range(10)] # [64, 1, 28, 28] -> 10 * [1, 28, 28]
        
    # Draw 2 x 5 grid image
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i][0].permute(1,2,0), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Label:{samples[i][1].item()}')
        
    # Output the image to path
    plt.tight_layout()
    current_time = current_time_str()
    file_name = f'{xid:03d}_{noise_type}_sample_{current_time}.png'
    plt.savefig(f'{path}{file_name}')

def visualize_epoch_loss(noise_type, epoch_loss, path):
    plt.figure(figsize=(10,6))
    plt.plot(epoch_loss)
    plt.title('Epoch Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    current_time = current_time_str()
    file_name = f'{xid:03d}_{noise_type}_epoch{len(epoch_loss)}_{current_time}.png'
    plt.savefig(f'{path}{file_name}')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 첫 번째 합성곱 레이어
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 두 번째 합성곱 레이어
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 세 번째 합성곱 레이어
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 완전 연결 레이어
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # 첫 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv1(x)))
        # 두 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv2(x)))
        # 세 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv3(x)))
        # 텐서를 평탄화
        x = x.view(-1, 64 * 3 * 3)
        # 완전 연결 레이어 + ReLU
        x = F.relu(self.fc1(x))
        # 출력 레이어 (이진 분류)
        x = torch.sigmoid(self.fc2(x))
        return x

def main(args):
    if args.verbose:
        print("Verbose mode is enabled.")

    # Main implementation

    # Data preprocessing
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root=path_dataset, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=path_dataset, train=False, download=True, transform=transform)

    train_ratio = round(args.train_dataset_ratio, 2)
    test_ratio = round(args.test_dataset_ratio, 2)
    
    train_dataset = get_subset(train_dataset, train_ratio)
    test_dataset = get_subset(test_dataset, test_ratio)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    noised_train_dataset = NoisedDataset(train_loader, noise_type=args.noise_type)
    noised_test_dataset = NoisedDataset(test_loader, noise_type=args.noise_type)

    noised_train_loader = DataLoader(noised_train_dataset, batch_size=args.batch_size, shuffle=True)
    noised_test_loader = DataLoader(noised_test_dataset, batch_size=args.batch_size, shuffle=False)

    # Sanity check; output noisy sample
    visualize_noisy_sample(noised_test_loader, args.noise_type, path_result_image)

    # Setup hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN()
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = args.epoch

    epoch_loss_rec = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in noised_train_loader:
            # 이진 분류를 위해 라벨을 0 또는 1로 변경 (임시로)
            labels = labels.float().unsqueeze(1)
    
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 옵티마이저 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(inputs)
            
            # 손실 계산
            loss = criterion(outputs, labels)
            
            # 역전파
            loss.backward()
            
            # 옵티마이저 업데이트
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        if( epoch % 10 == 9 ):
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        epoch_loss_rec.append(epoch_loss)

    visualize_epoch_loss(args.noise_type, epoch_loss_rec, path_result_loss)

    # Evaluation
    model.eval()
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in noised_test_loader:
            labels = labels.float().unsqueeze(1)  # 이진 분류를 위해 라벨을 0 또는 1로 변경 (임시로)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    current_time = current_time_str()
    file_name = f'{xid:03d}_{args.noise_type}_accuracy_{current_time}.png'
    plt.savefig(f'{path_result_accuracy}{file_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Sanity check; directory existence
    ensure_directory(path_result_image)
    ensure_directory(path_result_loss)
    ensure_directory(path_result_accuracy)
    ensure_directory(path_result_meta)
    
    # Command-line arguments
    parser.add_argument('-n', '--noise_type', type=str, required=False, default='gaussian') # noise_type
    parser.add_argument('-t', '--train_dataset_ratio', type=float, required=False, default=1.0) # train_dataset_size
    parser.add_argument('-y', '--test_dataset_ratio', type=float, required=False, default=1.0) # test_dataset_size
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64) # batch_size
    parser.add_argument('-e', '--epoch', type=int, required=False, default=50) # epoch
    parser.add_argument('-i', '--output_path_image', type=str, required=False, default=path_result_image)
    parser.add_argument('-l', '--output_path_loss', type=str, required=False, default=path_result_loss)
    parser.add_argument('-a', '--output_path_accuracy', type=str, required=False, default=path_result_accuracy)
    parser.add_argument('-m', '--output_path_meta', type=str, required=False, default=path_result_meta)
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Enable verbose mode")

    # Parsing arguments
    args = parser.parse_args()

    # Write meta data
    current_time = current_time_str()
    xid = get_next_xid(path_result_meta)
    lines = [
        f'datetime: {current_time}',
        f'noise_type: {args.noise_type}',
        f'train_dataset_ratio: {args.train_dataset_ratio}',
        f'train_dataset_ratio: {args.test_dataset_ratio}',
        f'batch_size: {args.batch_size}',
        f'epoch: {args.epoch}',
        f'output_path_image: {args.output_path_image}',
        f'output_path_loss: {args.output_path_loss}',
        f'output_path_accuracy: {args.output_path_accuracy}',
        f'output_path_meta: {args.output_path_meta}',
        f'verbose: {args.verbose}'
    ]
    file_path = path_result_meta + f'{xid:03d}_{args.noise_type}_{current_time}.txt'
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    
    # Execution
    try:
        main(args)
        write_metadata(file_path, 'SUCCESS')
    except KeyboardInterrupt:
        write_metadata(file_path, 'HALTED')
    except Exception as e:
        write_metadata(file_path, f'FAILED({e})')