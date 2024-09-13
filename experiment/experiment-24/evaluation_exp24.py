# Experiment no.24 evaluation

# Constants
from pathlib import Path
EXP_NO = 'exp24' # NOTICE: Change at each experiment!
DIRECTORY = current_file_path = Path(__file__).resolve()
PATH_ROOT = str(DIRECTORY.parent.parent.parent) + '/'
PATH_DATASET = PATH_ROOT + 'dataset/'
PATH_RESULT = PATH_ROOT + 'result/'
PATH_RESULT_ROOT = PATH_RESULT + EXP_NO + '/'
PATH_RESULT_IMAGE = PATH_RESULT_ROOT + 'image/'
PATH_RESULT_MODEL = PATH_RESULT_ROOT + 'model/'
PATH_RESULT_LOSS = PATH_RESULT_ROOT + 'loss/'
PATH_RESULT_ACCURACY = PATH_RESULT_ROOT + 'accuracy/'
PATH_RESULT_META = PATH_RESULT_ROOT + 'meta/'
PATH_UTILITY = PATH_ROOT + 'utility/'

# Imports
import sys
sys.path.append(PATH_ROOT)
sys.path.append(PATH_DATASET)
sys.path.append(PATH_RESULT)

# PyTorch family
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import surrogate
import snntorch.functional as SF
from snntorch._layers.bntt import BatchNormTT2d

# Utilities
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import itertools
import traceback
from types import SimpleNamespace

'''
from utility.common import *
from utility.preprocess import *
from utility.synthesize import *
from utility.visualize import *
from model.SimpleCNN import *
from model.LearningUtils import *
'''
from utility.parser import *

from utility.statistic import *
from utility.common import *
from utility.preprocessing import *
from utility.synthesization import *
from utility.visualization import *
from model.simple_cnn import *
from model.learning_utility import *

'''
START of visualization
'''
def visualize_sample_from_list(pilot: bool, list, path: str=None):
    samples = []
    for _ in range(10):
        sample_idx = random.randint(0, len(list) - 1)
        samples.append(list[sample_idx])
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        image=samples[i]
        if isinstance(image, torch.Tensor):
            image=image.cpu().numpy()
        if image.ndim == 4:
            image = image[0]  # Remove batch dimension, now (C, H, W)
            label = label[0]

        # Handle 3D image (C, H, W) or (1, H, W)
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)  # Convert (1, H, W) to (H, W) for grayscale
            else:
                image = np.transpose(image, (1, 2, 0))  # Convert (C, H, W) to (H, W, C) for color images
        elif image.ndim != 2:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        ax.imshow(image, cmap='gray' if image.ndim == 2 else None)  # cmap='gray' for grayscale images
        ax.axis('off')

    if pilot:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

def visualize_sample_from_loader_list(pilot: bool, loader_list, path: str=None):
    samples = []

     # 10개의 샘플을 무작위로 선택하여 가져오기
    for _ in range(10):
        loader_idx = random.randint(0, len(loader_list) - 1)  # 무작위로 DataLoader 선택
        loader = loader_list[loader_idx]
        x, y, _ = next(iter(loader))  # 선택된 DataLoader에서 하나의 배치 가져오기

        # 무작위 배치에서 첫 번째 샘플 선택
        sample_idx = random.randint(0, len(x) - 1)
        samples.append((x[sample_idx], y[sample_idx]))

    # Create a 2 x 5 grid for images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        image = samples[i][0]
        label = samples[i][1]
            
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Handle 4D image (B, C, H, W): remove batch dimension
        if image.ndim == 4:
            image = image[0]  # Remove batch dimension, now (C, H, W)
            label = label[0]

        # Handle 3D image (C, H, W) or (1, H, W)
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)  # Convert (1, H, W) to (H, W) for grayscale
            else:
                image = np.transpose(image, (1, 2, 0))  # Convert (C, H, W) to (H, W, C) for color images
        elif image.ndim != 2:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        ax.imshow(image, cmap='gray' if image.ndim == 2 else None)  # cmap='gray' for grayscale images
        ax.axis('off')
        
        if hasattr(label, 'item'):
            label = label.item()
        
        ax.set_title(f"Label: {label}")

    if pilot:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(path)
'''
END of visualization
'''

'''
START of preprocessing
'''
def get_multiple_subsets_by_size(dataset: Dataset, subset_size: int, drop_last: bool = False):
    total_size = len(dataset)
    indices = list(range(total_size))
    batches = [Subset(dataset, indices[i : i + subset_size]) for i in range(0, total_size, subset_size)]

    if drop_last and len(batches[-1]) < subset_size:
        batches = batches[:-1]

    return batches
'''
END of preprocessing
'''

'''
START of model# Network Architecture
'''
class BNC_CNN_MNIST(nn.Module):
    def __init__(self):
        super(BNC_CNN_MNIST, self).__init__()
        # 첫 번째 합성곱 레이어 (입력 채널: 1, 출력 채널: 16)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 두 번째 합성곱 레이어 (입력 채널: 16, 출력 채널: 32)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 세 번째 합성곱 레이어 (입력 채널: 32, 출력 채널: 64)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 완전 연결 레이어
        self.fc1 = nn.Linear(64 * 3 * 3, 128)  # MNIST 크기(1x28x28)에 맞춰 조정
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # 첫 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv1(x)))
        # 두 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv2(x)))
        # 세 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv3(x)))
        # 텐서를 평탄화
        x = x.view(-1, 64 * 3 * 3)  # 64 채널의 3x3 특성 맵
        # 완전 연결 레이어 + ReLU
        x = F.relu(self.fc1(x))
        # 출력 레이어 (이진 분류)
        x = torch.sigmoid(self.fc2(x)).squeeze(1)
        return x


# SNN 모델 정의
class CNV_SNN(nn.Module):
    def __init__(self, snn_loss_function, time_steps):
        super(CNV_SNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  # [16, 24, 24]
        #self.bn1 = BatchNormTT2d(16, time_steps=time_steps)  # BatchNorm 추가
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # [32, 20, 20]
        #self.bn2 = BatchNormTT2d(32, time_steps=time_steps)  # BatchNorm 추가
        self.fc1 = nn.Linear(32 * 20 * 20, 10)  # Fully connected layer
        
        # Leaky Integrate-and-Fire Neurons
        spike_grad = surrogate.fast_sigmoid() if snn_loss_function else None
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad, output=True)
    
    def forward(self, x, encode, num_steps, snn_loss_function=False):
        # 결과를 누적하기 위한 텐서 초기화 / (might be) encoded input => x.size(0) is prohibited
        # for pytorch loss function
        if snn_loss_function is None:
            outputs = torch.zeros(x.size(1), 10, device=x.device)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk3_rec = []
        if encode == 'latency':
            for step in range(num_steps):
                # 각 타임스텝마다 BatchNormTT2d 적용
                x_step = self.conv1(x[step])
                #x_step = self.bn1[step](x_step)  # ModuleList에서 개별 모듈 호출
                spk1, mem1 = self.lif1(x_step, mem1)
                
                x_step = self.conv2(spk1)
                #x_step = self.bn2[step](x_step)  # ModuleList에서 개별 모듈 호출
                spk2, mem2 = self.lif2(x_step, mem2)

                spk2 = spk2.view(spk2.size(0), -1)
                spk3, mem3 = self.lif3(self.fc1(spk2), mem3)
                spk3_rec.append(spk3)
                if spk3.any():
                    for _ in range(num_steps - step - 1):
                        spk3_rec.append(torch.zeros_like(spk3))
                    break

        else:
            for step in range(num_steps):
                # 각 타임스텝마다 BatchNormTT2d 적용
                x_step = self.conv1(x[step])
                #x_step = self.bn1[step](x_step)  # ModuleList에서 개별 모듈 호출
                spk1, mem1 = self.lif1(x_step, mem1)
                
                x_step = self.conv2(spk1)
                #x_step = self.bn2[step](x_step)  # ModuleList에서 개별 모듈 호출
                spk2, mem2 = self.lif2(x_step, mem2)

                spk2 = spk2.view(spk2.size(0), -1)
                spk3, mem3 = self.lif3(self.fc1(spk2), mem3)
                spk3_rec.append(spk3)

        # for pytorch loss function
        if snn_loss_function == False:
            outputs = torch.stack(spk3_rec, dim=1).sum(dim=1)  # 타임 스텝 차원을 합쳐서 최종 출력 계산
        else:
            outputs = torch.stack(spk3_rec)
            
        return outputs

'''
END of model
'''

'''
START of synthesization.py
'''
import random
def generate_complexly_noised_data(data, gaussian: float=0.0, snp: float=0.0, uniform: float=0.0, poisson: float=0.0,
                                  rescale: bool=True, normalize: bool=False, device: torch.device=None):
    #pri
    # nt(f'before:{data.shape}')
    ### 입력 데이터 자료형 확인
    if isinstance( data, torch.Tensor ):
        data = generate_gaussian(data=data, intensity=gaussian, 
                                       rescale=rescale, device=device)
        data = generate_salt_and_pepper(data=data, intensity=snp) # No need device
        data = generate_uniform(data=data, intensity=uniform, 
                                      rescale=rescale, device=device)
        data = generate_poisson(data=data, intensity=poisson, 
                                          rescale=rescale, device=device)
        noisy_data = data
    elif isinstance( data, np.ndarray ):
        data = generate_gaussian(data=data, intensity=gaussian, 
                                       rescale=rescale)
        data = generate_salt_and_pepper(data=data, intensity=snp)
        data = generate_uniform(data=data, intensity=uniform, 
                                      rescale=rescale)
        data = generate_poisson(data=data, intensity=poisson, 
                                      rescale=rescale)
        noisy_data = data
    else:
        raise TypeError("ERROR: The type of data is incorrect. Expected torch.Tensor or numpy.ndarray.")
    ###

    ### 정규화
    if normalize is True:
        max_val, min_val = noisy_data.max(), noisy_data.min()
        noisy_data = ( noisy_data - min_val )/( max_val - min_val )
    ###
    #print(f'after:{data.shape}')
    return noisy_data


class RandomlyNoisedDataset(Dataset):
    def __init__(self, data_loader, same_value=False, device=None):
        self.x = []
        self.y = []
        self.meta = {}

        if same_value is True:
            intensity = self.create_gaussian_intensity()
            for noise_type in ['gaussian', 'snp', 'uniform', 'poisson']:
                self.meta[noise_type] = intensity

        else:
            for noise_type in ['gaussian', 'snp', 'uniform', 'poisson']:
                self.meta[noise_type] = self.create_gaussian_intensity()
        
        for image, label in data_loader:
            image, label = image.to(device), label.to(device)
            image, label = image.squeeze(0), label.squeeze(0)
            self.x.append(generate_complexly_noised_data(
                image, gaussian=self.meta['gaussian'], snp=self.meta['snp'],
                uniform=self.meta['uniform'], poisson=self.meta['poisson'],
                rescale=True, normalize=True, device=device))
            self.y.append(label)
                    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_data = self.x[idx]
        y_data = self.y[idx]
        meta_data = self.meta
        return x_data, y_data, meta_data

    def create_gaussian_intensity(self, gaussian_average=0, min_val=0, max_val=1, sigma_reduction=10):
        if sigma_reduction==0:
            return np.random.rand()
        while True:
            intensity = np.random.rand()/sigma_reduction + gaussian_average
            if (intensity <= max_val) and (intensity >= min_val):
                return intensity
                

def snn_training_loop(args, paths, encode, model, train_loader, device, scheduler, early_stopping, optimizer, criterion):
    epoch_loss_rec = []
    max_epoch_loss = math.inf
    for epoch in range(args.epoch):
        last_lr = scheduler.get_last_lr()[0]
        model.train() #change model's mode to train
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 옵티마이저 초기화
            optimizer.zero_grad()
            
            # 순전파
            #print(f"{encode} Original inputs shape before: {inputs.shape}")
            match encode:
                case 'rate':
                    inputs = spikegen.rate(inputs, num_steps=args.steps)
                case 'latency':
                    inputs = spikegen.latency(inputs, tau=0.1, num_steps=args.steps)
            #print(f"{encode} Original inputs shape after: {inputs.shape}")
            #inputs = inputs.view(args.steps, inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3))
            outputs = model(inputs, encode, args.steps, args.enable_snn_loss_function)
            
            # 손실 계산
            loss = criterion(outputs, labels)
            
            # 역전파
            loss.backward()
            
            # 옵티마이저 업데이트
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Save best model
        if( epoch_loss < max_epoch_loss ):
            print(f'{encode} Model saved: Epoch [{epoch+1}] [Current] {epoch_loss:.4f} << {max_epoch_loss:.4f} [Max]')
            torch.save(model.state_dict(), getattr(paths, encode).model)
            best_model_state = model.state_dict()
            best_epoch_idx = epoch + 1
            max_epoch_loss = epoch_loss
        
        # 학습률 감소 스케줄러에 검증 손실 전달
        if args.lr_scheduler:
            scheduler.step(epoch_loss)

            if scheduler.get_last_lr()[0] < (last_lr / 2):
                last_lr = scheduler.get_last_lr()[0]
                with open(paths.meta, 'a') as file:
                    file.write(f'\n{encode}: learning rate changed to {last_lr} at Epoch {epoch + 1}')
                print(f'{encode}: learning rate changed to {last_lr} at Epoch {epoch + 1}')
            
            # 학습률 확인 및 출력
            if args.verbose:
                print(f"{encode}: Learning rate after epoch {epoch + 1}: {scheduler.get_last_lr()[0]}")
        
        if( epoch % 10 == 9 ):
            print(f"{encode}: Epoch {epoch + 1}/{args.epoch}, Loss: {epoch_loss:.4f}")
        
        epoch_loss_rec.append(epoch_loss)

        if args.early_stopping:
            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print(f"{encode}: Early stopping at epoch {epoch+1} (epoch loss:{epoch_loss:.4f})")
                with open(paths.meta, 'a') as file:
                    file.write(f"\n{encode}: Early stopping at epoch {epoch+1} (epoch loss:{epoch_loss:.4f})")
                break

    # Load model parameters at best epoch loss
    if args.verbose:
        print(f'{encode}: Load model state at best epoch loss [{best_epoch_idx}]')
    model.load_state_dict(best_model_state)
    visualize_epoch_loss(pilot=False, epoch_loss=epoch_loss_rec, path=getattr(paths, encode).loss)

def snn_evaluation(args, paths, encode, model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 모델에 데이터 전달
            #print(f"before:{inputs.shape}")
            match encode:
                case 'rate':
                    inputs = spikegen.rate(inputs, num_steps=args.steps)
                case 'latency':
                    inputs = spikegen.latency(inputs, tau=0.1, num_steps=args.steps)
            #print(f"after:{inputs.shape}")
            #inputs = inputs.view(args.steps, inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3))
            output = model(inputs, encode, args.steps, args.enable_snn_loss_function)
            if args.enable_snn_loss_function == False:
                _, predicted_class = output.max(1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted_class.cpu().numpy())
            else:
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(output.sum(dim=0).argmax(dim=1).cpu().numpy())

    visualize_confusion_matrix(pilot=False, all_labels=all_labels, all_predictions=all_predictions, num_classes=10, path=getattr(paths, encode).accuracy)

    accuracy, _, _, _ = get_classification_metrics(all_labels, all_predictions, None)
    print(f'SNN {encode} Accuracy: {accuracy:.6f}')

def snn_train_pipeline(args, paths, device, encode, train_loader, test_loader):
    print_verbose(args.verbose, f"{encode} is on way")
    start_time = time.time()
    # Hyperparameter
    model = CNV_SNN(args.enable_snn_loss_function, args.steps)
    model = model.to(device)
    if args.enable_snn_loss_function:
        if encode == 'latency':
            criterion = SF.ce_max_membrane_loss()
        else:
            criterion = SF.ce_count_loss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    # 학습률 스케줄링
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # early stopping
    early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

    print_verbose(args.verbose, f"{encode} is on way for training")
    # Training loop
    snn_training_loop(args, paths, encode, model, train_loader, device, scheduler, early_stopping, optimizer, criterion)

    print_verbose(args.verbose, f"{encode} is on way for testing")
    # Test loop (evaluation)
    snn_evaluation(args, paths, encode, model, test_loader, device)
    
    
    end_time = time.time()
    print(f"{end_time-start_time} second at {encode} encoding train")

    return model

def cnn_training_loop(args, paths, model, train_loader, device, scheduler, early_stopping, optimizer, criterion):
    epoch_loss_rec = []
    max_epoch_loss = math.inf
    for epoch in range(args.epoch):
        last_lr = scheduler.get_last_lr()[0]
        model.train() #change model's mode to train
        running_loss = 0.0
        for inputs, labels in train_loader:
            labels = labels.float()
            #labels = labels.float().unsqueeze(1)            
            inputs, labels = inputs.to(device), labels.to(device)
            
            #print(f"Original data shape: {inputs.shape}, Original target shape: {labels.shape}")
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
        
        # Save best model
        if( epoch_loss < max_epoch_loss ):
            print(f'Model saved: Epoch [{epoch+1}] [Current] {epoch_loss:.4f} << {max_epoch_loss:.4f} [Max]')
            torch.save(model.state_dict(), paths.cnn.model)
            best_model_state = model.state_dict()
            best_epoch_idx = epoch + 1
            max_epoch_loss = epoch_loss
        
        # 학습률 감소 스케줄러에 검증 손실 전달
        if args.lr_scheduler:
            scheduler.step(epoch_loss)

            if scheduler.get_last_lr()[0] < (last_lr / 2):
                last_lr = scheduler.get_last_lr()[0]
                with open(paths.meta, 'a') as file:
                    file.write(f'\nlearning rate changed to {last_lr} at Epoch {epoch + 1}')
                print(f'learning rate changed to {last_lr} at Epoch {epoch + 1}')
            
            # 학습률 확인 및 출력
            if args.verbose:
                print(f"Learning rate after epoch {epoch + 1}: {scheduler.get_last_lr()[0]}")
        
        if( epoch % 10 == 9 ):
            print(f"Epoch {epoch + 1}/{args.epoch}, Loss: {epoch_loss:.4f}")
        
        epoch_loss_rec.append(epoch_loss)

        if args.early_stopping:
            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1} (epoch loss:{epoch_loss:.4f})")
                with open(paths.meta, 'a') as file:
                    file.write(f"\nEarly stopping at epoch {epoch+1} (epoch loss:{epoch_loss:.4f})")
                break
    
    # Load model parameters at best epoch loss
    if args.verbose:
        print(f'Load model state at best epoch loss [{best_epoch_idx}]')
    model.load_state_dict(best_model_state)
    visualize_epoch_loss(pilot=False, epoch_loss=epoch_loss_rec, path=paths.cnn.loss)

def cnn_evaluation(args, paths, model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.float()
            #labels = labels.float().unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    visualize_confusion_matrix(pilot=False, all_labels=all_labels, all_predictions=all_predictions, num_classes=2, path=paths.cnn.accuracy)

    accuracy, precisions, recalls, f1_scores = get_classification_metrics(all_labels, all_predictions)
    print(f'CNN Accuracy : {accuracy:.6f}, {precisions}, {recalls}, {f1_scores}')

    ## TODO : save to csv

def cnn_train_pipeline(args, paths, device, train_loader, test_loader):
    # Hyperparaeter
    model = BNC_CNN_MNIST()
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 학습률 스케줄링
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Training loop
    cnn_training_loop(args, paths, model, train_loader, device, scheduler, early_stopping, optimizer, criterion)

    # Test loop (evaluation)
    cnn_evaluation(args, paths, model, test_loader, device)

    return model
'''
def relabel_list_or_not(rate_model, latency_model, encoding_A, encoding_B, device, loader=None, loader_list=None):
    if loader is None and loader_list is None:
        raise CustomError("No loader or loader list")
    elif loader is not None:
        return relabel_based_on_evaluation(loader, rate_model, latency_model, encoding_A, encoding_B, device)
    elif loader_list is not None:
        return 
'''
# Define a function to evaluate each model on each dataset and relabel with the model index (0 or 1)
def relabel_based_on_evaluation(args, ref_splited_loader_list, loader_list, rate_model, latency_model, device):

    append_only = -1
    total_label_count = [0, 0]
    model_label_list = []  # List to store model indices (0 or 1) as new labels
    data_list = []
    hard_noise_idx=0

    while True:
        print(f"Attempt : {append_only} / total_label_count: [{total_label_count[0]},{total_label_count[1]}]")
        #label_count = [0, 0]
        
        for loader in loader_list:
            rate_model.eval()
            latency_model.eval()
            
            rate_model_correct = 0
            latency_model_correct = 0
            total = 0

            intensity = {}

            with torch.no_grad():
                for data, target, meta in loader:
                    intensity = {key: f"{value.item():.2f}" for key, value in meta.items()}
                    # Move data and target to the specified device (GPU or CPU)
                    data, target = data.to(device), target.to(device)
                    
                    # Check the shapes here
                    #print(f"Original data shape: {data.shape}, Original target shape: {target.shape}")
                    
                    data, target = data.squeeze(0), target.squeeze(0)
                    #data = data.squeeze(0)  # This might be modifying the batch size, be careful
                    #target = target.squeeze(0)  # Same here

                    # Check shapes again after squeezing
                    #print(f"Squeezed data shape: {data.shape}, Squeezed target shape: {target.shape}")
                    
                    # Store data for relabeling
                    data_list.append(data)
                    
                    # Rate Model predictions and accuracy
                    # 모델에 데이터 전달
                    #print(data.shape)
                    rate_input = spikegen.rate(data, num_steps=args.steps)
                    #print(rate_input.shape)
                    #rate_input = rate_input.view(args.steps, rate_input.size(0), rate_input.size(1), rate_input.size(2), rate_input.size(3))
                    rate_output = rate_model(rate_input, 'rate', args.steps, args.enable_snn_loss_function)
                    if args.enable_snn_loss_function:
                        #rate_predicted = rate_output.argmax(dim=1)
                        rate_predicted = rate_output.sum(dim=0).argmax(dim=1)
                    else:
                        _, rate_predicted = torch.max(rate_output.data, 1)
                    rate_model_correct += (rate_predicted == target).sum().item()
                    
                    # Model B predictions and accuracy
                    latency_input = spikegen.latency(data, tau=0.1, num_steps=args.steps)
                    #latency_input = latency_input.view(args.steps, rate_input.size(0), rate_input.size(1), rate_input.size(2), rate_input.size(3))
                    latency_output = latency_model(latency_input, 'latency', args.steps, args.enable_snn_loss_function)
                    if args.enable_snn_loss_function:
                        #latency_predicted = latency_output.argmax(dim=1)
                        latency_predicted = latency_output.sum(dim=0).argmax(dim=1)
                    else:
                        _, latency_predicted = torch.max(latency_output.data, 1)
                    latency_model_correct += (latency_predicted == target).sum().item()
                    
                    total += target.size(0)
            
            # Determine which model has higher accuracy for this dataset
            if rate_model_correct <= total*0.1 and latency_model_correct <= total*0.1:
                print_verbose(args.verbose, f"hard noise (intensity={intensity}) at rate: {rate_model_correct} | latency: {latency_model_correct}")
                # Hard Noise, no append
                if args.verbose:
                    # What is Hard Noise?
                    hard_noise = data_list[-1]
                    visualize_sample_from_list(pilot=False, list=hard_noise, path=f"{paths.image_addable}_hard_noise_{hard_noise_idx}_{intensity}.png")
                    hard_noise_idx += 1
                data_list = data_list[:-1]
            elif rate_model_correct >= latency_model_correct * 1.1:
                # Only if over 10% difference exist
                if append_only == 1 or rate_model_correct <= total*0.1:
                    data_list = data_list[:-1]
                else:
                    if append_only != -1:
                        #print(f"rate_model (intensity={intensity}) at rate: {rate_model_correct} | latency: {latency_model_correct}")
                        if args.verbose:
                            visualize_sample_from_list(pilot=False, list=data_list[-1], path=f"{paths.image_addable}_rate_{hard_noise_idx}_{intensity}.png")
                            hard_noise_idx += 1
                    # Use Model A as the label (0)
                    total_label_count[0] += 1
                    model_label = torch.zeros(total, dtype=torch.long).to(device)  # Model A is represented as '0'
                    model_label_list.append(model_label)
            else:
                if append_only == 0 or latency_model_correct <= total*0.1:
                    data_list = data_list[:-1]
                else:
                    if append_only != -1:
                        #print(f"latency_model (intensity={intensity}) at rate: {rate_model_correct} | latency: {latency_model_correct}")
                        if args.verbose:
                            visualize_sample_from_list(pilot=False, list=data_list[-1], path=f"{paths.image_addable}_latency_{hard_noise_idx}_{intensity}.png")
                            hard_noise_idx += 1
                    # Use Model B as the label (1)
                    total_label_count[1] += 1
                    model_label = torch.ones(total, dtype=torch.long).to(device)  # Model B is represented as '1'
                    model_label_list.append(model_label)


        # 한 라벨이 통틀어 하나도 생성되지 않았을 때
        if total_label_count[0]*total_label_count[1]==0:
            if total_label_count[1]==0:
                append_only=1
            else:
                append_only=0
        # 두 라벨이 균형있게 생성되었을 때
        elif (total_label_count[0]/total_label_count[1] >= 0.9) and (total_label_count[1]/total_label_count[0] >= 0.9):
            # Create a new dataset from data and the model label
            print_verbose(args.verbose, f"total label count : rate {total_label_count[0]}, latency {total_label_count[1]}")
            print(f"{len(data_list)} & {len(model_label_list)}")
            relabeled_dataset = torch.utils.data.TensorDataset(
                torch.cat(data_list),
                torch.cat(model_label_list)
            )
            print_verbose(args.verbose, f"total label count : rate {total_label_count[0]}, latency {total_label_count[1]}")
            break
        # 불균형하게 생성되었을 때
        elif (total_label_count[0]/total_label_count[1] >= 0.9):
            append_only=1
        elif (total_label_count[1]/total_label_count[0] >= 0.9):
            append_only=0
        modified_dataset_list = [RandomlyNoisedDataset(batch, args.identical_intensity, device) for batch in ref_splited_loader_list]
        modified_loader_list = [DataLoader(dataset=batch, batch_size=args.batch_size, shuffle=True) for batch in modified_dataset_list]
        loader_list = modified_loader_list
        
    print(f"{total_label_count[0]}, {total_label_count[1]}")
    return relabeled_dataset

'''
END of synthesization
'''
"""
import torch.multiprocessing as mp
#import multiprocessing

def train_rate_model(args, paths, device, snn_train_loader, snn_test_loader, shared_dict):
    try:
        rate_model = snn_train_pipeline(args, paths, device, 'rate', snn_train_loader, snn_test_loader)
        shared_dict['rate_model'] = rate_model  # 학습된 모델을 공유 딕셔너리에 추가
    except Exception as e:
        print(f"Error in train_rate_model: {e}")

def train_latency_model(args, paths, device, snn_train_loader, snn_test_loader, shared_dict):
    try:
        latency_model = snn_train_pipeline(args, paths, device, 'latency', snn_train_loader, snn_test_loader)
        shared_dict['latency_model'] = latency_model  # 학습된 모델을 공유 딕셔너리에 추가
    except Exception as e:
        print(f"Error in train_latency_model: {e}")
"""
class Exp24RensArgumentParser(RensArgumentParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], conflict_handler='resolve')

        # MOD
        self.parser.add_argument('-d', '--dataset_type', type=str, default='mnist', choices=['mnist','cifar10','cifar100'], help="Type of a dataset to use. (Default: mnist)")
        self.parser.add_argument('--training_dataset_size', help=argparse.SUPPRESS) 
        self.parser.add_argument('--test_dataset_size', help=argparse.SUPPRESS)
        #self.parser.add_argument('--training_dataset_ratio', type=restricted_float, default=1.0, help="Set a specific ratio of training dataset. (Default: 1.0)") 
        #self.parser.add_argument('--test_dataset_ratio', type=restricted_float, default=1.0, help="Set a specific ratio of test dataset. (Default: 1.0)")
        self.parser.add_argument('-m', '--model', help=argparse.SUPPRESS)
        self.parser.add_argument('-p', '--pretrained', help=argparse.SUPPRESS)
        self.parser.add_argument('-b', '--batch_size', type=int, default=64, help="Size of a batch. (Default: 64)")
        #self.parser.add_argument('-e', '--epoch', type=int, default=100, help="Size of an epoch. (Default: 100)")
        self.parser.add_argument('-o', '--optimizer', help=argparse.SUPPRESS)
        #self.parser.add_argument('--early_stopping', action='store_true', default=False, help="Applying Early Stopping method. (Default: False)")
        #self.parser.add_argument('--lr_scheduler', action='store_true', default=False, help="Applying LR Scheduling method. (Default: False)")
        self.parser.add_argument('--enable_training', help=argparse.SUPPRESS)
        self.parser.add_argument('--enable_validation', help=argparse.SUPPRESS)
        self.parser.add_argument('--enable_test', help=argparse.SUPPRESS)
        self.parser.add_argument('--write_to', help=argparse.SUPPRESS)
        #self.parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose mode. (Default: False)")
        #self.parser.add_argument('--username', type=str, default=None, help=f"Name of executor. (Default: None)")
        #self.parser.add_argument('--notes', type=str, default=None)
        self.parser.add_argument('--prefix', help=argparse.SUPPRESS)
        self.parser.add_argument('--suffix', help=argparse.SUPPRESS)
        
        # ADD
        self.parser.add_argument('--steps', type=int, default=10, help="Number of steps in SNN spike encoding. Must be greater than 1. (Default: 10)")
        self.parser.add_argument('--single_gpu', type=int, default=None, help="Enable singleGPU mode with GPU index. Disable to use parallel GPU or CPU(when GPU is unavailable). (Default: None)")
        self.parser.add_argument('--enable_snn_loss_function', action='store_true', default=False, help="Enable SNN model's SNN loss function use")
        self.parser.add_argument('--path_dataset', type=str, default=PATH_DATASET)
        self.parser.add_argument('--path_result_image', type=str, default=PATH_RESULT_IMAGE)
        self.parser.add_argument('--path_result_model', type=str, default=PATH_RESULT_MODEL)
        self.parser.add_argument('--path_result_loss', type=str, default=PATH_RESULT_LOSS)
        self.parser.add_argument('--path_result_accuracy', type=str, default=PATH_RESULT_ACCURACY)
        self.parser.add_argument('--path_result_meta', type=str, default=PATH_RESULT_META)
        self.parser.add_argument('--pretrained_rate', type=str, default=None, help="Pretrained rate encoding weights path (Default: None)")
        self.parser.add_argument('--pretrained_latency', type=str, default=None, help="Pretrained latency encoding weights path (Default: None)")
        self.parser.add_argument('--pretrained_cnn', type=str, default=None, help="Pretrained cnn weights path (Default: None)")
        self.parser.add_argument('--identical_intensity', action='store_true', default=False, help="Apply identical intensity at noise add. (Default: False)")
        

def print_verbose(verbose, sentence):
    if verbose:
        print(sentence)
        

def main(args, paths):
    """
    mp.set_start_method('spawn')
    manager = mp.Manager()
    shared_dict = manager.dict()
    """
    # Main implementation

    # Setup hyperparameters
    dtype = torch.float
    device = (
        torch.device(f"cuda:{args.single_gpu}") if (torch.cuda.is_available() and args.single_gpu != None) else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    
    # Data preprocessing
    transform = transforms.Compose([transforms.ToTensor()])
    
    if ( args.dataset_type == 'mnist' ):
        ref_train_dataset = datasets.MNIST(root=paths.dataset,
                                       train=True,
                                       transform=transform,
                                       download=True)
        ref_test_dataset = datasets.MNIST(root=paths.dataset,
                                       train=False,
                                       transform=transform,
                                       download=True)
    elif( args.dataset_type == 'cifar10' ):
        ref_train_dataset = datasets.CIFAR10(root=paths.dataset,
                                         train=True,
                                         transform=transform,
                                         download=True)
        ref_test_dataset = datasets.CIFAR10(root=paths.dataset,
                                        train=False,
                                        transform=transform,
                                        download=True)
                                      
    elif( args.dataset_type == 'cifar100' ):
        ref_train_dataset = datasets.CIFAR100(root=paths.dataset,
                                         train=True,
                                         transform=transform,
                                         download=True)
        ref_test_dataset = datasets.CIFAR100(root=paths.dataset,
                                        train=False,
                                        transform=transform,
                                        download=True)

    # Modify proportion of the dataset
    # On train dataset
    if args.pretrained_rate is not None and args.pretrained_latency is not None and args.pretrained_cnn is not None:
        print_verbose(args.verbose, "No need any train dataset. Move right into evaluation.")
    else:
        print_verbose(args.verbose, "Train dataset is on way")
        train_dataset = get_single_subset_by_ratio(ref_train_dataset, args.training_dataset_ratio)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        if args.pretrained_rate is not None and args.pretrained_latency is not None and not args.verbose:
            print_verbose(args.verbose, "No need SNN train dataset. Move to SNN run dataset: noised.")
        else:
            snn_train_dataset = train_dataset
            snn_train_loader = DataLoader(dataset=snn_train_dataset, batch_size=args.batch_size, shuffle=True)
        snn_splited_train_dataset_list = get_multiple_subsets_by_size(dataset=train_dataset, subset_size=args.batch_size, drop_last=True)
        snn_splited_train_loader_list = [DataLoader(dataset=batch, batch_size=args.batch_size, shuffle=False) for batch in snn_splited_train_dataset_list]
        snn_modified_train_dataset_list = [RandomlyNoisedDataset(batch, args.identical_intensity, device) for batch in snn_splited_train_loader_list]
        snn_modified_train_loader_list = [DataLoader(dataset=batch, batch_size=args.batch_size, shuffle=True) for batch in snn_modified_train_dataset_list]

        print_verbose(args.verbose, "Train dataset is ready")
    # On test dataset
    print_verbose(args.verbose, "Test dataset is on way")
    test_dataset = get_single_subset_by_ratio(ref_test_dataset, args.test_dataset_ratio)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    if args.pretrained_rate is not None and args.pretrained_latency is not None and not args.verbose:
        print_verbose(args.verbose, "No need SNN test dataset. Move to SNN run dataset: noised.")
    else:
        snn_test_dataset = test_dataset
        snn_test_loader = DataLoader(dataset=snn_test_dataset, batch_size=args.batch_size, shuffle=False)
    '''
    snn_modified_test_dataset = RandomlyNoisedDataset(snn_test_loader, False)
    snn_modified_test_loader = DataLoader(dataset=snn_modified_test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    '''
    snn_splited_test_dataset_list = get_multiple_subsets_by_size(dataset=test_dataset, subset_size=args.batch_size, drop_last=True)
    snn_splited_test_loader_list = [DataLoader(dataset=batch, batch_size=args.batch_size, shuffle=False) for batch in snn_splited_test_dataset_list]
    snn_modified_test_dataset_list = [RandomlyNoisedDataset(batch, args.identical_intensity, device) for batch in snn_splited_test_loader_list]
    snn_modified_test_loader_list = [DataLoader(dataset=batch, batch_size=args.batch_size, shuffle=True) for batch in snn_modified_test_dataset_list]
    print_verbose(args.verbose, "Test dataset is ready")

    # Sanity check
    visualize_sample_from_loader_list(pilot=False, loader_list=snn_modified_test_loader_list, path=paths.image)


    ## 1. Train SNN
    print_verbose(args.verbose, "1. Train SNN is on way")
    '''
    # 프로세스 생성
    process1 = mp.Process(target=train_rate_model, args=(args, paths, device, snn_train_loader, snn_test_loader, shared_dict))    # rate_model
    process2 = mp.Process(target=train_latency_model, args=(args, paths, device, snn_train_loader, snn_test_loader, shared_dict)) # latency_model

    # 프로세스 시작
    process1.start()
    process2.start()

    # 두 프로세스가 완료될 때까지 기다림
    process1.join()
    process2.join()

    # 프로세스가 완료된 후에 상태 확인
    if process1.exitcode != 0:
        print(f"Process 1 exited with code {process1.exitcode}")
    if process2.exitcode != 0:
        print(f"Process 2 exited with code {process2.exitcode}")
    
    if 'rate_model' not in shared_dict or 'latency_model' not in shared_dict:
        raise ValueError("One or both models were not trained properly.")
    
    # 메인 프로세스에서 학습된 모델 사용
    rate_model = shared_dict['rate_model']
    latency_model = shared_dict['latency_model']
    '''
    if args.pretrained_rate is not None:
        rate_model = CNV_SNN(args.enable_snn_loss_function, args.steps)
        rate_model.load_state_dict(torch.load(args.pretrained_rate))
        rate_model = rate_model.to(device)
        if args.verbose:
            print("Rate encoding model is given, no train needed. Load weight and move onto next step.")
            print("###Sanity Check###")
            snn_evaluation(args, paths, 'rate', rate_model, snn_test_loader, device)
    else:
        rate_model = snn_train_pipeline(args, paths, device, 'rate', snn_train_loader, snn_test_loader)
    if args.pretrained_latency is not None:
        latency_model = CNV_SNN(args.enable_snn_loss_function, args.steps)
        latency_model.load_state_dict(torch.load(args.pretrained_latency))
        latency_model = latency_model.to(device)
        if args.verbose:
            print("Latency encoding model is given, no train needed. Load weight and move onto next step.")
            print("###Sanity Check###")
            snn_evaluation(args, paths, 'latency', latency_model, snn_test_loader, device)
    else:
        latency_model = snn_train_pipeline(args, paths, device, 'latency', snn_train_loader, snn_test_loader)

    print_verbose(args.verbose, "END")

    ## 2. Run SNN
    print_verbose(args.verbose, "2. Run SNN is on way")
    if args.pretrained_cnn is not None:
        print_verbose(args.verbose, "No need CNN train dataset. Move to CNN test dataset.")
    else:
        labeled_train_dataset = relabel_based_on_evaluation(args, snn_splited_train_loader_list, snn_modified_train_loader_list, rate_model, latency_model, device)
        labeled_train_loader = DataLoader(dataset=labeled_train_dataset, batch_size=args.batch_size, shuffle=True)
    labeled_test_dataset = relabel_based_on_evaluation(args, snn_splited_test_loader_list, snn_modified_test_loader_list, rate_model, latency_model, device)
    labeled_test_loader = DataLoader(dataset=labeled_test_dataset, batch_size=args.batch_size, shuffle=True)
    print_verbose(args.verbose, "END")

    ## 3. Train CNN
    print_verbose(args.verbose, "3. Train CNN is on way")
    if args.pretrained_cnn is not None:
        print_verbose(args.verbose, "No need SNN train dataset. Move to CNN train dataset: noised.")
        cnn_model = BNC_CNN_MNIST()
        cnn_model.load_state_dict(torch.load(args.pretrained_cnn))
        cnn_model = cnn_model.to(device)
        if args.verbose:
            print("CNN model is given, no train needed. Load weight and move onto next step.")
            print("###Sanity Check###")
            cnn_evaluation(args, paths, cnn_model, labeled_test_loader, device)
    else:
        cnn_model = cnn_train_pipeline(args, paths, device, labeled_train_loader, labeled_test_loader)
    print_verbose(args.verbose, "END")
   
if __name__ == "__main__":
    parser = Exp24RensArgumentParser()
    args = parser.get_argument_parser()

    # Sanity check; directory existence
    ensure_directory(args.path_dataset)
    ensure_directory(args.path_result_image)
    ensure_directory(args.path_result_model)
    ensure_directory(args.path_result_loss)
    ensure_directory(args.path_result_accuracy)
    ensure_directory(args.path_result_meta)

    # Write meta data
    current_time = get_current_time_str()
    xid = get_next_xid(args.path_result_meta)
    lines = parser.get_argument_info('string')

    paths = SimpleNamespace(
        dataset = args.path_dataset,
        meta = f'{args.path_result_meta}/{xid:03d}_{EXP_NO}_meta_{args.dataset_type}.txt',
        image = f'{args.path_result_image}/{xid:03d}_{EXP_NO}_sample_{args.dataset_type}.png',
        rate = SimpleNamespace(
            model = f'{args.path_result_model}/{xid:03d}_{EXP_NO}_model_{args.dataset_type}_rate.weights',
            loss = f'{args.path_result_loss}/{xid:03d}_{EXP_NO}_loss_{args.dataset_type}_rate.png',
            accuracy = f'{args.path_result_accuracy}/{xid:03d}_{EXP_NO}_accuarcy_{args.dataset_type}_rate.png',
        ),
        latency = SimpleNamespace(
            model = f'{args.path_result_model}/{xid:03d}_{EXP_NO}_model_{args.dataset_type}_latency.weights',
            loss = f'{args.path_result_loss}/{xid:03d}_{EXP_NO}_loss_{args.dataset_type}_latency.png',
            accuracy = f'{args.path_result_accuracy}/{xid:03d}_{EXP_NO}_accuarcy_{args.dataset_type}_latency.png',
        ),
        cnn = SimpleNamespace(
            model = f'{args.path_result_model}/{xid:03d}_{EXP_NO}_model_{args.dataset_type}_cnn.weights',
            loss = f'{args.path_result_loss}/{xid:03d}_{EXP_NO}_loss_{args.dataset_type}_cnn.png',
            accuracy = f'{args.path_result_accuracy}/{xid:03d}_{EXP_NO}_accuarcy_{args.dataset_type}_cnn.png',
        ),
        accuracy_csv = args.path_result_accuracy + f'{EXP_NO}_accuracy.csv',
        accuracy_summary_csv = args.path_result_accuracy + f'{EXP_NO}_accuracy_summary.csv',
        image_addable = f'{args.path_result_image}/{xid:03d}_{EXP_NO}'
    )

    # Sanity check: Print meta data
    if args.verbose:
        print("## Meta data ##")
        for line in lines:
            print(line)
        print("#####")
    
    with open(paths.meta, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    
    # Execution
    try:
        start_time = time.time()
        main(args, paths)
        write_metadata_status(paths.meta, 'SUCCESS')
        print("SUCCESS")
    except KeyboardInterrupt:
        write_metadata_status(paths.meta, 'HALTED')
        print("HALTED")
    except Exception as e:
        _, _, tb = sys.exc_info()
        trace = traceback.format_tb(tb)
        
        write_metadata_status(paths.meta, f'FAILED({e})')
        with open(paths.meta, 'a') as file:
            file.writelines(trace)
            
        print(f"FAILED({type(e).__name__}: {e})")
        print(''.join(trace))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(paths.meta, 'a') as file:
        file.write(f'\nTotal elapsed time: {elapsed_time:.6f} s')