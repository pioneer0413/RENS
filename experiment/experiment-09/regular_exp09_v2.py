# Experiment no.09 [regular]

#
# Paths
#
from pathlib import Path
exp_no = 'exp09' # NOTICE: Change at each experiment!
directory = current_file_path = Path(__file__).resolve()
path_root = str(directory.parent.parent.parent) + '/'
path_dataset = path_root + 'dataset/'
path_result = path_root + 'result/'
path_result_root = path_result + exp_no + '/'
path_result_image = path_result_root + 'image/'
path_result_model = path_result_root + 'model/'
path_result_loss = path_result_root + 'loss/'
path_result_metrics = path_result_root + 'metrics/'
path_result_meta = path_result_root + 'meta/'
path_utility = path_root + 'utility/'

#
# Imports
#
import sys
sys.path.append(path_root)
sys.path.append(path_dataset)
sys.path.append(path_result)
sys.path.append(path_utility)

# PyTorch family
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR #학습률 스케줄링
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import resnet50, resnet101

# SnnTorch family
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt 

# Utilities
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

from utility.common import *
from utility.preprocess import *
from utility.synthesize import *
from utility.visualize import *

from model.LearningUtils import *

import traceback

#
# Specific Functions and Classes
#
# TODO: 유틸리티에 추가 >> (v1.0.2)
def calculate_metrics(labels, predictions, average='weighted'):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average=average)
    recall = recall_score(labels, predictions, average=average)
    f1 = f1_score(labels, predictions, average=average)
    return (accuracy, precision, recall, f1)

# TODO: 유틸리티에 추가 >> (v1.0.2)
def normalize(tensor):
    vmax, vmin = tensor.max(), tensor.min()
    normalized_tensor = (tensor-vmin)/(vmax-vmin)
    return normalized_tensor

# TODO: 유틸리티에 추가 >> (v1.0.2)
def generate_noisy_batch(inputs, noise_intensity, device):
    noise = torch.randn(size=inputs.shape) * noise_intensity
    noise = noise.to(device)
    noisy_inputs = inputs + noise

    # 정규화(Optional)
    # noisy_inputs = normalize(noisy_inputs)
    
    return noisy_inputs

def forward_pass(net, num_steps, data, batch_size):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data, batch_size)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    ret1 = torch.stack(spk_rec)
    ret2 = torch.stack(mem_rec)
    
    return ret1, ret2

class Net(nn.Module):
    def __init__(self, beta, spike_grad):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64*5*5, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, batch_size):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2) # [256, 64, 5, 5]

        cur3 = self.fc1(spk2.view(batch_size, -1)) # [256, 10]
        spk3, mem3 = self.lif3(cur3, mem3) # [256, 10]

        return spk3, mem3

#
# Main Implementation
#
def main(args):
    # 트랜스폼 준비
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    ###
    
    # 데이터셋 종류에 따라 다르게 준비
    if( args.dataset_type == 'cifar10' ):
        train_dataset = datasets.CIFAR10(root=path_dataset, train=True, transform=transform, download=False)
        test_dataset = datasets.CIFAR10(root=path_dataset, train=False, transform=transform, download=False)
    else:
        train_dataset = datasets.CIFAR100(root=path_dataset, train=True, transform=transform, download=False)
        test_dataset = datasets.CIFAR100(root=path_dataset, train=False, transform=transform, download=False)
    ###
    
    # 데이터셋 크기 조절
    if( args.train_dataset_ratio < 1.0 ):
        train_dataset = get_single_subset_by_ratio(train_dataset, ratio=args.train_dataset_ratio)
    if( args.test_dataset_ratio < 1.0 ):
        test_dataset = get_single_subset_by_ratio(test_dataset, ratio=args.test_dataset_ratio)
    ###
    
    # 훈련 데이터셋과 검증 데이터셋으로 분할
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    ###

    # 데이터로더 준비
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    ###
    
    # 디바이스 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ###
    
    # 모델 생성
    if( args.model == 'resnet50' ):
        model = resnet50(weights=None, num_classes=10) if args.dataset_type=='cifar10' else resnet50(weights=None, num_classes=100)
        criterion = nn.CrossEntropyLoss()
    elif( args.model == 'resnet101' ):
        model = resnet101(weights=None, num_classes=10) if args.dataset_type=='cifar10' else resnet50(weights=None, num_classes=100)
        criterion = nn.CrossEntropyLoss()
    elif( args.model == 'snn'):
        num_steps = 50
        beta = 0.9
        model = Net(beta, surrogate.fast_sigmoid(slope=25))
        criterion = SF.ce_rate_loss()
    if( args.pretrained is not None ):
        model.load_state_dict(torch.load(args.pretrained))
    model = model.to(device)
    ###

    ### 옵티마이저 설정
    if( args.optimizer == 'adam' ):
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    elif( args.optimizer == 'sgd' ):
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    ###

    ### 에폭 수 설정
    num_epochs = args.epoch
    ###
        
    # 학습 기록용 변수 설정
    epoch_loss_rec = [] # 훈련 로스 기록
    val_loss_rec = [] # 검증 로스 기록
    max_epoch_loss = math.inf # 모델 저장 시 직전 로스
    ###

    # LR 스케줄러 생성
    scheduler = CosineAnnealingLR(optimizer, T_max=200)
    ###

    # 조기 종료 설정
    early_stopping = EarlyStopping(patience=10, min_delta=1e-6)
    ###

    # 훈련 루프
    if( args.pretrained is None ):
        for epoch in range(num_epochs):
            running_loss = 0.0

            model.train()
            
            # 훈련 단계
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 33.3% 확률로 무작위 강도 잡음 주입
                if( args.noise_training is True ):
                    trigger = torch.rand(1).item()
                    if( trigger >= 0.66 ):
                        inputs = generate_noisy_batch(inputs, torch.rand(1).item(), device)
                
                optimizer.zero_grad()

                # 배치 순전파
                if args.model == 'snn':
                    outputs, _ = forward_pass(model, num_steps, inputs, args.batch_size)
                else:
                    outputs = model(inputs)
                ###
                
                # 손실 계산
                loss = criterion(outputs, labels)
                ###

                # 역전파
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                ###
            ### 에폭 종료
            
            # 검증 단계
            model.eval()
            val_loss = 0.0
            acc = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # 배치 순전파
                    if args.model == 'snn':
                        outputs, _ = forward_pass(model, num_steps, inputs, args.batch_size)
                    else:
                        outputs = model(inputs)
                    ###
                    
                    # 손실 계산
                    loss = criterion(outputs, labels)
                    ###

                    # 손실 저장
                    val_loss += loss.item()
                    ###

                    # 검증 정확도 재료 수집
                    """
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    """
                    acc += SF.accuracy_rate(outputs, labels) * outputs.size(1)
                    total += outputs.size(1)
                    ###

            # 검증 정확도 산출
            #val_accuracy = 100 * correct / total
            val_accuracy = 100 * ( acc / total )
            
            ### 검증 종료

            # 에폭 훈련 결과 출력
            print(f'\nEpoch: [{epoch+1}/{num_epochs}] ###############################################')
            print(f'Epoch Loss: {running_loss / len(train_loader.dataset):.4f} | Validation Loss: {val_loss / len(valid_loader.dataset):.4f} |  Validation Accuracy: {val_accuracy:.2f}%')
            ###

            # 에폭 훈련 결과 기록
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_loss_rec.append(epoch_loss)
            val_loss_rec.append(val_loss)
            ###
            
            # 손실 개선이 있었는가?
            if( epoch_loss < max_epoch_loss ):
                if( args.verbose ):
                    print(f'Model saved: [Last] {max_epoch_loss:.4f} >> {epoch_loss:.4f} [Current]')
                best_model_state = model.state_dict()
                torch.save(best_model_state, model_file_path)
                best_epoch_idx = epoch + 1
                max_epoch_loss = epoch_loss
                
            # 손실 변화에 따른 학습률 변경
            if args.lr_scheduler:
                scheduler.step(epoch_loss)
                if args.verbose:
                    print(f"Learning rate after epoch: {scheduler.get_last_lr()[0]}")
                
            # 손실 개선이 없어 종료해야 하는가?
            if args.early_stopping:
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    if args.verbose:
                        print(f"Early stopping with: (epoch loss:{epoch_loss:.4f})")
                    break
        ### 전체 훈련 루프 종료

        # 시험용 모델 획득
        if args.verbose:
            print(f'\nLoad model state at best epoch loss [{best_epoch_idx}]\n')
        model.load_state_dict(best_model_state)
        ###

        # 훈련 진행에 따른 손실 시각화
        if 'loss' in result_set:
            visualize_epoch_loss(pilot=False, epoch_loss=epoch_loss_rec, file_path=loss_file_path)
        ###
        
    # 시험
    model.eval()
    #all_labels = []
    #all_predictions = []

    for i in range(20): # 재활용 시 반드시 수정할 것
        all_labels_000 = []
        all_predictions_000 = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
    
                # 잡음 주입
                if( args.noise_test ):
                    inputs = generate_noisy_batch(inputs, args.noise_intensity, device)
    
                # 순전파
                if args.model == 'snn':
                    outputs, _ = forward_pass(model, num_steps, inputs, args.batch_size)
    
                    # 메트릭 수집
                    spk_rec_avg = outputs.mean(dim=0)
                    predictions = spk_rec_avg.argmax(dim=1)
                    all_labels_000.extend(labels.detach().cpu().numpy())
                    all_predictions_000.extend(predictions.detach().cpu().numpy())
                    ###
                else:
                    outputs = model(inputs)
    
                    # 메트릭 수집
                    predicted_value, predicted_class = torch.max(outputs, dim=1)
                    all_labels.extend(labels.detach().cpu().numpy())
                    all_predictions.extend(predicted_class.detach().cpu().numpy())
                    ###
                ###
    
        # SNN 잡음 테스트
        all_labels_025 = []
        all_predictions_025 = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
    
                # 잡음 주입
                inputs = generate_noisy_batch(inputs, 0.25, device)
    
                # 순전파
                if args.model == 'snn':
                    outputs, _ = forward_pass(model, num_steps, inputs, args.batch_size)
    
                    # 메트릭 수집
                    spk_rec_avg = outputs.mean(dim=0)
                    predictions = spk_rec_avg.argmax(dim=1)
                    all_labels_025.extend(labels.detach().cpu().numpy())
                    all_predictions_025.extend(predictions.detach().cpu().numpy())
                    ###
                else:
                    outputs = model(inputs)
    
                    # 메트릭 수집
                    predicted_value, predicted_class = torch.max(outputs, dim=1)
                    all_labels.extend(labels.detach().cpu().numpy())
                    all_predictions.extend(predicted_class.detach().cpu().numpy())
                    ###
                ###
    
        all_labels_050 = []
        all_predictions_050 = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
    
                # 잡음 주입
                inputs = generate_noisy_batch(inputs, 0.5, device)
    
                # 순전파
                if args.model == 'snn':
                    outputs, _ = forward_pass(model, num_steps, inputs, args.batch_size)
    
                    # 메트릭 수집
                    spk_rec_avg = outputs.mean(dim=0)
                    predictions = spk_rec_avg.argmax(dim=1)
                    all_labels_050.extend(labels.detach().cpu().numpy())
                    all_predictions_050.extend(predictions.detach().cpu().numpy())
                    ###
                else:
                    outputs = model(inputs)
    
                    # 메트릭 수집
                    predicted_value, predicted_class = torch.max(outputs, dim=1)
                    all_labels.extend(labels.detach().cpu().numpy())
                    all_predictions.extend(predicted_class.detach().cpu().numpy())
                    ###
                ###
    
        all_labels_075 = []
        all_predictions_075 = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
    
                # 잡음 주입
                inputs = generate_noisy_batch(inputs, 0.75, device)
    
                # 순전파
                if args.model == 'snn':
                    outputs, _ = forward_pass(model, num_steps, inputs, args.batch_size)
    
                    # 메트릭 수집
                    spk_rec_avg = outputs.mean(dim=0)
                    predictions = spk_rec_avg.argmax(dim=1)
                    all_labels_075.extend(labels.detach().cpu().numpy())
                    all_predictions_075.extend(predictions.detach().cpu().numpy())
                    ###
                else:
                    outputs = model(inputs)
    
                    # 메트릭 수집
                    predicted_value, predicted_class = torch.max(outputs, dim=1)
                    all_labels.extend(labels.detach().cpu().numpy())
                    all_predictions.extend(predicted_class.detach().cpu().numpy())
                    ###
                ###
                
        all_labels_100 = []
        all_predictions_100 = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
    
                # 잡음 주입
                inputs = generate_noisy_batch(inputs, 1.0, device)
    
                # 순전파
                if args.model == 'snn':
                    outputs, _ = forward_pass(model, num_steps, inputs, args.batch_size)
    
                    # 메트릭 수집
                    spk_rec_avg = outputs.mean(dim=0)
                    predictions = spk_rec_avg.argmax(dim=1)
                    all_labels_100.extend(labels.detach().cpu().numpy())
                    all_predictions_100.extend(predictions.detach().cpu().numpy())
                    ###
                else:
                    outputs = model(inputs)
    
                    # 메트릭 수집
                    predicted_value, predicted_class = torch.max(outputs, dim=1)
                    all_labels.extend(labels.detach().cpu().numpy())
                    all_predictions.extend(predicted_class.detach().cpu().numpy())
                    ###
                ###
            
        ### 시험 종료
    
        # 시험 결과 지표 획득 및 출력
        accuracy_000, precision_000, recall_000, f1_000 = calculate_metrics(all_labels_000, all_predictions_000)
        accuracy_025, precision_025, recall_025, f1_025 = calculate_metrics(all_labels_025, all_predictions_025)
        accuracy_050, precision_050, recall_050, f1_050 = calculate_metrics(all_labels_050, all_predictions_050)
        accuracy_075, precision_075, recall_075, f1_075 = calculate_metrics(all_labels_075, all_predictions_075)
        accuracy_100, precision_100, recall_100, f1_100 = calculate_metrics(all_labels_100, all_predictions_100)

        print(f'[{i+1}/20]')
        print(f'[000] Accuracy: {accuracy_000:.2f} | Precision: {precision_000:.2f} | Recall: {recall_000:.2f} | F1-Score: {f1_000:.2f}')
        print(f'[025] Accuracy: {accuracy_025:.2f} | Precision: {precision_025:.2f} | Recall: {recall_025:.2f} | F1-Score: {f1_025:.2f}')
        print(f'[050] Accuracy: {accuracy_050:.2f} | Precision: {precision_050:.2f} | Recall: {recall_050:.2f} | F1-Score: {f1_050:.2f}')
        print(f'[075] Accuracy: {accuracy_075:.2f} | Precision: {precision_075:.2f} | Recall: {recall_075:.2f} | F1-Score: {f1_075:.2f}')
        print(f'[100] Accuracy: {accuracy_100:.2f} | Precision: {precision_100:.2f} | Recall: {recall_100:.2f} | F1-Score: {f1_100:.2f}\n')
        ###
    
        # 시험 결과 내보내기
        metrics_record_000 = {'accuracy': accuracy_000, 'precision': precision_000, 'recall': recall_000, 'f1-score': f1_000}
        metrics_record_025 = {'accuracy': accuracy_025, 'precision': precision_025, 'recall': recall_025, 'f1-score': f1_025}
        metrics_record_050 = {'accuracy': accuracy_050, 'precision': precision_050, 'recall': recall_050, 'f1-score': f1_050}
        metrics_record_075 = {'accuracy': accuracy_075, 'precision': precision_075, 'recall': recall_075, 'f1-score': f1_075}
        metrics_record_100 = {'accuracy': accuracy_100, 'precision': precision_100, 'recall': recall_100, 'f1-score': f1_100}
        
        if 'metrics' in result_set:
            save_record_to_csv(f'/home/hwkang/jupyter/root/result/exp09/metrics/SNN_000.csv', metrics_record_000)
            save_record_to_csv(f'/home/hwkang/jupyter/root/result/exp09/metrics/SNN_025.csv', metrics_record_025)
            save_record_to_csv(f'/home/hwkang/jupyter/root/result/exp09/metrics/SNN_050.csv', metrics_record_050)
            save_record_to_csv(f'/home/hwkang/jupyter/root/result/exp09/metrics/SNN_075.csv', metrics_record_075)
            save_record_to_csv(f'/home/hwkang/jupyter/root/result/exp09/metrics/SNN_100.csv', metrics_record_100)
        ###
    
    # 시험 결과 시각화
    if 'confusion_matrix' in result_set:
        visualize_confusion_matrix(pilot=False,
                                   all_labels=all_labels_000,
                                   all_predictions=all_predictions_000,
                                   num_label=10 if args.dataset_type=='cifar10' else 100,
                                   accuracy=accuracy_000*100,
                                   noise_type=args.noise_type,
                                   file_path='/home/hwkang/jupyter/root/result/exp09/metrics/SNN_000.png')
        visualize_confusion_matrix(pilot=False,
                                   all_labels=all_labels_025,
                                   all_predictions=all_predictions_025,
                                   num_label=10 if args.dataset_type=='cifar10' else 100,
                                   accuracy=accuracy_025*100,
                                   noise_type=args.noise_type,
                                   file_path='/home/hwkang/jupyter/root/result/exp09/metrics/SNN_025.png')
        visualize_confusion_matrix(pilot=False,
                                   all_labels=all_labels_050,
                                   all_predictions=all_predictions_050,
                                   num_label=10 if args.dataset_type=='cifar10' else 100,
                                   accuracy=accuracy_050*100,
                                   noise_type=args.noise_type,
                                   file_path='/home/hwkang/jupyter/root/result/exp09/metrics/SNN_050.png')
        visualize_confusion_matrix(pilot=False,
                                   all_labels=all_labels_075,
                                   all_predictions=all_predictions_075,
                                   num_label=10 if args.dataset_type=='cifar10' else 100,
                                   accuracy=accuracy_075*100,
                                   noise_type=args.noise_type,
                                   file_path='/home/hwkang/jupyter/root/result/exp09/metrics/SNN_075.png')
        visualize_confusion_matrix(pilot=False,
                                   all_labels=all_labels_100,
                                   all_predictions=all_predictions_100,
                                   num_label=10 if args.dataset_type=='cifar10' else 100,
                                   accuracy=accuracy_100*100,
                                   noise_type=args.noise_type,
                                   file_path='/home/hwkang/jupyter/root/result/exp09/metrics/SNN_100.png')
    ###
    

#
# Initialization
#
if __name__=='__main__':
    parser = argparse.ArgumentParser() # 불필요한 보일러플레이트 코드이므로 클래스를 선언한 후 상속하도록 변경 >> (v1.0.2)

    # 데이터셋
    parser.add_argument('-d', '--dataset_type', type=str, required=False, default='cifar10', choices=['cifar10','cifar100'], help="Default: cifar10")
    # 데이터셋 비율
    parser.add_argument('--train_dataset_ratio', type=restricted_float, required=False, default=1.0, help="Default: 1.0") 
    parser.add_argument('--test_dataset_ratio', type=restricted_float, required=False, default=1.0, help="Default: 1.0")
    
    # 모델
    parser.add_argument('-m', '--model', type=str, required=False, default='resnet101', choices=['resnet50', 'resnet101', 'snn'], help="Default: resnet101")

    # 옵티마이저
    parser.add_argument('-o', '--optimizer', type=str, required=False, default='adam', choices=['adam', 'sgd'], help="Default: adam")

    # 잡음
    parser.add_argument('-n', '--noise_type', type=str, required=False, default='gaussian', choices=['gaussian', 'snp', 'uniform', 'poisson', 'multi', 'none'], help="Default: gaussian")
    parser.add_argument('--noise_training', action='store_true', required=False, default=False, help="Default: False")
    parser.add_argument('--noise_test', action='store_true', required=False, default=False, help="Default: False")
    parser.add_argument('--noise_intensity', type=restricted_float, required=False, default=0.0, help="Default: 0.0")

    # 하이퍼파라미터
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=256, help="Default: 256")
    parser.add_argument('-e', '--epoch', type=int, required=False, default=100, help="Default: 100")
    parser.add_argument('-p', '--pretrained', type=str, required=False, default=None, help="Default: None")
    parser.add_argument('--early_stopping', action='store_true', default=False, help="Default: False")
    parser.add_argument('--lr_scheduler', action='store_true', default=False, help="Default: False")
    
    # 파일 저장 경로
    parser.add_argument('--write_results', nargs='*', required=False, default=['meta'], choices=['meta', 'image', 'model', 'loss', 'confusion_matrix', 'metrics'], help="Default: ['meta']")
    parser.add_argument('--output_path_meta', type=str, required=False, default=path_result_meta, help=f"Default: {path_result_meta}")
    parser.add_argument('--output_path_image', type=str, required=False, default=path_result_image, help=f"Default: {path_result_image}")
    parser.add_argument('--output_path_model', type=str, required=False, default=path_result_model, help=f"Default: {path_result_model}")
    parser.add_argument('--output_path_loss', type=str, required=False, default=path_result_loss, help=f"Default: {path_result_loss}")
    parser.add_argument('--output_path_metrics', type=str, required=False, default=path_result_metrics, help=f"Default: {path_result_metrics}")
    ### 
    
    parser.add_argument('--metrics_csv_file_path', type=str, required=False, default=None, help=f"Default: None")

    # 메타 데이터
    parser.add_argument('--username', type=str, required=False, default=None, help=f"Default: None")

    # 진행 내용 터미널에 출력
    parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose mode")
    
    # 메모용
    parser.add_argument('--notes', type=str, required=False, default=None)
    parser.add_argument('--prefix', type=str, required=False, default=None)
    parser.add_argument('--suffix', type=str, required=False, default=None)

    # 객체화
    args = parser.parse_args()

    # 디렉터리 확인 및 없을 시 생성
    ensure_directory(path_dataset)
    ensure_directory(path_result)
    ensure_directory(path_result_root)
    ensure_directory(args.output_path_meta)
    ensure_directory(args.output_path_image)
    ensure_directory(args.output_path_model)
    ensure_directory(args.output_path_loss)
    ensure_directory(args.output_path_metrics)
    ###

    current_time = get_current_time_str()
    xid = get_next_xid(args.output_path_meta)
    lines = [
        f'datetime: {current_time}',
        f'xid: {xid}',
        f'dataset_type: {args.dataset_type}',
        f'train_dataset_ratio: {args.train_dataset_ratio}',
        f'train_dataset_ratio: {args.test_dataset_ratio}',
        f'model: {args.model}',
        f'optimizer: {args.optimizer}',
        f'noise_type: {args.noise_type}',
        f'noise_training: {args.noise_training}',
        f'noise_test: {args.noise_test}',
        f'noise_intensity: {args.noise_intensity}',
        f'batch_size: {args.batch_size}',
        f'epoch: {args.epoch}',
        f'pretrained: {args.pretrained}',
        f'early_stopping: {args.early_stopping}',
        f'lr_scheduler: {args.lr_scheduler}',
        f'write_results: {args.write_results}',
        f'output_path_meta: {args.output_path_meta}',
        f'output_path_image: {args.output_path_image}',
        f'output_path_model: {args.output_path_model}',
        f'output_path_loss: {args.output_path_loss}',
        f'output_path_metrics: {args.output_path_metrics}',
        f'metrics_csv_file_path: {args.metrics_csv_file_path}',
        f'username: {args.username}',
        f'verbose: {args.verbose}',
        f'notes: {args.notes}'
    ]
    
    meta_file_path = f'{args.output_path_meta}/{xid:03d}_{exp_no}_meta_{args.prefix}{args.dataset_type}_{args.noise_intensity}_{current_time}{args.suffix}.txt'
    image_file_path = f'{args.output_path_image}/{xid:03d}_{exp_no}_sample_{args.prefix}{args.dataset_type}_{args.noise_intensity}_{current_time}{args.suffix}.png'
    model_file_path = f'{args.output_path_model}/{xid:03d}_{exp_no}_model_{args.prefix}{args.dataset_type}_{args.noise_intensity}_{current_time}{args.suffix}.weights'
    loss_file_path = f'{args.output_path_loss}/{xid:03d}_{exp_no}_loss_{args.prefix}{args.dataset_type}_{args.noise_intensity}_{current_time}{args.suffix}.png'
    
    # 혼동 행렬을 표현하는 파일명으로 변경 >> (v1.0.2)
    confusion_matrix_file_path = f'{args.output_path_metrics}/{xid:03d}_{exp_no}_confusion_matrix_{args.prefix}{args.dataset_type}_{args.noise_intensity}_{current_time}{args.suffix}.png' 
    
    # 정확도 지표 뿐만 아니라 향후 다른 지표도 함께 명시한다는 것을 나타내기 위해 metrics를 중심으로 변경 >> (v1.0.2)
    if args.metrics_csv_file_path is None:
        metrics_csv_file_path = args.output_path_metrics + f'{exp_no}_{args.prefix}{args.dataset_type}_{args.noise_intensity}_metrics{args.suffix}.csv'
    else:
        metrics_csv_file_path = args.metrics_csv_file_path
    ###

    result_set = set(args.write_results)
    
    if args.verbose:
        print("#######################################################################")
        print("########################## Meta Data ##################################")
        print("#######################################################################")
        for line in lines:
            print(line)
        print("#######################################################################")
        print("########################## Output Filenames ###########################")
        print("#######################################################################")
        print(meta_file_path)
        print(image_file_path)
        print(model_file_path)
        print(loss_file_path)
        print(confusion_matrix_file_path)
        print(metrics_csv_file_path)
        print("#######################################################################")
        print("########################## End of Meta Data ###########################")
        print("#######################################################################")

    with open(meta_file_path, 'w') as file:
      for line in lines:
          file.write(line + '\n')
    
    # Execution
    try:
        start_time = time.time()
        main(args)
        write_metadata_status(meta_file_path, 'SUCCESS')
    except KeyboardInterrupt:
        write_metadata_status(meta_file_path, 'HALTED')
    except Exception as e:
        _, _, tb = sys.exc_info()
        trace = traceback.format_tb(tb)
        write_metadata_status(meta_file_path, f'FAILED({e})')
        with open(meta_file_path, 'a') as file:
            file.writelines(trace)
        print(f"FAILED({type(e).__name__}: {e})")
        print(''.join(trace))
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(meta_file_path, 'a') as file:
            file.write(f'\nTotal elapsed time: {elapsed_time:.6f} s')
        print( f'Total elapsed time: {elapsed_time:.6f} s\n' )