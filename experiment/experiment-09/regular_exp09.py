# Experiment no.09 [regular]

#
# Constants
#
from pathlib import Path
exp_no = 'exp09' # NOTICE: Change at each experiment!
directory = current_file_path = Path(__file__).resolve()
path_root = str(directory.parent.parent.parent) + '/'

#
# Imports
#

# Fundamentals
import os
import sys
sys.path.append(path_root)
sys.path.append(os.path.join(path_root,'utility/'))
sys.path.append(os.path.join(path_root,'model/'))

# PyTorchs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR #학습률 스케줄링
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet50, resnet101

# SnnTorchs
import snntorch as snn
from snntorch import surrogate
#from snntorch import backprop # Deprecated
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
#from snntorch.spikevision import spikedata # Deprecated

# Tonics
import tonic
import tonic.transforms
from tonic import MemoryCachedDataset


# Utilities
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback

from utility.common import *
from utility.evaluation import *
from utility.parser import RensArgumentParser
from utility.preprocessing import *
from utility.statistic import *
from utility.synthesization import *
from utility.visualization import *

from model.learning_utility import EarlyStopping
from models import *

#
# Specifics
#
class InheritedRensArgumentParser(RensArgumentParser):
    def __init__(self):
        super().__init__()
        ### 부모 파서의 인자 처리 정보를 받아옴, 충돌 시 아래 추가할 것임을 명시
        self.parser = argparse.ArgumentParser(parents=[self.parser], conflict_handler='resolve')
        ##*
        self.parser.add_argument('-m', '--model', type=str, default='resnet101', choices=['cnn', 'resnet50', 'resnet101', 'snn1', 'snn2', 'snn3'], help="Type of a model to use. (Default: resnet101)")
        self.parser.add_argument('--noise_training',action='store_true',
                                 default=False)
        self.parser.add_argument('--noise_test', action='store_true',
                                 default=False)
        self.parser.add_argument('--noise_intensity', type=float, default=0.5)
        self.parser.add_argument('--output_format', type=str, default='string')
        self.parser.add_argument('--path_file_metrics', type=str, default=None)
        self.parser.add_argument('--path_file_metrics_latency', type=str, default=None)
        self.parser.add_argument('--num_steps', type=int, default=50)
        self.parser.add_argument('--beta', type=float, default=0.99)
        self.parser.add_argument('-d', '--dataset_type', type=str, default='cifar10', choices=['cifar10, cifar100', 'mnist', 'nmnist'])
        self.parser.add_argument('--device_id', type=int, default=0)

def generate_noisy_data_with_psnr(data, target_psnr):
    max_pixel = 1.0  # PyTorch 텐서의 경우 0과 1 사이로 정규화된 이미지라고 가정
    mse_target = (max_pixel ** 2) / (10 ** (target_psnr / 10))
    
    # mse_target을 텐서로 변환
    mse_target_tensor = torch.tensor(mse_target, dtype=data.dtype, device=data.device)
    
    std_dev = torch.sqrt(mse_target_tensor)
    
    noise = torch.normal(0, std_dev, size=data.shape, device=data.device)
    noisy_image = data + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)  # 픽셀 값을 0과 1 사이로 제한
    
    return noisy_image

class MyNormalize:
    def __init__(self):
        pass
    def __call__(self, data):
        vmax, vmin = data.max(), data.min()
        return (data-vmin)/(vmax-vmin)

def visualize_loss(pilot: bool, train_loss: list, valid_loss, path: str=None):
    plt.figure(figsize=(10,6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if pilot is True:
        plt.show()
    else:
        plt.savefig(path)

def forward_pass(net, data, model_type='snn'):
    spk_rec = []
    if model_type == 'snn':
        utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out = net(data[step])
        spk_rec.append(spk_out)

    outputs = torch.stack(spk_rec)

    if model_type == 'cnn':
        outputs = torch.mean(outputs, dim=0)

    return outputs

#
# Main
#
def main(args):
    ### 트랜스폼 준비
    transform = transforms.Compose([transforms.ToTensor(),
                                    MyNormalize()])
    ##*

    ### 데이터셋 종류에 따라 다르게 준비
    if( args.dataset_type == 'cifar10' ):
        train_dataset = datasets.CIFAR10(root=path_dataset, train=True, transform=transform, download=False)
        test_dataset = datasets.CIFAR10(root=path_dataset, train=False, transform=transform, download=False)
    elif( args.dataset_type == 'cifar100' ):
        train_dataset = datasets.CIFAR100(root=path_dataset, train=True, transform=transform, download=False)
        test_dataset = datasets.CIFAR100(root=path_dataset, train=False, transform=transform, download=False)
    elif( args.dataset_type == 'mnist'):
        train_dataset = datasets.MNIST(root=path_dataset, train=True, transform=transform, download=False)
        test_dataset = datasets.MNIST(root=path_dataset, train=False, transform=transform, download=False)
    else:
        sensor_size = tonic.datasets.NMNIST.sensor_size
        transform = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.ToFrame(sensor_size=sensor_size,
                               time_window=1000),
            torch.from_numpy,
            torchvision.transforms.RandomRotation([-10,10]),
            MyNormalize()
        ])

        train_dataset = tonic.datasets.NMNIST(save_to='/DATA/hwkang', train=True, transform=transform)
        test_dataset = tonic.datasets.NMNIST(save_to='/DATA/hwkang', train=False, transform=transform)
    ##*
    
    ### 데이터셋 크기 조절
    if( args.training_dataset_ratio < 1.0 ):
        train_dataset = get_single_subset_by_ratio(train_dataset, ratio=args.training_dataset_ratio)
    if( args.test_dataset_ratio < 1.0 ):
        test_dataset = get_single_subset_by_ratio(test_dataset, ratio=args.test_dataset_ratio)
    ##*

    ### 훈련 데이터셋과 검증 데이터셋으로 분할
    if args.disable_validation == False:
        dataset_size = len(train_dataset)
        train_size = int(0.9 * dataset_size)
        valid_size = dataset_size - train_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    ##*

    ### NMNIST 데이터셋 전처리
    if args.dataset_type == 'nmnist':
        train_dataset = MemoryCachedDataset(train_dataset)
        if args.disable_validation == False:
            valid_dataset = MemoryCachedDataset(valid_dataset)
        test_dataset = MemoryCachedDataset(test_dataset)
    ##*

    ### 데이터로더 준비
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if args.disable_validation == False:
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.dataset_type == 'nmnist':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
        if args.disable_validation == False:
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=False)
    ##*

    ### 디바이스 설정
    device = torch.device(f'cuda:{int(args.device_id)}' if torch.cuda.is_available() else 'cpu')
    ##*
    
    ### 모델 생성
    if( args.model == 'resnet50' ):
        model = resnet50(weights=None, num_classes=10) if args.dataset_type=='cifar10' else resnet50(weights=None, num_classes=100)
    elif( args.model == 'resnet101' ):
        model = resnet101(weights=None, num_classes=10) if args.dataset_type=='cifar10' else resnet50(weights=None, num_classes=100)
    elif( args.model == 'cnn' ):
        model = CNN(dataset_type=args.dataset_type)
    elif( args.model[:3] == 'snn'):
        original_model = resnet50(weights=None, num_classes=10) #
        if( args.model == 'snn1' ):
            model = SpikingResNet50_1(original_model)
        elif( args.model == 'snn2' ):
            model = SpikingResNet50_2(original_model)
        elif( args.model == 'snn3' ):
            model = SpikingCNN(dataset_type=args.dataset_type, num_steps=args.num_steps)

    ### 사전 훈련 된 모델을 사용하는 경우
    if( args.pretrained is not None ):
        model.load_state_dict(torch.load(args.pretrained))
    ##*

    ### To GPU
    model = model.to(device)
    ##*

    ### 손실 함수 선택
    if args.model != 'snn3':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = SF.ce_rate_loss()
    ##*

    ### 옵티마이저 설정
    if( args.optimizer == 'adam' ):
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    elif( args.optimizer == 'sgd' ):
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    ##*
    
    ### 에폭 수 설정
    num_epochs = args.epoch
    ##*
    
    ### 학습 기록용 변수 설정
    train_loss_rec = [] # 훈련 로스 기록
    valid_loss_rec = [] # 검증 로스 기록
    max_valid_loss = math.inf # 모델 저장 시 직전 로스
    ###

    ### LR 스케줄러 생성
    if args.lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    ##*

    ### 조기 종료 설정
    early_stopping = EarlyStopping(patience=20, min_delta=1e-8)
    ##*

    ### 시간 측정용 리스트
    list_inference_time = []
    list_backprop_time = []
    list_noise_generation_time = []
    ##*

    ### 훈련 루프
    if( args.disable_training == False ):
        for epoch in range(num_epochs):
            running_loss = 0.0

            model.train()
            ### 훈련 단계
            step = 1
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                noise_generation_time_start = 0
                noise_generation_time_end = 0
                if( args.noise_training ):
                    noise_generation_time_start = time.time()
                    inputs = generate_noisy_data_with_psnr(inputs, torch.randint(1, 36, (1,)).item())
                    noise_generation_time_end = time.time()
                
                optimizer.zero_grad()

                ### 배치 순전파
                inference_time_start = time.time()
                if (args.dataset_type=='nmnist'):
                    outputs = forward_pass(model, inputs, model_type=args.model)
                else:
                    outputs = model(inputs)
                inference_time_end = time.time()
                ##*

                ### 역전파 시간 측정
                backprop_time_start = time.time()
                
                ### 손실 계산
                loss = criterion(outputs, labels)
                ##*

                ### 역전파
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                ##*
            
                backprop_time_end = time.time()
                ##*
            
                ### 시간 계산 수집
                list_noise_generation_time.append(noise_generation_time_end - noise_generation_time_start)
                list_inference_time.append(inference_time_end - inference_time_start)
                list_backprop_time.append(backprop_time_end - backprop_time_start)
                ##*

                if ( step % 1000 == 0 ) and ( args.verbose ) and ( args.dataset_type == 'nmnist'):
                    print(f'Epoch: {epoch+1}/{num_epochs} | Step: {step}/{len(train_loader)} | Running Loss: {running_loss/step:.4f}')
                step += 1
            ##* 에폭 종료
            
            ### 검증 단계
            if( args.disable_validation == False ):
                model.eval()
                valid_loss = 0.0
                all_labels = []
                all_predictions = []
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
    
                        ### 배치 순전파
                        inference_time_start = time.time()
                        if (args.dataset_type=='nmnist'):
                            outputs = forward_pass(model, inputs, model_type=args.model)
                        else:
                            outputs = model(inputs)
                        inference_time_end = time.time()
                        ##*
                        
                        ### 손실 계산
                        loss = criterion(outputs, labels)
                        ##*
    
                        ### 손실 저장
                        valid_loss += loss.item()
                        ##*
    
                        ### 검증 정확도 재료 수집                        
                        if args.model == 'snn3':
                            outputs = torch.sum(outputs, dim=0)
                            
                        _, predicted_class = torch.max(outputs, dim=1)
                        all_labels.extend(labels.detach().cpu().numpy())
                        all_predictions.extend(predicted_class.detach().cpu().numpy())
                        ##*
    
                # 검증 정확도 산출
                valid_accuracy, _, _, _ = get_classification_metrics(all_labels, all_predictions, 'weighted')
                ##* 검증 종료
            
            ### 훈련 결과 기록
            train_loss = running_loss / len(train_loader.dataset)
            train_loss_rec.append(train_loss)
            if( args.disable_validation == False ):
                valid_loss = valid_loss / len(valid_loader.dataset)
                valid_loss_rec.append(valid_loss)
            ##*
            
            ### 에폭 훈련 결과 출력
            print(f'\nEpoch: [{epoch+1}/{num_epochs}] ############################################################')
            if( args.disable_validation == False ):
                print(f'Epoch Loss: {train_loss:.4f} | Validation Loss: {valid_loss:.4f} | Validation Accuracy: {valid_accuracy*100:.2f}%')
            else:
                print(f'Epoch Loss: {train_loss:.4f}')
            
            avg_noise_generation_time = sum(list_noise_generation_time)/len(list_noise_generation_time)
            avg_inference_time = sum(list_inference_time)/len(list_inference_time)
            avg_backprop_time = sum(list_backprop_time)/len(list_backprop_time)
            print(f'Avg. Noise Generation Time: {avg_noise_generation_time*1000:.3f} ms | Avg. Inference Time: {avg_inference_time*1000:.3f} ms | Avg. Backprop Time: {avg_backprop_time*1000:.3f} ms')
    
            if 'metrics' in result_set:
                record = {
                    'noise_generation': avg_noise_generation_time,
                    'inference': avg_inference_time,
                    'backprop': avg_backprop_time
                }
                save_record_to_csv(path_file_metrics_latency, record)
            ##*

            # 검증 손실 개선이 있었는가?
            if( valid_loss < max_valid_loss ):
                if( args.verbose ):
                    print(f'Model saved: [Last] {max_valid_loss:.4f} >> {valid_loss:.4f} [Current]')
                best_model_state = model.state_dict()
                if 'model' in result_set:
                    torch.save(best_model_state, path_file_model)
                best_epoch_idx = epoch + 1
                max_valid_loss = valid_loss
                
            # 손실 변화에 따른 학습률 변경
            if args.lr_scheduler:
                scheduler.step(valid_loss)
                if args.verbose:
                    print(f"Learning rate after epoch: {scheduler.get_last_lr()[0]}")
                
            # 검증 손실 개선이 없어 종료해야 하는가?
            if args.early_stopping:
                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    if args.verbose:
                        print(f"\nEarly stopping with: (validation loss: {valid_loss:.6f})")
                    break
        ### 전체 훈련 루프 종료

        # 시험용 모델 획득
        if args.verbose:
            print(f'\nLoad model state at best validation loss [{best_epoch_idx}]\n')
        model.load_state_dict(best_model_state)
        ###

        # 훈련 진행에 따른 손실 시각화
        if 'loss' in result_set:
            visualize_loss(pilot=False, train_loss=train_loss_rec, valid_loss=valid_loss_rec, path=path_file_loss)
        ###

    ### 시험 단계
    if( args.disable_test == False ):
        model.eval()
        all_labels = []
        all_predictions = []
        acc = 0.0
        total = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                if( args.noise_test ):
                    inputs = generate_noisy_data_with_psnr(inputs, args.noise_intensity)
            
                ### 순전파
                if (args.dataset_type=='nmnist'):
                    outputs = forward_pass(model, inputs, model_type=args.model)
                else:
                    outputs = model(inputs)
                
                ### 메트릭 수집
                if args.model == 'snn3':
                    outputs = torch.sum(outputs, dim=0)
                            
                _, predicted_class = torch.max(outputs, dim=1)
                all_labels.extend(labels.detach().cpu().numpy())
                all_predictions.extend(predicted_class.detach().cpu().numpy())
                ##*
                ##*

        
        a, p, r, f = get_classification_metrics(all_labels, all_predictions, 'weighted')
        record = {
            'accuracy': a,
            'precision': p,
            'recall': r,
            'f1-score': f}
        print(f'Accuracy: {a:.3f} | Precision: {p:.3f} | Recall: {r:.3f} | F1-Score: {f:.3f}')
        
        ###
        if 'metrics' in result_set:
            save_record_to_csv(path_file_metrics, record)
        ##*
    
        ###
        if 'confusion_matrix' in result_set:
            visualize_confusion_matrix(False, all_labels, all_predictions, 10, path_file_confusion_matrix)
        ##*
    ##*
    print("\nSuccess!\n")

#
# Init
#
if __name__=='__main__':
    parser = InheritedRensArgumentParser()
    args = parser.get_argument_parser()

    ### 디렉터리 경로 설정
    path_dataset = os.path.join(path_root, 'dataset')
    path_result_root = os.path.join(path_root, 'result/exp09')
    path_image = os.path.join(path_result_root, 'image')
    path_loss = os.path.join(path_result_root, 'loss')
    path_meta = os.path.join(path_result_root, 'meta')
    path_metrics = os.path.join(path_result_root, 'metrics')
    path_model = os.path.join(path_result_root, 'model')
    ##*

    ### 메타 데이터 생성
    xid = get_next_xid(path_meta)
    current_time = get_current_time_str()
    argument_info = parser.get_argument_info(output_format=args.output_format)
    if args.output_format == 'string':
        meta_data = [f'xid: {str(xid)}'] + [current_time] + argument_info
    else:
        meta_data = {'xid': str(xid), 'current_time': current_time}
        meta_data = meta_data | argument_info
    ##*

    ### 메타 데이터 프롬프트에 출력
    if( args.verbose ):
        print('\n')
        if args.output_format == 'string':
            for line in meta_data:
                print(line)
        else:
            for k, v in meta_data.items():
                print(f'{k}: {v}')
        print('\n')
    else:
        print("\nSkip printing out meta data.\n")
    ##*

    ### 파일 경로 설정
    path_file_image = os.path.join(path_image, f'{xid:04d}_{exp_no}_image_{args.prefix}{args.body}{args.suffix}.png')
    path_file_loss = os.path.join(path_loss, f'{xid:04d}_{exp_no}_loss_{args.prefix}{args.body}{args.suffix}.png')
    path_file_meta = os.path.join(path_meta, f'{xid:04d}_{exp_no}_meta_{args.prefix}{args.body}{args.suffix}.txt')
    path_file_confusion_matrix = os.path.join(path_metrics, f'{xid:04d}_{exp_no}_confusion_matrix_{args.prefix}{args.body}{args.suffix}.png')
    path_file_metrics = os.path.join(path_metrics, f'{exp_no}_metrics_{args.prefix}{args.body}{args.suffix}.csv')
    path_file_metrics_latency = os.path.join(path_metrics, f'{exp_no}_metrics_latency_{args.prefix}{args.body}{args.suffix}.csv')
    path_file_model = os.path.join(path_model, f'{xid:04d}_{exp_no}_model_{args.prefix}{args.body}{args.suffix}.weights')
    ##*

    ### 메타 데이터 저장
    if( args.output_format == 'csv' ):
        path_file_meta = os.path.join(path_meta, f'{exp_no}_meta_{args.prefix}{args.body}{args.suffix}.csv')
        save_record_to_csv(path_file_meta, meta_data)
    else:
        with open(path_file_meta, 'w') as file:
            for line in meta_data:
                file.write(line + '\n')
    ##*

    ###
    if args.path_file_metrics is not None:
        path_file_metrics = args.path_file_metrics
    ##*
    if args.path_file_metrics_latency is not None:
        path_file_metrics_latency = args.path_file_metrics_latency



    ###
    result_set = set(args.write_to)
    ##*

    ### 실행
    try:
        start_time = time.time()
        main(args)
        write_metadata_status(path_file_meta, 'SUCCESS')
    except KeyboardInterrupt:
        write_metadata_status(path_file_meta, 'HALTED')
    except Exception as e:
        _, _, tb = sys.exc_info()
        trace = traceback.format_tb(tb)
        with open(path_file_meta, 'a') as file:
            file.writelines(trace)
        print(f"FAILED({type(e).__name__}: {e})")
        print(''.join(trace))
    finally:
        elapsed_time = time.time() - start_time
        with open(path_file_meta, 'a') as file:
            file.write(f'\nTotal elapsed time: {elapsed_time:.6f} s')
        print( f'\nTotal elapsed time: {elapsed_time:.6f} s\n' )
    ##*