# Experiment no.09 [regular] [v3]

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

from torchvision import datasets, transforms
from torchvision.models import resnet50, resnet101

# SnnTorchs
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt 

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

#
# Specifics
#
class InheritedRensArgumentParser(RensArgumentParser):
    def __init__(self):
        super().__init__()
        ### 부모 파서의 인자 처리 정보를 받아옴, 충돌 시 아래 추가할 것임을 명시
        self.parser = argparse.ArgumentParser(parents=[self.parser], conflict_handler='resolve')
        ##*
        self.parser.add_argument('-m', '--model', type=str, default='resnet101', choices=['cnn', 'resnet50', 'resnet101', 'snn'], help="Type of a model to use. (Default: resnet101)")
        self.parser.add_argument('--noise_training',action='store_true',
                                 default=False)
        self.parser.add_argument('--noise_test', action='store_true',
                                 default=False)
        self.parser.add_argument('--noise_intensity', type=float, default=0.5)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.fc1 = nn.Linear(64*5*5, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 64*5*5)
        x = self.fc1(x) #  >> [ batch_size, 10 ]

        return x

#
# Main
#
def main(args):
    ### 트랜스폼 준비
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    ##*

    ### 데이터셋 종류에 따라 다르게 준비
    if( args.dataset_type == 'cifar10' ):
        train_dataset = datasets.CIFAR10(root=path_dataset, train=True, transform=transform, download=False)
        test_dataset = datasets.CIFAR10(root=path_dataset, train=False, transform=transform, download=False)
    else:
        train_dataset = datasets.CIFAR100(root=path_dataset, train=True, transform=transform, download=False)
        test_dataset = datasets.CIFAR100(root=path_dataset, train=False, transform=transform, download=False)
    ##*
    
    ### 데이터셋 크기 조절
    if( args.training_dataset_ratio < 1.0 ):
        train_dataset = get_single_subset_by_ratio(train_dataset, ratio=args.training_dataset_ratio)
    if( args.test_dataset_ratio < 1.0 ):
        test_dataset = get_single_subset_by_ratio(test_dataset, ratio=args.test_dataset_ratio)
    ##*

    ### 훈련 데이터셋과 검증 데이터셋으로 분할
    if args.enable_validation::
        dataset_size = len(train_dataset)
        train_size = int(0.8 * dataset_size)
        valid_size = dataset_size - train_size
        train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    ##*

    ### 데이터로더 준비
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if args.enable_validation:
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    ##*
    
    ### 디바이스 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ##*
    
    ### 모델 생성
    if( args.model == 'resnet50' ):
        model = resnet50(weights=None, num_classes=10) if args.dataset_type=='cifar10' else resnet50(weights=None, num_classes=100)
        criterion = nn.CrossEntropyLoss()
    elif( args.model == 'resnet101' ):
        model = resnet101(weights=None, num_classes=10) if args.dataset_type=='cifar10' else resnet50(weights=None, num_classes=100)
        criterion = nn.CrossEntropyLoss()
    elif( args.model == 'cnn' ):
        model = CNN()
        criterion = nn.CrossEntropyLoss()
       
    elif( args.model == 'snn'):
        num_steps = 50
        beta = 0.9
        model = Net(beta, surrogate.fast_sigmoid(slope=25))
        criterion = SF.ce_rate_loss()
    if( args.pretrained is not None ):
        model.load_state_dict(torch.load(args.pretrained))
    model = model.to(device)
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
    epoch_loss_rec = [] # 훈련 로스 기록
    val_loss_rec = [] # 검증 로스 기록
    max_epoch_loss = math.inf # 모델 저장 시 직전 로스
    ###

    ### LR 스케줄러 생성
    scheduler = CosineAnnealingLR(optimizer, T_max=200)
    ##*

    ### 조기 종료 설정
    early_stopping = EarlyStopping(patience=10, min_delta=1e-6)
    ##*

    ### 훈련 루프
    if( args.enable_training ):
        for epoch in range(num_epochs):
            running_loss = 0.0
            model.train()
            ### 훈련 단계
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 33.3% 확률로 무작위 강도 잡음 주입
                if( args.noise_training ):
                    trigger = torch.rand(1).item()
                    if( trigger >= 0.66 ):
                        inputs = generate_noisy_data(inputs, torch.rand(1).item(), 'gaussian',
                                                     device=device)
                
                optimizer.zero_grad()

                ### 배치 순전파
                if args.model == 'snn':
                    outputs, _ = forward_pass(model, num_steps, inputs, args.batch_size)
                else:
                    outputs = model(inputs)
                ##*
                
                ### 손실 계산
                loss = criterion(outputs, labels)
                ##*

                ### 역전파
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                ##*
            ##* 에폭 종료
            
            ### 검증 단계
            if( args.enable_validation):
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
    
                        ### 배치 순전파
                        if args.model == 'snn':
                            outputs, _ = forward_pass(model, num_steps, inputs, args.batch_size)
                        else:
                            outputs = model(inputs)
                        ##*
                        
                        ### 손실 계산
                        loss = criterion(outputs, labels)
                        ##*
    
                        ### 손실 저장
                        val_loss += loss.item()
                        ##*
    
                        ### 검증 정확도 재료 수집
                        if args.model == 'snn':
                            correct += SF.accuracy_rate(outputs, labels) * outputs.size(1)
                            total += outputs.size(1)
                        else:
                            _, predicted = torch.max(outputs, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                        ##*
    
                # 검증 정확도 산출
                val_accuracy = 100 * correct / total
                ##* 검증 종료
            
            ### 에폭 훈련 결과 기록
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_loss_rec.append(epoch_loss)
            ##*
            
            ### 에폭 훈련 결과 출력
            print(f'\nEpoch: [{epoch+1}/{num_epochs}] ############################################################')
            if( args.enable_validation ):
                print(f'Epoch Loss: {running_loss / len(train_loader.dataset):.4f} | Validation Loss: {val_loss / len(valid_loader.dataset):.4f} |  Validation Accuracy: {val_accuracy:.2f}%')
                val_loss_rec.append(val_loss)
            else:
                print(f'Epoch Loss: {running_loss / len(train_loader.dataset):.4f}')
            ##*

            # 손실 개선이 있었는가?
            if( epoch_loss < max_epoch_loss ):
                if( args.verbose ):
                    print(f'Model saved: [Last] {max_epoch_loss:.4f} >> {epoch_loss:.4f} [Current]')
                best_model_state = model.state_dict()
                torch.save(best_model_state, path_file_model)
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
            visualize_epoch_loss(pilot=False, epoch_loss=epoch_loss_rec, path=path_file_loss)
        ###

    ### 시험 단계
    if( args.enable_test ):
        model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

            if( args.noise_training ):
                inputs = generate_noisy_data(inputs, args.noise_intensity, 'gaussian',
                                             device=device)
            ### 순전파
            if args.model == 'snn':
                outputs, _ = forward_pass(model, num_steps, inputs, args.batch_size)

                ### 메트릭 수집
                spk_rec_avg = outputs.mean(dim=0)
                predictions = spk_rec_avg.argmax(dim=1)
                all_labels_000.extend(labels.detach().cpu().numpy())
                all_predictions_000.extend(predictions.detach().cpu().numpy())
                ##*
            else:
                outputs = model(inputs)

                ### 메트릭 수집
                predicted_value, predicted_class = torch.max(outputs, dim=1)
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
        save_record_to_csv(path_file_metrics, record)
        print(f'Accuracy: {a:.3f} | Precision: {p:.3f} | Recall: {r:.3f} | F1-Score: {f:.3f}')
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
    output_format = 'string'
    argument_info = parser.get_argument_info(output_format=output_format)
    meta_data = [f'xid: {str(xid)}'] + [current_time] + argument_info
    ##*

    ### 메타 데이터 프롬프트에 출력
    if( args.verbose ):
        print('\n')
        for line in meta_data:
            print(line)
        print('\n')
    else:
        print("\nSkip printing out meta data.\n")
    ##*

    ### 파일 경로 설정
    path_file_image = os.path.join(path_image, f'{xid:04d}_{exp_no}_image_{args.prefix}{args.body}{args.suffix}.png')
    path_file_loss = os.path.join(path_loss, f'{xid:04d}_{exp_no}_loss_{args.prefix}{args.body}{args.suffix}.png')
    path_file_meta = os.path.join(path_meta, f'{xid:04d}_{exp_no}_meta_{args.prefix}{args.body}{args.suffix}.txt')
    path_file_metrics = os.path.join(path_metrics, f'{exp_no}_metrics_{args.prefix}{args.body}{args.suffix}.csv')
    path_file_model = os.path.join(path_model, f'{xid:04d}_{exp_no}_model_{args.prefix}{args.body}{args.suffix}.weights')
    ##*

    ### 메타 데이터 저장
    if( output_format == 'csv' ):
        path_file_meta = os.path.join(path_meta, f'{exp_no}_meta_{args.prefix}{args.body}{args.suffix}.csv')
        save_record_to_csv(path_file_meta, meta_data)
    else:
        with open(path_file_meta, 'w') as file:
            for line in meta_data:
                file.write(line + '\n')
    ##*

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