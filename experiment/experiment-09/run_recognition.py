# Experiment no.09 [regular]

#
EXP_NO = 'exp09'
PATH_ROOT = '/home/hwkang/jupyter/root/'

# Imports

# Fundamental
import os
import sys
import argparse

# PyTorch
import torch
torch.manual_seed(0) # 결과 일관성 확보 목적
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision

import torchmetrics

# SnnTorch
import snntorch as snn
import snntorch.functional as SF
import snntorch.surrogate as surrogate

# Tonic
import tonic
import tonic.transforms
from tonic import MemoryCachedDataset

# Utility
import math
from utility.common import (
    ensure_directories, write_metadata_status
)

from utility.parser import RensArgumentParser
from utility.preprocessing import *
from utility.statistic import *

from utility.visualization import (
    get_current_time_str, get_next_xid
)

from models import SpikingCNN_v1 as SpikingCNN, CNN, CustomResNet50


# Main
def main(args):
    # Transform 설정
    sensor_size = tonic.datasets.NMNIST.sensor_size
    
    transform_1 = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.n_time_bins),
        torch.from_numpy
    ])

    if args.transform_type == 'original':
        pass
    elif args.transform_type == 'uniform':
        transform_2 = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.UniformNoise(sensor_size=sensor_size, n=args.n_events),
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.n_time_bins),
            torch.from_numpy
        ])
    elif args.transform_type == 'jitter':
        transform_2 = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.TimeJitter(std=args.time_jitter_std, clip_negative=True),
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.n_time_bins),
            torch.from_numpy
        ])
    elif args.transform_type == 'complex':
        transform_2 = tonic.transforms.Compose([
            tonic.transforms.Denoise(filter_time=10000),
            tonic.transforms.TimeJitter(std=args.time_jitter_std, clip_negative=True),
            tonic.transforms.UniformNoise(sensor_size=sensor_size, n=args.n_events),
            tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.n_time_bins),
            torch.from_numpy
        ])
    else:
        raise ValueError(f'Invalid transform_type({args.transform_type}).')
    #..

    # 데이터셋 로드
    dataset = RensDataset(dataset_type=args.dataset_type, path='/DATA/hwkang/')
    if args.transform_type == 'original':
        train_dataset = dataset.get_dataset(train=True, transform=transform_1)
        test_dataset = dataset.get_dataset(train=False, transform=transform_1)
    else:
        train_dataset = dataset.get_dataset(train=True, transform=transform_2)
        test_dataset = dataset.get_dataset(train=False, transform=transform_2)
    train_dataset, test_dataset = MemoryCachedDataset(train_dataset), MemoryCachedDataset(test_dataset)
    #..

    # DataLoader 인스턴스 생성
    ## 데이터셋 크기 변환
    if(args.train_dataset_size > 0):
        train_dataset = get_single_subset_by_size(dataset=train_dataset, target_size=args.train_dataset_size, randomize=True)
    if(args.train_dataset_ratio < 1.0):
        train_dataset = get_single_subset_by_ratio(dataset=train_dataset, ratio=args.train_dataset_ratio)
    if(args.test_dataset_size > 0):
        test_dataset = get_single_subset_by_size(dataset=test_dataset, target_size=args.test_dataset_size, randomize=False)
    if(args.test_dataset_ratio < 1.0):
        test_dataset = get_single_subset_by_ratio(dataset=test_dataset, ratio=args.test_dataset_ratio)
    ##..
    ## 검증 데이터 분할
    if args.disable_validation is False:
        # 훈련 : 검증 = 8 : 2
        train_dataset, valid_dataset = get_two_subsets_by_ratio(dataset=train_dataset, split_ratio=8)
    ##..
    import multiprocessing
    batch_size = args.batch_size
    num_workers = multiprocessing.cpu_count() // 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=tonic.collation.PadTensors(batch_first=False))
    if args.disable_validation is False:
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=tonic.collation.PadTensors(batch_first=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=tonic.collation.PadTensors(batch_first=False))
    #..

    # 하이퍼파라미터 설정
    device_id = args.device_id
    if args.device_id is not None:
        device_id = args.device_id[0] # << Expected zero('0')
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    num_epochs = args.epoch
    ## SNN
    if args.model == 'snn':
        beta = args.beta
        spike_grad = surrogate.atan()
        model = SpikingCNN(beta=beta, spike_grad=spike_grad)
        criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    ##..
    ## CNN
    elif args.model != 'snn':
        if args.model == 'cnn':
            model = CNN()
        elif args.model == 'resnet':
            model = CustomResNet50(num_classes=10)
        criterion = nn.CrossEntropyLoss()
    ##..
    else:
        raise ValueError(f'Invalid model({args.model}).')
    optimizer = optim.Adam(model.parameters(), lr=2e-2, betas=(0.9, 0.999))
    model = model.to(device)
    if args.pretrained is not None:
        model_name = args.pretrained
        model_path = os.path.join(path_model, model_name)
        model.load_state_dict(torch.load(model_path))
    #..

    # 훈련 루프
    if args.disable_training is False:
        epoch_loss_rec = []
        batch_loss_rec = []
        valid_loss_rec = []
        valid_accuracy_rec = []
        min_epoch_loss = math.inf
        epoch_time_rec = []
        forward_time_rec = []
        backprop_time_rec = []
        for epoch in range(num_epochs):
            ## 1 Epoch
            epoch_time_start = time.time()
            model.train()
            batch_loss = 0.0
            current_step, current_size = 0, 0
            for inputs, labels in train_loader:
                ### Forward Feed
                optimizer.zero_grad()
                forward_time_start = time.time()
                if args.model == 'snn':
                    outputs = model(inputs.to(device))
                else: # CNN
                    outputs = model(inputs.to(device))
                forward_time_rec.append(time.time()-forward_time_start)
                ###..
                ### Backprop
                backprop_time_start = time.time()
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                backprop_time_rec.append(time.time()-backprop_time_start)
                ###..
                ### 배치 훈련 결과 정리
                batch_loss += loss.item()
                current_step += 1
                current_size += labels.size(0)
                ###..
                ### 배치 훈련 결과 출력
                target_step = ( len(train_loader.dataset) // 5 ) // batch_size
                if( current_step % target_step == 0):
                    print(f'Epoch: {epoch}/{num_epochs} | Inputs: {current_size}/{len(train_loader.dataset)} | Batch Loss(Train): {batch_loss/current_size:.6f}')
                batch_loss_rec.append(batch_loss/current_size)
                ###..
            epoch_loss = batch_loss / len(train_loader.dataset)
            print(f'Epoch: {epoch+1}/{num_epochs} | Epoch Loss: {epoch_loss:.6f}')

            ### 모델 성능 개선 시 저장
            if min_epoch_loss > epoch_loss:
                best_state = model.state_dict()
                if args.save:
                    print(f'Model Saved [{min_epoch_loss:.6f}] -> [{epoch_loss:.6f}].')
                    torch.save(best_state, path_file_model)
                min_epoch_loss = epoch_loss
            epoch_loss_rec.append(epoch_loss)
            ###..
            ##..
            
            ## 검증
            if args.disable_validation is False:
                model.eval()
                batch_loss = 0.0
                current_step, current_size = 0, 0
                all_predictions = []
                all_labels = []
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        ### Forward Feed
                        if args.model == 'snn':
                            outputs = model(inputs.to(device))
                        else: # CNN
                            outputs = model(inputs.to(device))
                        ###..
                        batch_loss += criterion(outputs, labels.to(device))                    

                        ### 결과 수집
                        all_labels.append(labels.detach().cpu()) # >> n x [b] (n: 전체 배치 개수)
                        if args.model == 'snn':
                            all_predictions.append(outputs.detach().cpu()) # >> [t, b, num_classes] 
                        else:
                            predictions = torch.argmax(outputs, dim=1) # [b, num_classes] >> [b]
                            all_predictions.append(predictions.detach().cpu())
                        ###..

                ### 지표 산출
                valid_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=10)
                all_labels = torch.cat(all_labels, dim=0) # n x [b] >> [n x b]
                if args.model == 'snn':
                    all_predictions = torch.cat(all_predictions, dim=1)
                    valid_accuracy = SF.accuracy_rate(all_predictions, all_labels)
                else:
                    all_predictions = torch.cat(all_predictions, dim=0)
                    valid_accuracy = valid_accuracy_metric(all_predictions, all_labels)
                ###..
                print(f'Epoch: {epoch+1}/{num_epochs} | Valid Loss: {batch_loss / len(valid_loader.dataset):.6f} | Valid Accuracy: {valid_accuracy*100:.4f}%\n')
                valid_loss_rec.append(batch_loss / len(valid_loader.dataset))
                valid_accuracy_rec.append(valid_accuracy)
            ##..
            else:
                print('Skip Validation.\n')
            epoch_time_rec.append(time.time()-epoch_time_start)
    else:
        print('Skip Training.\n')
    #..

    # 평가
    if args.disable_test is False:
        model.eval()
        test_accuracy_metrics = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        current_step = 0
        current_size = 0
        target_step = (len(test_loader.dataset) // 5) // batch_size
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                if args.model == 'snn':
                    outputs = model(inputs.to(device))
                    test_batch_accuracy = SF.accuracy_rate(outputs.detach().cpu(), labels)
                    all_predictions.append(outputs.detach().cpu())
                else:
                    outputs = model(inputs.to(device))
                    test_batch_accuracy = test_accuracy_metrics(outputs.detach().cpu(), labels)
                    all_predictions.append(outputs.argmax(dim=1).detach().cpu())
                all_labels.append(labels.detach().cpu())
                
                current_step += 1
                current_size += labels.size(0)
                if current_step % target_step == 0:
                    print(f'Step {current_size}/{len(test_loader.dataset)} | Batch Accuracy(Test): {test_batch_accuracy*100:.4f}%')
        
        all_labels = torch.cat(all_labels, dim=0)
        if args.model == 'snn':
            all_predictions = torch.cat(all_predictions, dim=1) # >> [t, n*b, num_classes]
            test_accuracy = SF.accuracy_rate(all_predictions, all_labels)
            #all_predictions = all_predictions.permute(1,0,2).sum(dim=2).argmax(dim=1) # >> [n*b, t, num_classes] >> [n*b, t] >> [n*b]
        else:
            all_predictions = torch.cat(all_predictions, dim=0 if args.model == 'cnn' else 1)
            test_accuracy = test_accuracy_metrics(all_predictions, all_labels)
            test_accuracy = test_accuracy.item()
        print(f'Test Accuracy: {test_accuracy*100:4f}%\n')
    else:
        print('Skip Test.\n')
    #..

    # 결과 정리
    if 'metrics' in which_to_write:
        ## 예측 지표
        #a, p, r, f = get_classification_metrics(all_labels, all_predictions, 'weighted')
        classification_metrics_record = {'xid': xid, 'accuracy': test_accuracy}
        save_record_to_csv(path_file_metrics_accuracy, classification_metrics_record)
        ##..
        if args.disable_training is False: # 재학습에 활용 가능성을 확인하기 위해 훈련 루프에 적용
            ## 지연 시간
            average_epoch_time = sum(epoch_time_rec)/len(epoch_time_rec)
            average_forward_time = sum(forward_time_rec)/len(forward_time_rec)
            average_backprop_time = sum(backprop_time_rec)/len(backprop_time_rec)
            latency_record = {'xid': xid, 'epoch': average_epoch_time, 'forward': average_forward_time, 'backprop': average_backprop_time}
            save_record_to_csv(path_file_metrics_latency, latency_record)
            ##..
            ## 전력 소비 (다른 스크립트에서 진행)
            ##..
    #..

    # 시각화 (다른 스크립트에서 진행)
    ## Loss Trend
    ##..
    #..
    
    print('\nSuccess!\n')

# Classes
class ExpandedRensArgumentParser(RensArgumentParser):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(parents=[self.parser], conflict_handler='resolve')
        self.parser.add_argument('-d', '--dataset_type', type=str, default='nmnist', 
                                 choices=['cifar10', 'cifar100', 'mnist', 'nmnist'], 
                                 help="Type of a dataset to use. (Default: nmnist)")
        self.parser.add_argument('--transform_type', type=str, default='original', choices=['original', 'uniform', 'jitter', 'complex'])
        self.parser.add_argument('--n_time_bins', type=int, default=30)
        self.parser.add_argument('--n_events', type=int, default=3000)
        self.parser.add_argument('--time_jitter_std', type=int, default=100)
        self.parser.add_argument('--save', action='store_true', default=False)

# Init
if __name__=='__main__':
    # 인자 처리 인스턴스 생성
    parser = ExpandedRensArgumentParser()
    args = parser.get_argument_parser()
    #..

    # 디렉터리 경로 설정
    path_dataset = os.path.join(PATH_ROOT, 'dataset')
    path_result_root = os.path.join(PATH_ROOT, f'result/{EXP_NO}')
    path_image = os.path.join(path_result_root, 'image')
    path_loss = os.path.join(path_result_root, 'loss')
    path_meta = os.path.join(path_result_root, 'meta')
    path_metrics = os.path.join(path_result_root, 'metrics')
    path_model = os.path.join(path_result_root, 'model')
    #..

    # 디렉터리 경로 확인
    ensure_directories(path_image, path_loss, path_meta, path_metrics, path_model)
    #..

    # 메타 데이터 설정
    xid = get_next_xid(path=path_meta)
    current_time = get_current_time_str()
    argument_info = parser.get_argument_info(output_format=args.output_format)
    if args.output_format == 'string':
        meta_data = [f'xid: {str(xid)}'] + [current_time] + argument_info
    elif args.output_format == 'csv':
        meta_data = {'xid': str(xid), 'current_time': current_time}
        meta_data = meta_data | argument_info
    else:
        raise ValueError(f'The output format [{args.output_format}] is invalid.')
    #..

    # 파일 경로 설정
    path_file_image = os.path.join(path_image, f'{xid:04d}_{EXP_NO}_image_{args.prefix}{args.body}{args.suffix}.png')
    path_file_loss = os.path.join(path_loss, f'{xid:04d}_{EXP_NO}_loss_{args.prefix}{args.body}{args.suffix}.png')
    path_file_meta = os.path.join(path_meta, f'{xid:04d}_{EXP_NO}_meta_{args.prefix}{args.body}{args.suffix}.txt')
    path_file_confusion_matrix = os.path.join(path_metrics, f'{xid:04d}_{EXP_NO}_confusion_matrix_{args.prefix}{args.body}{args.suffix}.png')
    ## header: [xid, accuracy, precision, recall, f1-score]
    path_file_metrics_accuracy = os.path.join(path_metrics, f'{EXP_NO}_metrics_accuracy_{args.prefix}{args.body}{args.suffix}.csv')
    ##..
    ## header: [xid, epoch-latency, forward-latency, backprop-latency]
    path_file_metrics_latency = os.path.join(path_metrics, f'{EXP_NO}_metrics_latency_{args.prefix}{args.body}{args.suffix}.csv')
    ##..
    ## header: [xid, power consumption of each layer]
    path_file_metrics_power = os.path.join(path_metrics, f'{EXP_NO}_metrics_power_{args.prefix}{args.body}{args.suffix}.csv')
    ##..
    path_file_model = os.path.join(path_model, f'{xid:04d}_{EXP_NO}_model_{args.prefix}{args.body}{args.suffix}.pt')
    #..

    # 메타 데이터 출력
    if args.verbose:
        print('\n')
        if args.output_format == 'string':
            for line in meta_data:
                print(line)
        elif args.output_format == 'csv':
            for k, v in meta_data.items():
                print(f'{k}: {v}')
        else:
            raise ValueError(f'The output format {args.output_format} is invalid.')
        print('\n')
    else:
        print("\nSkip printing out meta data.\n")
    #..

    # 메타 데이터 저장
    if( args.output_format == 'csv' ):
        path_file_meta = os.path.join(path_meta, f'{EXP_NO}_meta_{args.prefix}{args.body}{args.suffix}.csv')
        save_record_to_csv(path_file_meta, meta_data)
    else:
        with open(path_file_meta, 'w') as file:
            for line in meta_data:
                file.write(line + '\n')
    #..

    which_to_write = set(args.write_to)

    # 실행
    try:
        import time
        start_time = time.time()
        main(args)
        write_metadata_status(path_file_meta, 'SUCCESS')
    except KeyboardInterrupt:
        write_metadata_status(path_file_meta, 'HALTED')
    except Exception as e:
        import traceback
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
        print( f'\nXID: {xid} | Total elapsed time: {elapsed_time:.6f} s\n' )
    #..