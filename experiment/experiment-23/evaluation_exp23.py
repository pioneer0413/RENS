# Experiment no.23 evaluation

# Constants
from pathlib import Path
exp_no = 'exp23' # NOTICE: Change at each experiment!
directory = current_file_path = Path(__file__).resolve()
path_root = str(directory.parent.parent.parent) + '/'
path_dataset = path_root + 'dataset/'
path_result = path_root + 'result/'
path_result_root = path_result + exp_no + '/'
path_result_image = path_result_root + 'image/'
path_result_model = path_result_root + 'model/'
path_result_loss = path_result_root + 'loss/'
path_result_accuracy = path_result_root + 'accuracy/'
path_result_meta = path_result_root + 'meta/'
path_utility = path_root + 'utility/'

# Imports
import sys
sys.path.append(path_root)
sys.path.append(path_dataset)
sys.path.append(path_result)

# PyTorch family
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau #학습률 스케줄링
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Utilities
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

'''
from utility.common import *
from utility.preprocess import *
from utility.synthesize import *
from utility.visualize import *
from model.SimpleCNN import *
from model.LearningUtils import *
'''
from utility.statistic import *
from utility.common import *
from utility.preprocessing import *
from utility.synthesization import *
from utility.visualization import *
from model.simple_cnn import *
from model.learning_utility import *

'''
START of synthesize
'''
#import torch
#from torch.utils.data import Dataset
#import numpy as np
import random

class BinarilyNoisedDataset(Dataset):
    def __init__(self, data_loader, device, noise_type='gaussian', intensity_threshold=0.5, margin=0.0, sigma_reduction=0, label_balance=False):
        # intensity_threshold : label=0(약한잡음)의 분류기준점
        # margin : 분류기준점으로부터 데이터 미생성 범위
        # sigma_reduction : gaussian 정규분포 입력데이터에 대해 시그마 축소배율을 지정하여, 범위 내 데이터가 전체 정규분포 중 일정%를 대변하도록 함
        #                  sigma_reduction!=0 일 때 gaussian 정규분포 입력데이터 사용, ==0일 때는 uniform 균등분포 사용
        # label_balance : 각 label 별 할당되는 데이터 양이 같도록 함. 기존 실험16에서 사용된 방법
        self.x = []
        self.y = []

        if margin>=intensity_threshold or margin >= (1-intensity_threshold):
            raise ValueError("margin can't be equal or bigger than one label's range")
        
        for image, label in data_loader:
            image, label = image.to(device), label.to(device)
            image = image.squeeze(0)
            if(self.label_creation(intensity_threshold, label_balance)):
                self.x.append(generate_noisy_data(data=image, intensity=intensity_threshold + margin + self.intensity_creation(1-intensity_threshold-margin, sigma_reduction), noise_type=noise_type, rescale=True, normalize=True, device=device))
                self.y.append(1)
            else:
                self.x.append(generate_noisy_data(data=image, intensity=self.intensity_creation(intensity_threshold-margin, sigma_reduction), noise_type=noise_type, rescale=True, normalize=True, device=device))
                self.y.append(0)
                    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_data = self.x[idx]
        y_data = self.y[idx]
        return x_data, y_data

    def label_creation(self, intensity_threshold, label_balance):
        if label_balance==True:
            intensity_threshold=0.5
        return (np.random.rand() >= intensity_threshold)

    def intensity_creation(self, intensity_range, sigma_reduction):
        if sigma_reduction==0:
            return np.random.rand()*intensity_range
        else:
            while True:
                intensity = np.random.randn()/sigma_reduction
                '''
                sigma_reduction 별 abs(intensity) <= 1 의 의미 차이
                - sigma_reduction==1 : 정규분포 66% 이내의 값인지 여부를 확인
                - sigma_reduction==2 : 정규분포 95% 이내의 값인지 여부를 확인
                - sigma_reduction==3 : 정규분포 99.7% 이내의 값인지 여부를 확인
                * abs(value) <= sigma : 66%이내 값, sigma*2 : 95%이내 값, sigma*3 : 99.7%이내 값
                해당 범위 외 값이라면, 해당 범위 내 값이 될 때까지 반복
                '''
                if abs(intensity) <= 1:
                    # intensity의 데이터 존재 가능 범위 : -1 ~ 1 => 0 ~ 1 로 변환하기 위해 + 1 (0~2) 후 / 2 (0~1)
                    return (intensity + 1) / 2 * intensity_range
                
'''
END of synthesize
'''

'''
START of visualization
'''
def visualize_sample(pilot: bool, loader, path: str=None):
    # Unzip sample_batch to 10 samples
    x, y = next(iter(loader)) # Assume x: [n, 64, 1, 28, 28] -> [64, 1, 28, 28] after indexing
    
    # Select 10 samples
    samples = [(x[i], y[i]) for i in range(10)] # [64, 1, 28, 28] -> 10 * [1, 28, 28]
    
    # Create a 2 x 5 grid for images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        image = samples[i][0]
        
        # Convert tensor to numpy if needed
        if hasattr(image, 'numpy'):
            image = image.cpu().numpy()
        
        # If the image is a tensor from PyTorch, it will have a shape of [C, H, W].
        # Use permute to change this to [H, W, C] for matplotlib.
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))  # convert from [C, H, W] to [H, W, C]

        ax.imshow(image.squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f"Label:{samples[i][1].item() if hasattr(samples[i][1], 'item') else samples[i][1]}")

    if pilot:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(path)
'''
END of visualization
'''

def main(args):

    # Main implementation

    # Data preprocessing
    transform = transforms.Compose([transforms.ToTensor()])
    
    if( args.dataset_type == 'cifar10' ):
        train_dataset = datasets.CIFAR10(root=path_dataset,
                                         train=True,
                                         transform=transform,
                                         download=True)
        test_dataset = datasets.CIFAR10(root=path_dataset,
                                        train=False,
                                        transform=transform,
                                        download=True)
                                      
    else:
        train_dataset = datasets.CIFAR100(root=path_dataset,
                                         train=True,
                                         transform=transform,
                                         download=True)
        test_dataset = datasets.CIFAR100(root=path_dataset,
                                        train=False,
                                        transform=transform,
                                        download=True)
        
    # Setup hyperparameters
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = BNC_CNN()
    if( args.pretrained is not None ):
        model.load_state_dict(torch.load(args.pretrained))
    model = model.to(device)
    criterion = nn.BCELoss() #이진분류
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = args.epoch
    
    # Modify proportion of the dataset
    # On train dataset
    if( args.pretrained is None ):
        train_dataset = get_single_subset_by_ratio(train_dataset, args.train_dataset_ratio)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        binarily_noised_train_dataset = BinarilyNoisedDataset(train_loader, noise_type=args.noise_type, intensity_threshold=args.intensity_threshold, margin=args.margin, sigma_reduction=args.sigma_reduction, label_balance=args.label_balance, device=device)    
        binarily_noised_train_loader = DataLoader(binarily_noised_train_dataset, batch_size=args.batch_size, shuffle=True)
    # On test dataset 
    test_dataset = get_single_subset_by_ratio(test_dataset, args.test_dataset_ratio)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    binarily_noised_test_dataset = BinarilyNoisedDataset(test_loader, noise_type=args.noise_type, intensity_threshold=args.intensity_threshold, margin=0, sigma_reduction=0, label_balance=False, device=device) # based on reality : margin=0 & sigma_reduction=0(uniform) & label_balance=False
    binarily_noised_test_loader = DataLoader(binarily_noised_test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Sanity check
    visualize_sample(pilot=False, loader=binarily_noised_test_loader, path=image_file_path)

    #학습률 스케줄링
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    #early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    epoch_loss_rec = []
    max_epoch_loss = math.inf
    
    # Training loop
    if( args.pretrained is None ):
        for epoch in range(num_epochs):
                last_lr = str(scheduler.get_last_lr()[0])
                model.train() #change model's mode to train
                running_loss = 0.0
                for inputs, labels in binarily_noised_train_loader:
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
                
                # Save best model
                if( epoch_loss < max_epoch_loss ):
                    print(f'Model saved: Epoch [{epoch+1}] [Current] {epoch_loss:.4f} << {max_epoch_loss:.4f} [Max]')
                    torch.save(model.state_dict(), model_file_path)
                    best_model_state = model.state_dict()
                    best_epoch_idx = epoch + 1
                    max_epoch_loss = epoch_loss
                
                # 학습률 감소 스케줄러에 검증 손실 전달
                if args.lr_scheduler:
                    scheduler.step(epoch_loss)

                    if len(last_lr) < len(str(scheduler.get_last_lr()[0])):
                        last_lr = str(scheduler.get_last_lr()[0])
                        with open(meta_file_path, 'a') as file:
                            file.write(f'\nlearning rate changed to {last_lr} at Epoch {epoch + 1}')
                    print(f'\nlearning rate changed to {last_lr} at Epoch {epoch + 1}')
                    
                    # 학습률 확인 및 출력
                    if args.verbose:
                        print(f"Learning rate after epoch {epoch + 1}: {scheduler.get_last_lr()[0]}")
                
                if( epoch % 10 == 9 ):
                    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
                
                epoch_loss_rec.append(epoch_loss)

                if args.early_stopping:
                    early_stopping(epoch_loss, model)
                    if early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch+1} (epoch loss:{epoch_loss:.4f})")
                        with open(meta_file_path, 'a') as file:
                            file.write(f"\nEarly stopping at epoch {epoch+1} (epoch loss:{epoch_loss:.4f})")
                        break
        
        # Load model parameters at best epoch loss
        if args.verbose:
            print(f'Load model state at best epoch loss [{best_epoch_idx}]')
        model.load_state_dict(best_model_state)
        visualize_epoch_loss(pilot=False, epoch_loss=epoch_loss_rec, path=loss_file_path)
    
    # Evaluation
    model.eval()
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in binarily_noised_test_loader:
            labels = labels.float().unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    
    visualize_confusion_matrix(pilot=False, all_labels=all_labels, all_predictions=all_predictions, num_classes=2, path=accuracy_file_path)

    accuracies, precisions, recalls, f1_scores = get_classification_metrics(all_labels, all_predictions)
    print(f'new Accuracy: {accuracies*100:.2f}%')

    accuracy_summary_record = {'xid': xid,
                               'accuracy': accuracy,
                               'dataset_type': args.dataset_type,
                               'noise_type': args.noise_type,
                               'intensity_threshold': args.intensity_threshold,
                               'margin': args.margin,
                               'sigma_reduction': args.sigma_reduction,
                               'label_balance': args.label_balance,
                               'memo': args.memo,
                              }
    save_record_to_csv(accuracy_summary_csv_file_path, accuracy_summary_record)

    for class_idx, precision, recall, f1_score in zip(list(range(2)), precisions, recalls, f1_scores):
        accuracy_record = {'class': class_idx,
                           'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
                           'dataset_type': args.dataset_type,
                           'noise_type': args.noise_type,
                           'train_dataset_ratio': args.train_dataset_ratio,
                           'test_dataset_ratio': args.test_dataset_ratio,
                           'batch_size': args.batch_size,
                           'epoch': args.epoch,
                           'margin': args.margin,
                           'sigma_reduction': args.sigma_reduction,
                           'label_balance': args.label_balance
                          }
        save_record_to_csv(accuracy_csv_file_path, accuracy_record)
        print(f'Class [{class_idx}] | Pr.: {precision:.6f} | Re.: {recall:.6f} | F1.: {f1_score:.6f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Command-line arguments
    parser.add_argument('-d', '--dataset_type', type=str, required=False, default='cifar10', choices=['cifar10','cifar100'])
    parser.add_argument('-n', '--noise_type', type=str, required=False, default='gaussian', choices=['gaussian', 'snp', 'uniform', 'poisson'])
    parser.add_argument('--train_dataset_ratio', type=restricted_float, required=False, default=1.0)
    parser.add_argument('--test_dataset_ratio', type=restricted_float, required=False, default=1.0)
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64)
    parser.add_argument('-e', '--epoch', type=int, required=False, default=50)
    parser.add_argument('-p', '--pretrained', type=str, required=False, default=None)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    parser.add_argument('--intensity_threshold', type=restricted_float, required=False, default=0.5)
    parser.add_argument('--margin', type=restricted_float, required=False, default=0.0)
    parser.add_argument('--sigma_reduction', type=int, required=False, default=0)
    parser.add_argument('--label_balance', action='store_true', default=False)
    parser.add_argument('--username', type=str, required=False, default=None)
    parser.add_argument('--output_path_meta', type=str, required=False, default=path_result_meta)
    parser.add_argument('--output_path_image', type=str, required=False, default=path_result_image)
    parser.add_argument('--output_path_model', type=str, required=False, default=path_result_model)
    parser.add_argument('--output_path_loss', type=str, required=False, default=path_result_loss)
    parser.add_argument('--output_path_accuracy', type=str, required=False, default=path_result_accuracy)
    parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose mode")
    parser.add_argument('--memo', type=str, required=False, default=None)

    # Parsing arguments
    args = parser.parse_args()

    # Sanity check; directory existence
    ensure_directory(path_dataset)
    ensure_directory(path_result)
    ensure_directory(path_result_root)
    ensure_directory(args.output_path_meta)
    ensure_directory(args.output_path_image)
    ensure_directory(args.output_path_model)
    ensure_directory(args.output_path_loss)
    ensure_directory(args.output_path_accuracy)

    # Write meta data
    current_time = get_current_time_str()
    xid = get_next_xid(path_result_meta)
    lines = [
        f'datetime: {current_time}',
        f'dataset_type: {args.dataset_type}',
        f'noise_type: {args.noise_type}',
        f'train_dataset_ratio: {args.train_dataset_ratio}',
        f'test_dataset_ratio: {args.test_dataset_ratio}',
        f'batch_size: {args.batch_size}',
        f'epoch: {args.epoch}',
        f'pretrained: {args.pretrained}',
        f'early_stopping: {args.early_stopping}',
        f'lr_scheduler: {args.lr_scheduler}',
        f'intensity_threshold: {args.intensity_threshold}',
        f'margin: {args.margin}',
        f'sigma_reduction: {args.sigma_reduction}',
        f'label_balance: {args.label_balance}',
        f'username: {args.username}',
        f'output_path_meta: {args.output_path_meta}',
        f'output_path_image: {args.output_path_image}',
        f'output_path_model: {args.output_path_model}',
        f'output_path_loss: {args.output_path_loss}',
        f'output_path_accuracy: {args.output_path_accuracy}',        
        f'verbose: {args.verbose}',
        f'memo: {args.memo}'
    ]
    
    meta_file_path = f'{args.output_path_meta}/{xid:03d}_{exp_no}_meta_{args.dataset_type}_{args.noise_type}_{current_time}.txt'
    image_file_path = f'{args.output_path_image}/{xid:03d}_{exp_no}_sample_{args.dataset_type}_{args.noise_type}_{current_time}.png'
    model_file_path = f'{args.output_path_model}/{xid:03d}_{exp_no}_model_{args.dataset_type}_{args.noise_type}_{current_time}.weights'
    loss_file_path = f'{args.output_path_loss}/{xid:03d}_{exp_no}_loss_{args.dataset_type}_{args.noise_type}_{current_time}.png'
    accuracy_file_path = f'{args.output_path_accuracy}/{xid:03d}_{exp_no}_accuarcy_{args.dataset_type}_{args.noise_type}_{current_time}.png'
    accuracy_csv_file_path = args.output_path_accuracy + f'{exp_no}_accuracy.csv'
    accuracy_summary_csv_file_path = args.output_path_accuracy + f'{exp_no}_accuracy_summary.csv'

    # Sanity check: Print meta data
    if args.verbose:
        print(f"xid : {xid}")
        print("## Meta data ##")
        for line in lines:
            print(line)
        print(meta_file_path)
        print(image_file_path)
        print(model_file_path)
        print(loss_file_path)
        print(accuracy_file_path)
        print(accuracy_csv_file_path)
        print("#####")
    
    with open(meta_file_path, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    
    # Execution
    try:
        start_time = time.time()
        main(args)
        write_metadata_status(meta_file_path, 'SUCCESS')
        print("SUCCESS")
    except KeyboardInterrupt:
        write_metadata_status(meta_file_path, 'HALTED')
        print("HALTED")
    except Exception as e:
        import sys
        import traceback
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