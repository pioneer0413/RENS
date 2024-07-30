# Experiment no.19

# Constants
exp_no = 'exp19'
path_root = '/home/hwkang/jupyter/root/'
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

from utility.common import *
from utility.preprocess import *
from utility.synthesize import *
from utility.visualize import *
from model.SimpleCNN import SimpleCNN

def fgsm_attack(model, criterion, images, labels, epsilon):
    # 입력 데이터의 기울기를 추적
    images.requires_grad = True

    # 모델의 예측값 계산
    outputs = model(images)
    loss = criterion(outputs, labels)

    # 모델의 기울기를 0으로 설정하고 역전파 수행
    model.zero_grad()
    loss.backward()

    # 기울기의 부호를 구함
    sign_data_grad = images.grad.data.sign()

    # 입력 데이터에 교란 추가
    perturbed_images = images + epsilon * sign_data_grad

    # 데이터 값을 [0,1] 범위로 클립
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images

def main(args):
    if args.verbose:
        print("Verbose mode is enabled.")

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

    # Modify proportion of the dataset
    # On train dataset
    if( args.pretrained is None ):
        train_dataset = get_subset(train_dataset, args.train_dataset_ratio)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        noised_train_dataset = NoisedDataset(train_loader, noise_type=args.noise_type)    
        noised_train_loader = DataLoader(noised_train_dataset, batch_size=args.batch_size, shuffle=True)

    # On test dataset 
    test_dataset = get_subset(test_dataset, args.test_dataset_ratio)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    noised_test_dataset = NoisedDataset(test_loader, noise_type=args.noise_type)
    noised_test_loader = DataLoader(noised_test_dataset, batch_size=args.batch_size, shuffle=False)

    # Sanity check
    visualize_noisy_sample(pilot=False, loader=noised_test_loader, file_path=image_file_path)

    # Setup hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN()
    if( args.pretrained is not None ):
        model.load_state_dict(torch.load(args.pretrained))
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = args.epoch

    epoch_loss_rec = []
    max_epoch_loss = math.inf

    # Training loop
    if( args.pretrained is None ):
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

            # Save best model
            if( epoch_loss < max_epoch_loss ):
                print(f'Model saved: Epoch [{epoch+1}] [Current] {epoch_loss:.4f} << {max_epoch_loss:.4f} [Max]')
                torch.save(model.state_dict(), model_file_path)
                max_epoch_loss = epoch_loss
                
            if( epoch % 10 == 9 ):
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            epoch_loss_rec.append(epoch_loss)
    
        visualize_epoch_loss(pilot=False, epoch_loss=epoch_loss_rec, file_path=loss_file_path)

    # Generate FGSM dataset
    epsilon = args.epsilon
    fgsm_data = []
    
    for images, labels in noised_test_loader:
        labels = labels.float().unsqueeze(1)
        images, labels = images.to(device), labels.to(device)
        # FGSM perturbed image
        perturbed_images = fgsm_attack(model, criterion, images, labels, epsilon)
        fgsm_data.append((perturbed_images.cpu().detach(), labels.int().cpu().detach()))
    
    # Evaluation
    model.eval()
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in fgsm_data:
            #labels = labels.float().unsqueeze(1)  # 이진 분류를 위해 라벨을 0 또는 1로 변경 (임시로)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    visualize_confusion_matrix(pilot=False, all_labels=all_labels, all_predictions=all_predictions, num_label=2, noise_type=args.noise_type, accuracy=accuracy, file_path=accuracy_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Sanity check; directory existence
    ensure_directory(path_result_meta)
    ensure_directory(path_result_image)
    ensure_directory(path_result_model)
    ensure_directory(path_result_loss)
    ensure_directory(path_result_accuracy)
    
    # Command-line arguments
    parser.add_argument('-d', '--dataset_type', type=str, required=False, default='cifar10', choices=['cifar10','cifar100'])
    parser.add_argument('-n', '--noise_type', type=str, required=False, default='gaussian', choices=['gaussian', 'snp', 'uniform', 'poisson']) # noise_type
    parser.add_argument('-s', '--epsilon', type=float, required=False, default=1e-2) # FGSM intensity
    parser.add_argument('-t', '--train_dataset_ratio', type=float, required=False, default=1.0) # train_dataset_size
    parser.add_argument('-y', '--test_dataset_ratio', type=float, required=False, default=1.0) # test_dataset_size
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64) # batch_size
    parser.add_argument('-e', '--epoch', type=int, required=False, default=50) # epoch
    parser.add_argument('-p', '--pretrained', type=str, required=False, default=None)
    parser.add_argument('-i', '--output_path_image', type=str, required=False, default=path_result_image)
    parser.add_argument('-l', '--output_path_loss', type=str, required=False, default=path_result_loss)
    parser.add_argument('-a', '--output_path_accuracy', type=str, required=False, default=path_result_accuracy)
    parser.add_argument('-m', '--output_path_meta', type=str, required=False, default=path_result_meta)
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Enable verbose mode")
    

    # Parsing arguments
    args = parser.parse_args()

    # Write meta data
    current_time = get_current_time_str()
    xid = get_next_xid(path_result_meta)
    lines = [
        f'datetime: {current_time}',
        f'dataset_type: {args.dataset_type}',
        f'noise_type: {args.noise_type}',
        f'epsilon: {args.epsilon}',
        f'train_dataset_ratio: {args.train_dataset_ratio}',
        f'train_dataset_ratio: {args.test_dataset_ratio}',
        f'batch_size: {args.batch_size}',
        f'epoch: {args.epoch}',
        f'pretrained: {args.pretrained}',
        f'output_path_image: {args.output_path_image}',
        f'output_path_loss: {args.output_path_loss}',
        f'output_path_accuracy: {args.output_path_accuracy}',
        f'output_path_meta: {args.output_path_meta}',
        f'verbose: {args.verbose}'
    ]
    
    meta_file_path = f'{path_result_meta}/{xid:03d}_{exp_no}_meta_{args.dataset_type}_{args.noise_type}_{args.epsilon}_{current_time}.txt'
    image_file_path = f'{path_result_image}/{xid:03d}_{exp_no}_sample_{args.dataset_type}_{args.noise_type}_{args.epsilon}_{current_time}.png'
    model_file_path = f'{path_result_model}/{xid:03d}_{exp_no}_model_{args.dataset_type}_{args.noise_type}_{args.epsilon}_{current_time}.weights'
    loss_file_path = f'{path_result_loss}/{xid:03d}_{exp_no}_loss_{args.dataset_type}_{args.noise_type}_{args.epsilon}_{current_time}.png'
    accuracy_file_path = f'{path_result_accuracy}/{xid:03d}_{exp_no}_accuarcy_{args.dataset_type}_{args.noise_type}_{args.epsilon}_{current_time}.png'
    
    with open(meta_file_path, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    
    # Execution
    try:
        start_time = time.time()
        main(args)
        write_metadata(meta_file_path, 'SUCCESS')
    except KeyboardInterrupt:
        write_metadata(meta_file_path, 'HALTED')
    except Exception as e:
        write_metadata(meta_file_path, f'FAILED({e})')
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(meta_file_path, 'a') as file:
            file.write(f'\nTotal elapsed time: {elapsed_time:.6f} s') 