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

# Utilities
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import itertools
import traceback

from utility.common import *
from utility.preprocess import *
from utility.synthesize import *
from utility.visualize import *
from model.SimpleCNN import *
from model.LearningUtils import *

'''
START of synthesize
'''
#import torch
#from torch.utils.data import Dataset
#import numpy as np
import random

####### NEED TO UPDATE!!! (FROM V1.0.2) ########
def generate_variously_noised_image(image: torch.Tensor, gaussian: float=0.0, snp: float=0.0, uniform: float=0.0, poisson: float=0.0):
    np_original_image = image.numpy()
    c, w, h = np_image.shape

    np_image = generate_gaussian(image=np_image, c=c, w=w, h=h, intensity=gaussian)
    np_image = generate_salt_and_pepper(image=np_image, intensity=snp)
    np_image = generate_uniform(image=np_image, intensity=uniform)
    np_image = generate_poisson(image=np_image, c=c, w=w, h=h, intensity=poisson)
    
    max_val, min_val = np_image.max(), np_image.min()
    normalized_np_image = ( np_image - min_val )/( max_val - min_val )
    noisy_image = torch.from_numpy(normalized_np_image)
    noisy_image = noisy_image.float()
    
    return noisy_image


class RandomlyNoisedDataset(Dataset):
    def __init__(self, data_loader):
        self.x = []
        self.y = []

        for image, label in data_loader:
            image = image.squeeze(0)
            self.x.append(generate_variously_noised_image(image, gaussian=np.random.rand(), snp=np.random.rand(), uniform=np.random.rand(), poisson=np.random.rand()))
            self.y.append(label)
                    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_data = self.x[idx]
        y_data = self.y[idx]
        return x_data, y_data

def snn_training_loop(args, model, train_loader, modified_train_loader, device, scheduler):
    epoch_loss_rec = []
    max_epoch_loss = math.inf
    for epoch in range(args.num_epochs):
        last_lr = str(scheduler.get_last_lr()[0])
        model.train() #change model's mode to train
        running_loss = 0.0
        for inputs, labels in modified_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 옵티마이저 초기화
            optimizer.zero_grad()
            
            # 순전파
            match args.encode:
                case 'rate':
                    inputs = spikegen.rate(inputs, num_steps=args.num_steps)
                case 'latency':
                    inputs = spikegen.latency(inputs, tau=0.1, num_steps=args.num_steps)
            inputs = inputs.view(args.num_steps, inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3))
            outputs = model(inputs)
            
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
    visualize_epoch_loss(pilot=False, epoch_loss=epoch_loss_rec, file_path=loss_file_path)

def snn_evaluation(args, model, test_loader, modified_test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in modified_test_loader:
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
    
    visualize_confusion_matrix(pilot=False, all_labels=all_labels, all_predictions=all_predictions, num_label=2, noise_type=args.noise_type, accuracy=accuracy, file_path=accuracy_file_path)

    #return precisions, recalls, f1_scores
    return get_classification_metrics(all_labels, all_predictions, None)

'''
END of synthesize
'''

def main(args):

    # Main implementation

    # Data preprocessing
    transform = transforms.Compose([transforms.ToTensor()])
    
    if ( args.dataset_type == 'mnist' ):
        train_dataset = datasets.MNIST(root=PATH_DATASET,
                                       train=True,
                                       transform=transform,
                                       download=True)
        test_dataset = datasets.MNIST(root=PATH_DATASET,
                                       train=False,
                                       transform=transform,
                                       download=True)
    elif( args.dataset_type == 'cifar10' ):
        train_dataset = datasets.CIFAR10(root=PATH_DATASET,
                                         train=True,
                                         transform=transform,
                                         download=True)
        test_dataset = datasets.CIFAR10(root=PATH_DATASET,
                                        train=False,
                                        transform=transform,
                                        download=True)
                                      
    elif( args.dataset_type == 'cifar100' ):
        train_dataset = datasets.CIFAR100(root=PATH_DATASET,
                                         train=True,
                                         transform=transform,
                                         download=True)
        test_dataset = datasets.CIFAR100(root=PATH_DATASET,
                                        train=False,
                                        transform=transform,
                                        download=True)

    # Modify proportion of the dataset
    # On train dataset
    if( args.pretrained is None ):
        train_dataset = get_single_subset_by_ratio(train_dataset, args.train_dataset_ratio)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
        modified_train_dataset = RandomlyNoisedDataset(train_loader)
        modified_train_loader = DataLoader(modified_train_dataset, batch_size=args.batch_size, shuffle=True)
    # On test dataset 
    test_dataset = get_single_subset_by_ratio(test_dataset, args.test_dataset_ratio)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    modified_test_dataset = RandomlyNoisedDataset(train_loader)
    modified_test_loader = DataLoader(modified_test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Sanity check
    visualize_noisy_sample(pilot=False, loader=modified_test_loader, file_path=image_file_path)

    # Setup hyperparameters
    dtype = torch.float
    device = (
        torch.device(f"cuda:{args.single_gpu}") if (torch.cuda.is_available() and args.single_gpu != None) else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    snn_model = CNV_SNN()
    if( args.pretrained is not None ):
        model.load_state_dict(torch.load(args.pretrained))
    snn_model = snn_model.to(device)
    snn_criterion = None ####TODO : after exp22, choose one
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = args.epoch

    epoch_loss_rec = []
    max_epoch_loss = math.inf

    #학습률 스케줄링
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    #early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

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
        visualize_epoch_loss(pilot=False, epoch_loss=epoch_loss_rec, file_path=loss_file_path)
    
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
    
    visualize_confusion_matrix(pilot=False, all_labels=all_labels, all_predictions=all_predictions, num_label=2, noise_type=args.noise_type, accuracy=accuracy, file_path=accuracy_file_path)

    precisions, recalls, f1_scores = get_classification_metrics(all_labels, all_predictions, None)

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
    parser.add_argument('-d', '--dataset_type', type=str, required=False, default='mnist', choices=['mnist','cifar10','cifar100'])
    parser.add_argument('--train_dataset_ratio', type=restricted_float, required=False, default=1.0)
    parser.add_argument('--test_dataset_ratio', type=restricted_float, required=False, default=1.0)
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64)
    parser.add_argument('-e', '--epoch', type=int, required=False, default=50)
    parser.add_argument('-p', '--pretrained', type=str, required=False, default=None)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    parser.add_argument('--username', type=str, required=False, default=None)
    parser.add_argument('--output_path_meta', type=str, required=False, default=PATH_RESULT_META)
    parser.add_argument('--output_path_image', type=str, required=False, default=PATH_RESULT_IMAGE)
    parser.add_argument('--output_path_model', type=str, required=False, default=PATH_RESULT_MODEL)
    parser.add_argument('--output_path_loss', type=str, required=False, default=PATH_RESULT_LOSS)
    parser.add_argument('--output_path_accuracy', type=str, required=False, default=PATH_RESULT_ACCURACY)
    parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose mode")
    parser.add_argument('--memo', type=str, required=False, default=None)

    # Parsing arguments
    args = parser.parse_args()

    # Sanity check; DIRECTORY existence
    ensure_DIRECTORY(PATH_DATASET)
    ensure_DIRECTORY(PATH_RESULT)
    ensure_DIRECTORY(PATH_RESULT_ROOT)
    ensure_DIRECTORY(args.output_path_meta)
    ensure_DIRECTORY(args.output_path_image)
    ensure_DIRECTORY(args.output_path_model)
    ensure_DIRECTORY(args.output_path_loss)
    ensure_DIRECTORY(args.output_path_accuracy)

    # Write meta data
    current_time = get_current_time_str()
    xid = get_next_xid(PATH_RESULT_META)
    lines = [
        f'datetime: {current_time}',
        f'dataset_type: {args.dataset_type}',
        f'train_dataset_ratio: {args.train_dataset_ratio}',
        f'test_dataset_ratio: {args.test_dataset_ratio}',
        f'batch_size: {args.batch_size}',
        f'epoch: {args.epoch}',
        f'pretrained: {args.pretrained}',
        f'early_stopping: {args.early_stopping}',
        f'lr_scheduler: {args.lr_scheduler}',
        f'username: {args.username}',
        f'output_path_meta: {args.output_path_meta}',
        f'output_path_image: {args.output_path_image}',
        f'output_path_model: {args.output_path_model}',
        f'output_path_loss: {args.output_path_loss}',
        f'output_path_accuracy: {args.output_path_accuracy}',
        f'verbose: {args.verbose}',
        f'memo: {args.memo}'
    ]
    
    meta_file_path = f'{args.output_path_meta}/{xid:03d}_{EXP_NO}_meta_{args.dataset_type}_{args.noise_type}_{current_time}.txt'
    image_file_path = f'{args.output_path_image}/{xid:03d}_{EXP_NO}_sample_{args.dataset_type}_{args.noise_type}_{current_time}.png'
    model_file_path = f'{args.output_path_model}/{xid:03d}_{EXP_NO}_model_{args.dataset_type}_{args.noise_type}_{current_time}.weights'
    loss_file_path = f'{args.output_path_loss}/{xid:03d}_{EXP_NO}_loss_{args.dataset_type}_{args.noise_type}_{current_time}.png'
    accuracy_file_path = f'{args.output_path_accuracy}/{xid:03d}_{EXP_NO}_accuarcy_{args.dataset_type}_{args.noise_type}_{current_time}.png'
    accuracy_csv_file_path = args.output_path_accuracy + f'{EXP_NO}_accuracy.csv'

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