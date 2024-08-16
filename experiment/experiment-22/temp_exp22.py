# Experiment no.22 evaluation

# Constants
from pathlib import Path
exp_no = 'exp22' # NOTICE: Change at each experiment!
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
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

# Utilities
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import itertools

from utility.common import *
from utility.preprocess import *
from utility.synthesize import *
from utility.visualize import *
from model.SimpleCNN import *
from model.LearningUtils import *

'''
START of model# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10
'''
class SNN(nn.Module):
    def __init__(self, inputs=28*28, hidden=1000, outputs=10, beta=0.95):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(inputs, hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden, outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(args.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
'''
END of model
'''


def main(args):

    # Main implementation

    # Data preprocessing
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    if (args.dataset_type=='mnist'):
        train_dataset = datasets.MNIST(root=path_dataset,
                                       train=True,
                                       transform=transform,
                                       download=True)
        test_dataset = datasets.MNIST(root=path_dataset,
                                       train=False,
                                       transform=transform,
                                       download=True)
    else:
        print()
        #TBA

    # Modify proportion of the dataset
    # On train dataset
    if( args.pretrained is None ):
        train_dataset = get_subset(train_dataset, args.train_dataset_ratio)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    # On test dataset 
    test_dataset = get_subset(test_dataset, args.test_dataset_ratio)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Sanity check
    visualize_noisy_sample(pilot=False, loader=test_loader, file_path=image_file_path)

    # Setup hyperparameters
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = SNN()
    if( args.pretrained is not None ):
        model.load_state_dict(torch.load(args.pretrained))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() # 다중분류
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    num_epochs = args.epoch

    epoch_loss_rec = []
    max_epoch_loss = math.inf

    # 학습률 스케줄링
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    # Training loop
    if( args.pretrained is None ):
        for epoch in range(num_epochs):
            # learning rate update (before goes)
            last_lr = str(scheduler.get_last_lr()[0])
            train_batch = iter(train_loader)

            model.train() # change model's mode to train
            running_loss = 0.0
        
            for inputs, labels in train_batch:
                inputs, labels = inputs.to(device), labels.to(device)
                #print(inputs.shape)

                # 옵티마이저 초기화
                optimizer.zero_grad()

                # 순전파
                model.train() #change model's mode to train
                spk_rec, mem_rec = model(inputs.view(inputs.size(0), -1))
                #spk_rec, mem_rec = model(inputs.view(args.batch_size, -1))

                # 손실 계산
                loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(args.num_steps):
                    loss += criterion(mem_rec[step], labels)

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
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 순전파
            outputs, _ = model(inputs.view(inputs.size(0), -1))
            # 손실 계산
            _, predicted_class = outputs.sum(dim=0).max(1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())
            total += labels.size(0)
            correct += (predicted_class == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    visualize_confusion_matrix(pilot=False, all_labels=all_labels, all_predictions=all_predictions, num_label=10, noise_type="None", accuracy=accuracy, file_path=accuracy_file_path)

    precisions, recalls, f1_scores = calculate_confusion_metrics(all_labels, all_predictions, num_class=10)

    for class_idx, precision, recall, f1_score in zip(list(range(args.classes)), precisions, recalls, f1_scores):
        accuracy_record = {'class': class_idx,
                           'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
                           'dataset_type': args.dataset_type,
                           'train_dataset_ratio': args.train_dataset_ratio,
                           'test_dataset_ratio': args.test_dataset_ratio,
                           'batch_size': args.batch_size,
                           'epoch': args.epoch}
        save_record_to_csv(accuracy_csv_file_path, accuracy_record)
        print(f'Class [{class_idx}] | Pr.: {precision:.6f} | Re.: {recall:.6f} | F1.: {f1_score:.6f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Command-line arguments
    parser.add_argument('-d', '--dataset_type', type=str, required=False, default='mnist', choices=['mnist'])
    parser.add_argument('--train_dataset_ratio', type=restricted_float, required=False, default=1.0) # train_dataset_size
    parser.add_argument('--test_dataset_ratio', type=restricted_float, required=False, default=1.0) # test_dataset_size
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64) # batch_size
    parser.add_argument('-e', '--epoch', type=int, required=False, default=50) # epoch
    parser.add_argument('-p', '--pretrained', type=str, required=False, default=None)
    parser.add_argument('--num_steps', type=int, required=False, default=10)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
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
    
    meta_file_path = f'{args.output_path_meta}/{xid:03d}_{exp_no}_meta_{args.dataset_type}_{current_time}.txt'
    image_file_path = f'{args.output_path_image}/{xid:03d}_{exp_no}_sample_{args.dataset_type}_{current_time}.png'
    model_file_path = f'{args.output_path_model}/{xid:03d}_{exp_no}_model_{args.dataset_type}_{current_time}.weights'
    loss_file_path = f'{args.output_path_loss}/{xid:03d}_{exp_no}_loss_{args.dataset_type}_{current_time}.png'
    accuracy_file_path = f'{args.output_path_accuracy}/{xid:03d}_{exp_no}_accuarcy_{args.dataset_type}_{current_time}.png'
    accuracy_csv_file_path = args.output_path_accuracy + f'{exp_no}_accuracy.csv'
    accuracy_summary_csv_file_path = args.output_path_accuracy + f'{exp_no}_accuracy_summary.csv'

    # Sanity check: Print meta data
    if args.verbose:
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
        write_metadata(meta_file_path, 'SUCCESS')
        print("SUCCESS")
    except KeyboardInterrupt:
        write_metadata(meta_file_path, 'HALTED')
        print("HALTED")
    except Exception as e:
        write_metadata(meta_file_path, f'FAILED({e})')
        print(f"FAILED({e})")
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(meta_file_path, 'a') as file:
            file.write(f'\nTotal elapsed time: {elapsed_time:.6f} s')