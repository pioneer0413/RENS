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
from snntorch import surrogate
from snntorch import utils
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
START of synthesization
'''
def calculate_psnr_torch(image1, image2):
    """
    Calculate PSNR using PyTorch tensors.
    """
    mse = torch.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming image is normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def add_gaussian_noise_torch(image, mean=0.0, std=1.0):
    """
    Add Gaussian noise to a PyTorch tensor image.
    """
    gaussian_noise = torch.randn_like(image) * std + mean
    noisy_image = image + gaussian_noise
    return torch.clamp(noisy_image, 0.0, 1.0)

def add_salt_and_pepper_noise_torch(image, amount=0.05, salt_vs_pepper=0.5):
    """
    Add salt and pepper noise to a PyTorch tensor image.
    """
    noisy_image = image.clone()
    
    num_salt = torch.ceil(torch.tensor(amount * image.numel() * salt_vs_pepper)).int()
    num_pepper = torch.ceil(torch.tensor(amount * image.numel() * (1.0 - salt_vs_pepper))).int()

    # Add salt noise (white pixels)
    indices = torch.randperm(image.numel(), device=image.device)[:num_salt]
    noisy_image.view(-1)[indices] = 1

    # Add pepper noise (black pixels)
    indices = torch.randperm(image.numel(), device=image.device)[:num_pepper]
    noisy_image.view(-1)[indices] = 0

    return noisy_image

def add_uniform_noise_torch(image, low=-0.1, high=0.1):
    """
    Add uniform noise to a PyTorch tensor image.
    """
    # Adjust 'low' and 'high' to be within the [0, 1] range
    low = max(low, -image.min().item())
    high = min(high, 1 - image.max().item())
    
    uniform_noise = torch.empty_like(image).uniform_(low, high)
    noisy_image = image + uniform_noise
    return torch.clamp(noisy_image, 0.0, 1.0)

def add_poisson_noise_torch(image):
    """
    Add Poisson noise to a PyTorch tensor image.
    """
    noisy_image = torch.poisson(image * 255) / 255.0
    return torch.clamp(noisy_image, 0.0, 1.0)

def adjust_combined_noise_to_psnr_torch(image, target_psnr=30, tolerance=0.5):
    """
    Adjusts the combined noise level to achieve a target PSNR for the image using iterative adjustment.
    
    Args:
    - image (torch.Tensor): Input image tensor with values in range [0, 1].
    - target_psnr (float): Target PSNR value for the combined noise image.
    - tolerance (float): Acceptable deviation from the target PSNR.
    
    Returns:
    - noisy_image_combined (torch.Tensor): Image with combined noises adjusted to target PSNR.
    """
    std_gaussian = 0.0
    amount_sp = 0.0
    high_uniform = 0.0

    current_psnr = float('inf')

    while current_psnr > target_psnr + tolerance:
        # Gradually increase noise levels to reduce PSNR
        std_gaussian += 0.1
        amount_sp += 0.1
        high_uniform += 0.1

        # Apply combined noise
        noisy_image_combined = (
            add_gaussian_noise_torch(image, std=std_gaussian) + 
            add_salt_and_pepper_noise_torch(image, amount=amount_sp) + 
            add_uniform_noise_torch(image, low=-high_uniform, high=high_uniform) + 
            add_poisson_noise_torch(image)
        ) / 4
        
        current_psnr = calculate_psnr_torch(image, noisy_image_combined)
        print(f"PSNR: {current_psnr:.2f} dB, std_gaussian: {std_gaussian:.2f}, amount_sp: {amount_sp:.2f}, high_uniform: {high_uniform:.2f}")

    return noisy_image_combined

def get_combined_noised_image(image, intensity):
    return (
            add_gaussian_noise_torch(image, std=intensity) + 
            add_salt_and_pepper_noise_torch(image, amount=intensity) + 
            add_uniform_noise_torch(image, low=-intensity, high=intensity) + 
            add_poisson_noise_torch(image)
            ) / 4

class SpecificallyNoisedDataset(Dataset):
    def __init__(self, data_loader, device, noise_type='gaussian', intensity=0.5):
        self.x=[]
        self.y=[]

        for image, label in data_loader:
            image, label = image.to(device), label.to(device)
            image, label = image.squeeze(0), label.squeeze(0)
            #noisy_image = adjust_combined_noise_to_psnr_torch(image, target_psnr=15)
            noisy_image = get_combined_noised_image(image, 0.7)
            self.x.append(noisy_image)
            #self.x.append(generate_noisy_data(data=image, intensity=intensity, noise_type=noise_type, rescale=True, normalize=True, device=device))
            self.y.append(label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_data = self.x[idx]
        y_data = self.y[idx]
        return x_data, y_data
        
'''
END of synthesization
'''

'''
START of model# Network Architecture
'''
# SNN 모델 정의
class CNV_SNN(nn.Module):
    def __init__(self, time_steps):
        super(CNV_SNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  # [16, 24, 24]
        self.bn1 = BatchNormTT2d(16, time_steps=time_steps, momentum=0.05, eps=1e-4)  # BatchNorm 추가
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # [32, 20, 20]
        self.bn2 = BatchNormTT2d(32, time_steps=time_steps, momentum=0.05, eps=1e-4)  # BatchNorm 추가
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 3 * 3, 10)  # Fully connected layer
        
        # snntorch의 Leaky Integrate and Fire (LIF) 뉴런을 사용
        if args.loss_function is None:
            self.lif1 = snn.Leaky(beta=0.9, init_hidden=True)
            self.lif2 = snn.Leaky(beta=0.9, init_hidden=True)
            self.lif3 = snn.Leaky(beta=0.9, init_hidden=True)
        else:
            self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True)
            self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True)
            self.lif3 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True)
    
    def forward(self, x, encoding):
        # 결과를 누적하기 위한 텐서 초기화 / (might be) encoded input => x.size(0) is prohibited
        # for pytorch loss function
        if args.loss_function is None:
            outputs = torch.zeros(x.size(1), 10, device=x.device)

        spk3_rec = []

        if encoding == 'latency':
            for step in range(args.num_steps):
                x_step = self.conv1(x[step])
                x_step = self.bn1[step](x_step)
                spk1 = self.lif1(x_step)
                spk1 = self.pool1(spk1)

                x_step = self.conv2(spk1)
                x_step = self.bn2[step](x_step)
                spk2 = self.lif2(x_step)
                spk2 = self.pool2(spk2)

                spk2 = self.flatten(spk2)
                spk3 = self.lif3(self.fc1(spk2))
                spk3_rec.append(spk3)
                if spk3.any():
                    for _ in range(args.num_steps - step - 1):
                        spk3_rec.append(torch.zeros_like(spk3))
                    break

        else:
            for step in range(args.num_steps):
                x_step = self.conv1(x[step])
                x_step = self.bn1[step](x_step)
                spk1 = self.lif1(x_step)
                spk1 = self.pool1(spk1)

                x_step = self.conv2(spk1)
                x_step = self.bn2[step](x_step)
                spk2 = self.lif2(x_step)
                spk2 = self.pool2(spk2)

                spk2 = self.flatten(spk2)
                spk3 = self.lif3(self.fc1(spk2))
                spk3_rec.append(spk3)

        # for pytorch loss function
        if args.loss_function is None:
            outputs = torch.stack(spk3_rec, dim=1).sum(dim=1)  # 타임 스텝 차원을 합쳐서 최종 출력 계산
        else:
            outputs = torch.stack(spk3_rec)
            
        return outputs
"""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk3_rec = []

        if encoding == 'latency':
            for step in range(args.num_steps):
                spk1, mem1 = self.lif1(self.conv1(x[step]), mem1)
                spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
                spk2 = spk2.view(spk2.size(0), -1)
                spk3, mem3 = self.lif3(self.fc1(spk2), mem3)
                spk3_rec.append(spk3)
                if spk3.any():
                    for _ in range(args.num_steps - step - 1):
                        spk3_rec.append(torch.zeros_like(spk3))
                    break

        else:
            for step in range(args.num_steps):
                spk1, mem1 = self.lif1(self.conv1(x[step]), mem1)
                spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
                spk2 = spk2.view(spk2.size(0), -1)
                spk3, mem3 = self.lif3(self.fc1(spk2), mem3)
                spk3_rec.append(spk3)

        # for pytorch loss function
        if args.loss_function is None:
            outputs = torch.stack(spk3_rec, dim=1).sum(dim=1)  # 타임 스텝 차원을 합쳐서 최종 출력 계산
        else:
            outputs = torch.stack(spk3_rec)
            
        return outputs
"""
'''
END of model
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
    
    # Setup hyperparameters
    dtype = torch.float
    device = (
        torch.device(f"cuda:{args.single_gpu}") if (torch.cuda.is_available() and args.single_gpu != None) else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    model = CNV_SNN(args.num_steps)
    if( args.pretrained is not None ):
        model.load_state_dict(torch.load(args.pretrained))
    model = model.to(device)
    # to(device) 후 `DataParallel`로 모델 래핑
    if (torch.cuda.device_count() > 1) and (args.single_gpu is None):
        model = nn.DataParallel(model)
    match args.loss_function:
        case 'ce_count_loss()':
            criterion = SF.ce_count_loss()
        case 'ce_max_membrane_loss()':
            criterion = SF.ce_max_membrane_loss()
        case _:
            criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    num_epochs = args.epoch

    # Modify proportion of the dataset
    # On train dataset
    if( args.pretrained is None ):
        if args.verbose is True:
            print("train data is on way")
        train_dataset = get_single_subset_by_ratio(train_dataset, args.train_dataset_ratio)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, drop_last=True)
        #modified_train_dataset = SpecificallyNoisedDataset(train_loader, device, args.noise_type, args.intensity)
        modified_train_dataset = train_dataset
        modified_train_loader = DataLoader(dataset=modified_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        if args.verbose is True:
            print("train data is now ready")
    # On test dataset 
    if args.verbose is True:
        print("test data is on way")
    test_dataset = get_single_subset_by_ratio(test_dataset, args.test_dataset_ratio)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=True)
    modified_test_dataset = SpecificallyNoisedDataset(test_loader, device, args.noise_type, args.intensity)
    modified_test_loader = DataLoader(dataset=modified_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    if args.verbose is True:
        print("test data is now ready")
    
    # Sanity check
    visualize_sample(pilot=False, loader=modified_test_loader, path=image_file_path)

    epoch_loss_rec = []
    max_epoch_loss = math.inf

    # 학습률 스케줄링
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # early stopping
    early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

    if (args.encode is not None) and (args.num_steps == 1):
        print("[WARNING] Poor Accuracy: When num_steps is 1, rate encoding will not provide qualified input (and output) / latency coding will always emit spikes so input is no longer valuable.")

    # Training loop
    if( args.pretrained is None ):
        for epoch in range(num_epochs):
            # learning rate update (before goes)
            last_lr = str(scheduler.get_last_lr()[0])
            model.train() # change model's mode to train
            running_loss = 0.0
        
            for inputs, labels in modified_train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 옵티마이저 초기화
                optimizer.zero_grad()

                # Reset hidden states
                utils.reset(model)

                # 순전파
                match args.encode:
                    case 'rate':
                        enc_inputs = spikegen.rate(inputs, num_steps=args.num_steps)
                    case 'latency':
                        enc_inputs = spikegen.latency(inputs, tau=0.1, num_steps=args.num_steps)
                    case _:
                        # 스파이크는 아니나, 변수 통일을 위함
                        enc_inputs = inputs
                enc_inputs = enc_inputs.view(args.num_steps, inputs.size(0), 1, 28, 28)
                output = model(enc_inputs, args.encode)

                # 손실 계산
                loss = criterion(output, labels)

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
    eval_time = time.time()
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in modified_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 모델에 데이터 전달
            match args.encode:
                case 'rate':
                    enc_inputs = spikegen.rate(inputs, num_steps=args.num_steps)
                case 'latency':
                    enc_inputs = spikegen.latency(inputs, tau=0.1, num_steps=args.num_steps)
                case _:
                    # 스파이크는 아니나, 변수 통일을 위함
                    enc_inputs = inputs
            enc_inputs = enc_inputs.view(args.num_steps, inputs.size(0), 1, 28, 28)
            output = model(enc_inputs, args.encode)
            if args.loss_function is None:
                _, predicted_class = output.max(1)
    
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted_class.cpu().numpy())
            else:
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(output.sum(dim=0).argmax(dim=1).cpu().numpy())
    eval_time = time.time() - eval_time
    
    visualize_confusion_matrix(pilot=False, all_labels=all_labels, all_predictions=all_predictions, num_classes=10, path=accuracy_file_path)

    accuracy, precisions, recalls, f1_scores = get_classification_metrics(all_labels, all_predictions)
    print(f"Accuracy: {accuracy*100:.2f}")

    accuracy_summary_record = {'xid': xid,
                               'accuracy': accuracy,
                               'dataset_type': args.dataset_type,
                               'noise_type': args.noise_type,
                               'encode': args.encode,
                               'memo': args.memo,
                               'intensity': args.intensity,
                               'time': eval_time
                              }
    save_record_to_csv(accuracy_summary_csv_file_path, accuracy_summary_record)

    for class_idx, precision, recall, f1_score in zip(list(range(10)), precisions, recalls, f1_scores):
        accuracy_record = {'class': class_idx,
                           'accuracy': accuracy, 'precision': precision,
                           'recall': recall, 'f1_score': f1_score,
                           'dataset_type': args.dataset_type,
                           'noise_type': args.noise_type,
                           'encode': args.encode,
                           'train_dataset_ratio': args.train_dataset_ratio,
                           'test_dataset_ratio': args.test_dataset_ratio,
                           'num_steps': args.num_steps,
                           'batch_size': args.batch_size,
                           'epoch': args.epoch,
                           'intensity': args.intensity
                          }
        save_record_to_csv(accuracy_csv_file_path, accuracy_record)
        print(f'Class [{class_idx}] | Pr.: {precision:.6f} | Re.: {recall:.6f} | F1.: {f1_score:.6f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Command-line arguments
    parser.add_argument('-d', '--dataset_type', type=str, required=False, default='mnist', choices=['mnist'])
    parser.add_argument('--encode', type=str, required=False, default=None, choices=['rate','latency'])
    parser.add_argument('--train_dataset_ratio', type=restricted_float, required=False, default=1.0) # train_dataset_size
    parser.add_argument('--test_dataset_ratio', type=restricted_float, required=False, default=1.0) # test_dataset_size
    parser.add_argument('-n', '--noise_type', type=str, required=False, default='gaussian', choices=['gaussian', 'snp', 'uniform', 'poisson'])
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64) # batch_size
    parser.add_argument('-e', '--epoch', type=int, required=False, default=50) # epoch
    parser.add_argument('-p', '--pretrained', type=str, required=False, default=None)
    parser.add_argument('--num_steps', type=int, required=False, default=1, help="Set this each When Encode: num_steps>1 / No encode: num_steps=1")
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
    parser.add_argument('--single_gpu', type=int, required=False, default=None, help="Enable singleGPU mode (only when its GPU is available / no input: parallel mode)")
    parser.add_argument('--intensity', type=restricted_float, required=False, default=0.5)
    parser.add_argument('--loss_function', type=str, required=False, default=None)

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
        f'encode: {args.encode}',
        f'train_dataset_ratio: {args.train_dataset_ratio}',
        f'test_dataset_ratio: {args.test_dataset_ratio}',
        f'noise_type: {args.noise_type}',
        f'batch_size: {args.batch_size}',
        f'epoch: {args.epoch}',
        f'pretrained: {args.pretrained}',
        f'num_steps: {args.num_steps}',
        f'early_stopping: {args.early_stopping}',
        f'lr_scheduler: {args.lr_scheduler}',
        f'username: {args.username}',
        f'output_path_meta: {args.output_path_meta}',
        f'output_path_image: {args.output_path_image}',
        f'output_path_model: {args.output_path_model}',
        f'output_path_loss: {args.output_path_loss}',
        f'output_path_accuracy: {args.output_path_accuracy}',        
        f'verbose: {args.verbose}',
        f'memo: {args.memo}',
        f'single_gpu: {args.single_gpu}',
        f'intensity: {args.intensity}',
        f'loss_function: {args.loss_function}'
    ]
    
    meta_file_path = f'{args.output_path_meta}/{xid:03d}_{exp_no}_meta_{args.dataset_type}_{args.encode}_{current_time}.txt'
    image_file_path = f'{args.output_path_image}/{xid:03d}_{exp_no}_sample_{args.dataset_type}_{args.encode}_{current_time}.png'
    model_file_path = f'{args.output_path_model}/{xid:03d}_{exp_no}_model_{args.dataset_type}_{args.encode}_{current_time}.weights'
    loss_file_path = f'{args.output_path_loss}/{xid:03d}_{exp_no}_loss_{args.dataset_type}_{args.encode}_{current_time}.png'
    accuracy_file_path = f'{args.output_path_accuracy}/{xid:03d}_{exp_no}_accuarcy_{args.dataset_type}_{args.encode}_{current_time}.png'
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