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

class Net(nn.Module):
    def __init__(self, beta, spike_grad):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64*4*4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, device):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.reset_mem().to(device)
        mem2 = self.lif2.reset_mem().to(device)
        mem3 = self.lif3.reset_mem().to(device)

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem3

#
# Main Implementation
#
def main(args):
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if( args.dataset_type == 'cifar10' ):
        train_dataset = datasets.CIFAR10(root=path_dataset,
                                         train=True,
                                         transform=transform,
                                         download=False)
        test_dataset = datasets.CIFAR10(root=path_dataset,
                                        train=False,
                                        transform=transform,
                                        download=False)
                                      
    else:
        train_dataset = datasets.CIFAR100(root=path_dataset,
                                         train=True,
                                         transform=transform,
                                         download=False)
        test_dataset = datasets.CIFAR100(root=path_dataset,
                                        train=False,
                                        transform=transform,
                                        download=False)

    test_dataset[0]

    # Scale dataset size
    if( args.train_dataset_ratio < 1.0 ):
        train_dataset = get_single_subset_by_ratio(train_dataset, ratio=args.train_dataset_ratio)
    if( args.test_dataset_ratio < 1.0 ):
        test_dataset = get_single_subset_by_ratio(test_dataset, ratio=args.test_dataset_ratio)

    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size

    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # Hyperparameters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if( args.model == 'resnet50' ):
        if(args.dataset_type =='cifar10'):
            model = resnet50(weights=None, num_classes=10)
        else:
            model = resnet50(weights=None, num_classes=100)
    elif( args.model == 'resnet101' ):
        if(args.dataset_type =='cifar10'):
            model = resnet101(weights=None, num_classes=10)
        else:
            model = resnet101(weights=None, num_classes=100)
    elif( args.model == 'snn'):
        pass
    if( args.pretrained is not None ):
        model.load_state_dict(torch.load(args.pretrained))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if( args.optimizer == 'adam' ):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif( args.optimizer == 'sgd' ):
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                             weight_decay=5e-4)

    num_epochs = args.epoch
        
    # Setup record variables
    epoch_loss_rec = []
    val_loss_rec = []
    max_epoch_loss = math.inf

    # Learning rate scehduling
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scheduler = CosineAnnealingLR(optimizer, T_max=200)

    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

    # Training loop
    if( args.pretrained is None ):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Noise injection
                if( args.noise_training is True ):
                    trigger = torch.rand(1).item()
                    if( trigger >= 0.66 ):
                        inputs = generate_noisy_batch(inputs, torch.rand(1).item(), device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # After one epoch
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total

            if args.verbose:
                print(f'\nEpoch: [{epoch+1}/{num_epochs}] ###############################################')
                print(
                f'Epoch Loss: {running_loss / len(train_loader.dataset):.4f}, '
                f'Validation Loss: {val_loss / len(valid_loader.dataset):.4f}, '
                f'Validation Accuracy: {val_accuracy:.2f}%')

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_loss_rec.append(epoch_loss)
            val_loss_rec.append(val_loss)
            
            # Improved?
            if( epoch_loss < max_epoch_loss ):
                if( args.verbose ):
                    print(f'Model saved: [Last] {max_epoch_loss:.4f} >> {epoch_loss:.4f} [Current]')
                best_model_state = model.state_dict()
                torch.save(best_model_state, model_file_path)
                best_epoch_idx = epoch + 1
                max_epoch_loss = epoch_loss
                
            # Lr changed?
            if args.lr_scheduler:
                scheduler.step(epoch_loss)
                if args.verbose:
                    print(f"Learning rate after epoch: {scheduler.get_last_lr()[0]}")
                
            # Early stop?
            if args.early_stopping:
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    if args.verbose:
                        print(f"Early stopping with: (epoch loss:{epoch_loss:.4f})")
                        print(f'#############################################################')
                    break
            

        # After training loop
        if args.verbose:
            print(f'\nLoad model state at best epoch loss [{best_epoch_idx}]\n')
        model.load_state_dict(best_model_state)
        visualize_epoch_loss(pilot=False, epoch_loss=epoch_loss_rec, file_path=loss_file_path)
        
    # Evaluation
    model.eval()
    all_labels = []
    all_predictions = []
    once = False
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Noise injection
            if( args.noise_test ):
                inputs = generate_noisy_batch(inputs, args.noise_intensity, device)

                # Visualize noisy images
                """
                if once == False:
                    samples = [ inputs[i].detach().cpu() for i in range(10) ]
                    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                    for i, ax in enumerate(axes.flat):
                        ax.imshow(samples[i].permute(1,2,0), cmap='gray')
                        ax.axis('off')
                    plt.savefig(image_file_path)
                    once = True
                """
            
            outputs = model(inputs)
            predicted_value, predicted_class = torch.max(outputs, dim=1)
            all_labels.extend(labels.detach().cpu().numpy())
            all_predictions.extend(predicted_class.detach().cpu().numpy())

    accuracy, precision, recall, f1 = calculate_metrics(all_labels, all_predictions)
    print(f'\nAccuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1-Score: {f1:.2f}')

    metrics_record = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1
    }
    save_record_to_csv(args.metrics_csv_file_path, metrics_record)
    """
    visualize_confusion_matrix(pilot=False,
                               all_labels=all_labels,
                               all_predictions=all_predictions,
                               num_label=10 if args.dataset_type=='cifar10' else 100,
                               accuracy=accuracy*100,
                               noise_type=args.noise_type,
                               file_path=confusion_matrix_file_path)
    """
    

#
# Initialization
#
if __name__=='__main__':
    parser = argparse.ArgumentParser() # 불필요한 보일러플레이트 코드이므로 클래스를 선언한 후 상속하도록 변경 >> (v1.0.2)
    
    parser.add_argument('-d', '--dataset_type', type=str, required=False, default='cifar10', 
                        choices=['cifar10','cifar100'])
    parser.add_argument('-m', '--model', type=str, required=False, default='resnet101',
                        choices=['resnet50', 'resnet101', 'snn'])
    parser.add_argument('-o', '--optimizer', type=str, required=False, default='adam',
                        choices=['adam', 'sgd'])
    parser.add_argument('-n', '--noise_type', type=str, required=False, default='gaussian', 
                        choices=['gaussian', 'snp', 'uniform', 'poisson', 'multi', 'none'])
    parser.add_argument('--noise_training', action='store_true', required=False, default=False)
    parser.add_argument('--noise_test', action='store_true', required=False, default=False)
    parser.add_argument('--noise_intensity', type=restricted_float, required=False, default=0.0)
    parser.add_argument('--train_dataset_ratio', type=restricted_float, required=False, default=1.0) # train_dataset_size
    parser.add_argument('--test_dataset_ratio', type=restricted_float, required=False, default=1.0) # test_dataset_size
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64) # batch_size
    parser.add_argument('-e', '--epoch', type=int, required=False, default=50) # epoch
    parser.add_argument('-p', '--pretrained', type=str, required=False, default=None)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    parser.add_argument('--username', type=str, required=False, default=None)
    parser.add_argument('--output_path_meta', type=str, required=False, default=path_result_meta)
    parser.add_argument('--output_path_image', type=str, required=False, default=path_result_image)
    parser.add_argument('--output_path_model', type=str, required=False, default=path_result_model)
    parser.add_argument('--output_path_loss', type=str, required=False, default=path_result_loss)
    parser.add_argument('--output_path_metrics', type=str, required=False, default=path_result_metrics)
    parser.add_argument('--metrics_csv_file_path', type=str, required=True)
    parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose mode")
    parser.add_argument('--notes', type=str, required=False, default=None)

    args = parser.parse_args()

    # Check: directory existence
    ensure_directory(path_dataset)
    ensure_directory(path_result)
    ensure_directory(path_result_root)
    ensure_directory(args.output_path_meta)
    ensure_directory(args.output_path_image)
    ensure_directory(args.output_path_model)
    ensure_directory(args.output_path_loss)
    ensure_directory(args.output_path_metrics)

    current_time = get_current_time_str()
    xid = get_next_xid(args.output_path_meta)
    lines = [
        f'datetime: {current_time}',
        f'dataset_type: {args.dataset_type}',
        f'model: {args.model}',
        f'optimizer: {args.optimizer}',
        f'noise_type: {args.noise_type}',
        f'noise_training: {args.noise_training}',
        f'noise_test: {args.noise_test}',
        f'noise_intensity: {args.noise_intensity}',
        f'train_dataset_ratio: {args.train_dataset_ratio}',
        f'train_dataset_ratio: {args.test_dataset_ratio}',
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
        f'output_path_metrics: {args.output_path_metrics}',
        f'metrics_csv_file_path: {args.metrics_csv_file_path}',        
        f'verbose: {args.verbose}',
        f'notes: {args.notes}'
    ]

    meta_file_path = f'{args.output_path_meta}/{xid:03d}_{exp_no}_meta_{args.dataset_type}_{args.noise_intensity}_{current_time}.txt'
    image_file_path = f'{args.output_path_image}/{xid:03d}_{exp_no}_sample_{args.dataset_type}_{args.noise_intensity}_{current_time}.png'
    model_file_path = f'{args.output_path_model}/{xid:03d}_{exp_no}_model_{args.dataset_type}_{args.noise_intensity}_{current_time}.weights'
    loss_file_path = f'{args.output_path_loss}/{xid:03d}_{exp_no}_loss_{args.dataset_type}_{args.noise_intensity}_{current_time}.png'
    
    # 혼동 행렬을 표현하는 파일명으로 변경 >> (v1.0.2)
    confusion_matrix_file_path = f'{args.output_path_metrics}/{xid:03d}_{exp_no}_confusion_matrix_{args.dataset_type}_{args.noise_intensity}_{current_time}.png' 
    
    # 정확도 지표 뿐만 아니라 향후 다른 지표도 함께 명시한다는 것을 나타내기 위해 metrics를 중심으로 변경 >> (v1.0.2)
    #metrics_csv_file_path = args.output_path_metrics + f'{exp_no}_{args.dataset_type}_{args.noise_intensity}_metrics.csv' 
    
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
        #print(metrics_csv_file_path)
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
        if( args.verbose ): # 에러가 메타 파일 뿐만 아니라 표준 출력 되도록 설정 >> (v1.0.2)
            print(e)
        write_metadata_status(meta_file_path, f'FAILED({e})')
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(meta_file_path, 'a') as file:
            file.write(f'\nTotal elapsed time: {elapsed_time:.6f} s')
        print( f'Total elapsed time: {elapsed_time:.6f} s\n' )