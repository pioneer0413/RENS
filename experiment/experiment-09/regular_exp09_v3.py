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

from model.learning_utility import EalryStopping

#
# Specifics
#

#
# Main
#

#
# Init
#
if __name__=='__main__':
    parser = RensArgumentParser()
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
    output_format = 'csv'
    argument_info = parser.get_argument_info(output_format=output_format)
    meta_data = [str(xid)] + [current_time] + argument_info
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
    path_file_image = os.path.join(path_image, f'{xid:04d}_{exp_no}_image_{args.prefix}{}{args.suffix}.png')
    path_file_loss = os.path.join(path_loss, f'{xid:04d}_{exp_no}_loss_{args.prefix}{}{args.suffix}.png')
    path_file_meta = os.path.join(path_meta, f'{xid:04d}_{exp_no}_meta_{args.prefix}{}{args.suffix}.txt')
    path_file_metrics = os.path.join(path_metrics, f'{xid:04d}_{exp_no}_metrics_{args.prefix}{}{args.suffix}.csv')
    path_file_model = os.path.join(path_model, f'{xid:04d}_{exp_no}_model_{args.prefix}{}{args.suffix}.weights')
    ##*

    ### 메타 데이터 저장