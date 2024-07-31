# Experiment no.10 evaluation

from pathlib import Path
exp_no = 'exp10' # NOTICE: Change at each experiment!
directory = current_file_path = Path(__file__).resolve()
path_root = str(directory.parent.parent.parent) + '/'
#path_dataset = path_root + 'dataset/' # NOTICE: Change before running this script! 
path_result = path_root + 'result/' # NOTICE: Change before running this script!
path_result_root = path_result + exp_no + '/'
"""
path_result_image = path_result_root + 'image/'
path_result_model = path_result_root + 'model/'
path_result_loss = path_result_root + 'loss/'
path_result_accuracy = path_result_root + 'accuracy/'
"""
path_result_meta = path_result_root + 'meta/'
path_result_perf = path_result_root + 'perf/'
path_utility = path_root + 'utility/'

# Imports
import sys
sys.path.append(path_root)
#sys.path.append(path_dataset)
sys.path.append(path_result)

import pyRAPL
import torch
from snntorch import spikegen
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt

# Utilities
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

from utility.common import *
#from utility.preprocess import *
#from utility.synthesize import *
from utility.visualize import *
#from model.SimpleCNN import *
#from model.LearningUtils import *

def main(args):
    pyRAPL.setup()

    csv_output = pyRAPL.outputs.CSVOutput(perf_file_path)

    meter = pyRAPL.Measurement(label=args.encoding_type, output=csv_output)

    # Generate random data
    if( args.encoding_type == 'delta' ):
        x_in = torch.randn(args.num_step, args.input_size)
        # Measure perf
        for i in range(args.num_iter):
            meter.begin()
            x_out = spikegen.delta(x_in, threshold=1)
            meter.end()
            meter.export(csv_output)
            csv_output.save()
    
    elif( args.encoding_type == 'ttfs'):
        x_in = torch.randn(args.input_size)
        max_val, min_val = x_in.max(), x_in.min()
        x_in = ( x_in - min_val ) / ( max_val - min_val ) # Normalization
        # Measure perf
        for i in range(args.num_iter):
            meter.begin()
            x_out = spikegen.latency(x_in, num_steps=args.num_step)
            meter.end()
            meter.export(csv_output)
            csv_output.save()
    
    elif( args.encoding_type == 'stime'):
        x_in = torch.randn(args.input_size)
        # Measure perf
        for i in range(args.num_iter):
            meter.begin()
            x_out = spikegen.latency_code(x_in, num_steps=args.num_step)
            meter.end()
            meter.export(csv_output)
            csv_output.save()
    
    elif( args.encoding_type == 'srate'):
        x_in = torch.randn(args.input_size)
        # Measure perf
        for i in range(args.num_iter):
            meter.begin()
            x_out = spikegen.rate(x_in, num_steps=args.num_step)
            meter.end()
            meter.export(csv_output)
            csv_output.save()

    print("DONE")    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Command-line arguments
    parser.add_argument('--encoding_type', type=str, required=True, choices=['delta', 'ttfs', 'stime', 'srate'])
    parser.add_argument('--input_size', type=int, required=False, default=784)
    parser.add_argument('--num_step', type=int, required=False, default=50)
    parser.add_argument('--num_iter', type=int, required=False, default=1000)
    
    # Not in use at exp10
    """
    parser.add_argument('-d', '--dataset_type', type=str, required=False, default='cifar10', choices=['cifar10','cifar100'])
    parser.add_argument('-n', '--noise_type', type=str, required=False, default='multi', choices=['gaussian', 'snp', 'uniform', 'poisson', 'multi']) # noise_type
    parser.add_argument('--min_intensity', type=restricted_float, required=False, default=0.05)
    parser.add_argument('--train_dataset_ratio', type=restricted_float, required=False, default=1.0) # train_dataset_size
    parser.add_argument('--test_dataset_ratio', type=restricted_float, required=False, default=1.0) # test_dataset_size
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64) # batch_size
    parser.add_argument('-e', '--epoch', type=int, required=False, default=50) # epoch
    parser.add_argument('-p', '--pretrained', type=str, required=False, default=None)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    """
    parser.add_argument('--username', type=str, required=False, default=None)
    parser.add_argument('--output_path_meta', type=str, required=False, default=path_result_meta)
    #parser.add_argument('--output_path_image', type=str, required=False, default=path_result_image)
    #parser.add_argument('--output_path_model', type=str, required=False, default=path_result_model)
    #parser.add_argument('--output_path_loss', type=str, required=False, default=path_result_loss)
    #parser.add_argument('--output_path_accuracy', type=str, required=False, default=path_result_accuracy)
    parser.add_argument('--output_path_perf', type=str, required=False, default=path_result_perf)
    parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose mode")
    
    
    # Parsing arguments
    args = parser.parse_args()

    ensure_directory(path_result)
    ensure_directory(args.output_path_meta)
    ensure_directory(args.output_path_perf)
    
    current_time = get_current_time_str()
    xid = get_next_xid(path_result_meta)
    lines = [
        f'datetime: {current_time}',
        f'encoding_type: {args.encoding_type}',
        f'input_size: {args.input_size}',
        f'num_step: {args.num_step}',
        f'num_iter: {args.num_iter}',
        f'username: {args.username}',
        f'output_path_meta: {args.output_path_meta}',
        f'output_path_perf: {args.output_path_perf}',
        f'verbose: {args.verbose}'
    ]

    meta_file_path = f'{args.output_path_meta}/{xid:03d}_{exp_no}_meta_{args.encoding_type}_{args.input_size}_{args.num_step}_{current_time}.txt'
    perf_file_path = f'{args.output_path_perf}/{xid:03d}_{exp_no}_perf_{args.encoding_type}_{args.input_size}_{args.num_step}_{current_time}.csv'

    if args.verbose:
        print("###############################")
        print("########## Meta data ##########")
        print("###############################")
        for line in lines:
            print(line)
        print("######################################")
        print("########## Output filenames ##########")
        print("######################################")
        print(meta_file_path)
        print(perf_file_path)
        print("##########################################")
        print("########## End of the meta data ##########")
        print("##########################################\n")

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