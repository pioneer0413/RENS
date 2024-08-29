#!/bin/bash

pretrained_path=$1
csv_filename=$2
model_type=$3

# 범용

# SpikingCNN
python regular_exp09.py \
-d cifar10 \
--training_dataset_ratio 1.0 \
--test_dataset_ratio 1.0 \
-b 64 \
-e 200 \
-o adam \
--early_stopping \
--lr_scheduler \
--write_to meta loss model metrics confusion_matrix \
--verbose \
--username hwkang \
--body SpikingCNN \
-m snn3 \
--num_steps 50 \
--beta 0.99 \
--output_format string

# CNN 잡음 미훈련
python regular_exp09.py \
-d cifar10 \
--training_dataset_ratio 1.0 \
--test_dataset_ratio 1.0 \
-b 64 \
-e 200 \
-o adam \
--early_stopping \
--lr_scheduler \
--write_to meta loss model metrics confusion_matrix \
--verbose \
--username hwkang \
--body cnn_x \
-m cnn \
--num_steps 50 \
--beta 0.99 \
--output_format string

# CNN 잡음 훈련
python regular_exp09.py \
-d cifar10 \
--training_dataset_ratio 1.0 \
--test_dataset_ratio 1.0 \
-b 64 \
-e 200 \
-o adam \
--early_stopping \
--lr_scheduler \
--write_to meta loss model metrics confusion_matrix \
--verbose \
--username hwkang \
--body cnn_o \
-m cnn \
--num_steps 50 \
--beta 0.9 \
--output_format string \
--noise_training
######################################################################

# SpikingCNN 90 NMNIST
python regular_exp09.py \
-d nmnist \
--training_dataset_ratio 1.0 \
--test_dataset_ratio 1.0 \
-b 32 \
-e 2 \
-o adam \
--early_stopping \
--lr_scheduler \
--write_to meta loss model metrics confusion_matrix \
--verbose \
--username hwkang \
--body SpikingCNN_90_NMNIST \
-m snn3 \
--num_steps 50 \
--beta 0.90 \
--output_format string \
--device_id 0

# CNN X NMNIST
python regular_exp09.py \
-d nmnist \
--training_dataset_ratio 1.0 \
--test_dataset_ratio 1.0 \
-b 64 \
-e 2 \
-o adam \
--early_stopping \
--lr_scheduler \
--write_to meta loss model metrics confusion_matrix \
--verbose \
--username hwkang \
--body cnn_x \
-m cnn \
--num_steps 50 \
--beta 0.90 \
--output_format string \
--device_id 0

# CNN O NMNIST
python regular_exp09.py \
-d nmnist \
--training_dataset_ratio 1.0 \
--test_dataset_ratio 1.0 \
-b 64 \
-e 2 \
-o adam \
--early_stopping \
--lr_scheduler \
--write_to meta loss model metrics confusion_matrix \
--verbose \
--username hwkang \
--body cnn_o \
-m cnn \
--num_steps 50 \
--beta 0.90 \
--output_format string \
--device_id 1 \
--noise_training

######################################################################
# SpikingCNN 90
bash test.sh \
$JUPYTER_ROOT/result/exp09/model/3360_exp09_model_SpikingCNN.weights \
SpikingCNN_90 \
snn3 \
10

# SpikingCNN 99
bash test.sh \
$JUPYTER_ROOT/result/exp09/model/3367_exp09_model_SpikingCNN.weights \
SpikingCNN_99 \
snn3 \
3

# CNN X
bash test.sh \
$JUPYTER_ROOT/result/exp09/model/3361_exp09_model_cnn_x.weights \
cnn_x \
cnn \
3

# CNN O N-MNIST
bash test.sh \
$JUPYTER_ROOT/result/exp09/model/3365_exp09_model_cnn_o.weights \
cnn_o \
cnn \
3