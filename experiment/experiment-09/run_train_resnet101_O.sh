#!/bin/bash

python regular_exp09.py \
-d cifar10 \
-m resnet101 \
-o sgd \
--noise_training \
-b 256 \
-e 200 \
--early_stopping \
--lr_scheduler \
--username hwkang \
--verbose \
--metrics_csv_file_path $JUPYTER_ROOT/result/exp09/metrics/temp_resnet101_train.csv
