#!/bin/bash

pretrained_path=$1
csv_filename=$2
model_type=$3
num_iters=${4:-5}

# 범용
for i in $(seq 1 $num_iters)
do
    for j in 1 3 5 7 9 15 25 30 35
    do
        #intensity=$(echo "scale=2; $j/100" | bc)
        intensity=$j
        echo "X: [$i/$num_iters] [$j]"
        python regular_exp09.py \
        -d nmnist \
        --pretrained $pretrained_path \
        -b 64 \
        --disable_training \
        --write_to meta metrics\
        --username hwkang \
        --body ${csv_filename} \
        -m $model_type \
        --noise_test \
        --noise_intensity $intensity \
        --output_format csv \
        --path_file_metrics $JUPYTER_ROOT/result/exp09/metrics/${csv_filename}_$(printf "%03d" $j).csv
    done
done