#!/bin/bash

pretrained_path=$1
type_of_dataset=$2
type_of_model=$3
csv_filename=$4

# 범용
for i in {1..20}
do
    for j in 0 25 50 75 100
    do
        intensity=$(echo "scale=2; $j/100" | bc)
        echo "X/O: [$i/20] [$j] "
        python regular_exp09.py \
        -d $type_of_dataset \
        -b 256 \
        -m $type_of_model \
        --noise_test \
        --noise_intensity $intensity \
        --metrics_csv_file_path $JUPYTER_ROOT/result/exp09/metrics/${csv_filename}_$(printf "%03d" $j).csv \
        --pretrained $pretrained_path \
        --username hwkang
    done
done