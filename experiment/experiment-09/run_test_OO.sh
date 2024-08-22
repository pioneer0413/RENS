#!/bin/bash

# O / O
for i in {1..20}
do
    for j in 0 25 50 75 100
    do
        intensity=$(echo "scale=2; $j/100" | bc)
        echo "O/O: [$i/20] [$j] "
        python regular_exp09.py \
        -d cifar10 \
        -b 256 \
        -m resnet50 \
        --noise_test \
        --noise_intensity $intensity \
        --metrics_csv_file_path $JUPYTER_ROOT/result/exp09/metrics/OO_F_$(printf "%03d" $j).csv \
        --pretrained $JUPYTER_ROOT/result/exp09/model/434_exp09_model_cifar10_0.0_24-08-21_10-55-12.weights \
        --username hwkang
    done
done