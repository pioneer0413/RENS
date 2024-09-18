#!/bin/bash

# Jitter vs. Uniform

weights=("/home/hwkang/jupyter/root/result/exp09/model/0000_exp09_model_cnn_complex_100_100.pt" "/home/hwkang/jupyter/root/result/exp09/model/0000_exp09_model_cnn_complex_1000_100.pt" "/home/hwkang/jupyter/root/result/exp09/model/0000_exp09_model_cnn_complex_3000_10000.pt")
intensities=("300" "2000" "4000" "8000")
std=("100" "100" "5000" "10000")

for weight in "${weights[@]}"; do
    counter=0
    for intensity in "${intensities[@]}"; do
        echo "$weight"
        echo "$intensity"
        elem3=${std[$((counter))]}
        echo "$elem3"

        # CNN w. uniform
        python run_classification.py \
        --device_id 0 \
        -m cnn \
        --beta 0.5 \
        -b 100 \
        -e 3 \
        --write_to meta metrics \
        --output_format csv \
        --username hwkang \
        --prefix noise_trained_cnn_ \
        --body complex_ \
        --suffix ${intensity}_${elem3} \
        -d nmnist \
        --transform_type complex \
        --save \
        --n_events ${intensity} \
        --time_jitter_std ${elem3} \
        --pretrained $weight \
        --disable_training

        ((counter++))
    done
    
done