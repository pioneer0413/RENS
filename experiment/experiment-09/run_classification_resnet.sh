#!/bin/bash

# [prefix][body][suffix]: [model_type][noise_type][intensity]

model_type=$1
transform_type=$2
n_events=$3
std=$4

for i in {1..5}
do
echo "$i/5"
python run_classification.py \
--device_id 0 \
-m $model_type \
--beta 0.5 \
-b 100 \
-e 3 \
--write_to meta metrics \
--output_format csv \
--username hwkang \
--prefix ${model_type}_ \
--body ${transform_type}_ \
--suffix ${n_events}_${std} \
-d nmnist \
--transform_type $transform_type \
--save \
--verbose \
--n_events $n_events \
--time_jitter_std $std \
#--pretrained /home/hwkang/jupyter/root/experiment/experiment-09/weights/nmnist_snntorch_cnn_noise_train.pt \
#--disable_training
done
