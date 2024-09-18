#!/bin/bash

# [prefix][body][suffix]: [model_type][noise_type][intensity]

# At now, this is for generate model

transform_type=$1
n_events=$2
time_jitter_std=$3
pretrained=$4

# CNN
for i in {1..1}
do
echo "$i/5"
python run_classification.py \
--device_id 0 \
-m cnn \
--beta 0.5 \
-b 100 \
-e 3 \
--write_to meta metrics \
--output_format csv \
--username hwkang \
--prefix cnn_ \
--body ${transform_type}_ \
--suffix ${2}_${3} \
-d nmnist \
--transform_type ${transform_type} \
--save \
--verbose \
--n_events ${2} \
--time_jitter_std ${3} \
--pretrained /home/hwkang/jupyter/root/experiment/experiment-09/weights/nmnist_snntorch_cnn_no_train.pt \
--disable_training
done
