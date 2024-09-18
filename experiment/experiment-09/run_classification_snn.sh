#!/bin/bash

# [prefix][body][suffix]: [model_type][noise_type][intensity]

# At now, this is for generate model

for i in {1..5}
do
echo "$i/5"
python run_classification.py \
--device_id 0 \
-m snn \
--beta 0.5 \
-b 100 \
-e 3 \
--write_to meta metrics \
--output_format csv \
--username hwkang \
--prefix snn_ \
--body original_ \
--suffix 0_0 \
-d nmnist \
--transform_type original \
--save \
--verbose \
--n_events 0 \
--time_jitter_std 0 \
--pretrained /home/hwkang/jupyter/root/experiment/experiment-09/weights/nmnist_snntorch_snn_no_train.pt \
--disable_training
done
