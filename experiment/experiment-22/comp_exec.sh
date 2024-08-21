#!/bin/bash

# 데이터셋 배열을 선언합니다.
datasets=("mnist")

# 인코딩 방법을 선언합니다.
encoding=("" "--encode rate --num_steps 10" "--encode latency --num_steps 10")

# 에포크 수를 지정합니다.
epoch=1000

# 조기종료 여부를 지정합니다.
early='--early_stopping'

# 학습률 조정 여부를 지정합니다.
lr='--lr_scheduler'

# 중첩된 루프를 사용하여 모든 조합에 대해 명령을 실행합니다.

for dataset in "${datasets[@]}"
do
    for encode in "${encoding[@]}"
    do
        for i in {1..1}
        do
            echo "실행 중: $encoding"
            python comp_exp22.py -d $dataset $encode -e $epoch $early $lr --username mwkim --memo "comp" --single_gpu 1
        done
    done
done

echo "모든 작업이 완료되었습니다."