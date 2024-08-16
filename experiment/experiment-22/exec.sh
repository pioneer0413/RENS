#!/bin/bash

# 데이터셋 배열을 선언합니다.
datasets=("cifar10")

# 노이즈 유형 배열을 선언합니다.
noise_types=("gaussian" "snp" "uniform" "poisson")

# 0으로 분류할 비율을 지정합니다.
intensity_threshold=("0.5")
#intensity_threshold=("0.6" "0.7" "0.8" "0.9")

# trim 비율을 지정합니다.
trims=("0.01")
#trims=("0.01" "0.02" "0.03" "0.04" "0.05")

# 에포크 수를 지정합니다.
epoch=1000

# 조기종료 여부를 지정합니다.
early='--early_stopping'

# 학습률 조정 여부를 지정합니다.
lr='--lr_scheduler'

# 중첩된 루프를 사용하여 모든 조합에 대해 명령을 실행합니다.

for dataset in "${datasets[@]}"
do
    for intensity in "${intensity_threshold[@]}"
    do
        for noise in "${noise_types[@]}"
        do
            for trim in "${trims[@]}"
            do
                for i in {1..1}
                do
                    echo "실행 중: $dataset $noise $class $intensity $trim $epoch $early"
                    python evaluation_exp23.py -d $dataset -n $noise --intensity_threshold $intensity --trim $trim -e $epoch $early $lr --username mwkim --memo "test"
                done
            done
        done
    done
done

echo "모든 작업이 완료되었습니다."