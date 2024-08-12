#!/bin/bash

# 데이터셋 배열을 선언합니다.
datasets=("cifar100")

# 노이즈 유형 배열을 선언합니다.
noise_types=("gaussian" "snp" "uniform" "poisson")

# 클래스 수를 지정합니다.
#classes=("2")
class=2

# 0으로 분류할 비율을 지정합니다.
min_intensity=("0.5" "0.7" "0.8" "0.9")

# trim 비율을 지정합니다.
trims=("0.1" "0.125" "0.15")

# 에포크 수를 지정합니다.
epoch=1000

# 조기종료 여부를 지정합니다.
early='--early_stopping'

# 학습률 조정 여부를 지정합니다.
lr='--lr_scheduler'

# 중첩된 루프를 사용하여 모든 조합에 대해 명령을 실행합니다.

for dataset in "${datasets[@]}"
do
    for intensity in "${min_intensity[@]}"
    do
        for noise in "${noise_types[@]}"
        do
            for trim in "${trims[@]}"
            do
                for i in {1..10}
                do
                    echo "실행 중: $dataset $noise $class $intensity $trim $epoch $early"
                    python evaluation_exp16.py -d $dataset -n $noise --min_intensity $intensity -c $class --trim $trim -e $epoch $early $lr --username mwkim --memo "trim=0.15 added, on cifar100"
                done
            done
        done
    done
done

echo "모든 작업이 완료되었습니다."