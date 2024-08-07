#!/bin/bash

# 데이터셋 배열을 선언합니다.
datasets=("cifar10")
# "cifar100")

# 노이즈 유형 배열을 선언합니다.
noise_types=("gaussian" "snp" "uniform" "poisson")

# 클래스 수를 지정합니다.
classes=("3" "5" "7" "10" "20")

# trim 비율을 지정합니다.
trim=0.0

# 에포크 수를 지정합니다.
epoch=1000

# 조기종료 여부를 지정합니다.
early='--early_stopping'

# 학습률 조정 여부를 지정합니다.
lr='--lr_scheduler'

# 중첩된 루프를 사용하여 모든 조합에 대해 명령을 실행합니다.
for dataset in "${datasets[@]}"
do
    for class in "${classes[@]}"
            do
        for noise in "${noise_types[@]}"
        do
            echo "실행 중: $dataset $noise $class $trim $epoch $early $lr"
            python evaluation_exp16.py -d $dataset -n $noise -c $class --trim $trim -e $epoch $early $lr --username mwkim
        done
    done
done

echo "모든 작업이 완료되었습니다."