#!/bin/bash

# 데이터셋 배열을 선언합니다.
datasets=("cifar10" "cifar100")

# 노이즈 유형 배열을 선언합니다.
noise_types=("gaussian" "snp" "uniform" "poisson")

# 중첩된 루프를 사용하여 모든 조합에 대해 명령을 실행합니다.
for dataset in "${datasets[@]}"
do
    for noise in "${noise_types[@]}"
    do
        echo "실행 중: python evaluation_exp16.py -d $dataset -n $noise -c 20 -e 1000 --early_stopping --lr_scheduler --username mwkim"
        python evaluation_exp16.py -d $dataset -n $noise -c 20 -e 1000 --early_stopping --lr_scheduler --username mwkim
    done
done

echo "모든 작업이 완료되었습니다."