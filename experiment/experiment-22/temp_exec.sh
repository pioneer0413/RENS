#!/bin/bash

# 데이터셋 배열을 선언합니다.
datasets=("mnist")

# 에포크 수를 지정합니다.
epoch=1000

# 조기종료 여부를 지정합니다.
early='--early_stopping'

# 학습률 조정 여부를 지정합니다.
lr='--lr_scheduler'

# 중첩된 루프를 사용하여 모든 조합에 대해 명령을 실행합니다.

for dataset in "${datasets[@]}"
do
    for i in {1..1}
    do
        echo "실행 중: $dataset $noise $class $intensity $trim $epoch $early"
        python temp_exp22.py -d $dataset -e $epoch $early $lr --username mwkim --memo "test"
    done
done

echo "모든 작업이 완료되었습니다."