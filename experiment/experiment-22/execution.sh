#!/bin/bash

# 데이터셋 배열을 선언합니다.
#datasets=("mnist")

# 인코딩 방법을 선언합니다.

: << 'EOF'
encoding=("--loss_function ce_count_loss()" "--encode rate --num_steps 10 --loss_function ce_count_loss()" "--encode latency --num_steps 10 --loss_function ce_max_membrane_loss()")

EOF

noise=("gaussian" "snp" "uniform" "poisson")
: << 'EOF'
encoding=(
"--memo improved" "--loss_function ce_count_loss() --memo improved_count"
"--encode rate --num_steps 10 --memo improved" "--encode rate --num_steps 10 --loss_function ce_count_loss() --memo improved_count"
 "--encode latency --num_steps 10 --memo improved" "--encode latency --num_steps 10 --loss_function ce_max_membrane_loss() --memo improved_membrane"
)
EOF
encoding=("--encode latency --num_steps 10 --memo improved" "--encode latency --num_steps 10 --loss_function ce_max_membrane_loss() --memo improved")

# 에포크 수를 지정합니다.
epoch=1000

# 조기종료 여부를 지정합니다.
early='--early_stopping'

# 학습률 조정 여부를 지정합니다.
lr='--lr_scheduler'

intensities=("0.0" "0.3" "0.5" "0.7" "0.8" "0.9")

# 중첩된 루프를 사용하여 모든 조합에 대해 명령을 실행합니다.

for intensity in "${intensities[@]}"
do
    for noise_type in "${noise[@]}"
    do
        for encode in "${encoding[@]}"
        do
            for i in {1..1}
            do
                echo "실행 중: $encode"
                python improved_evaluation_exp22.py -n $noise_type --intensity $intensity $encode -e $epoch $early $lr --username mwkim --single_gpu 1
            done
        done
    done
done

echo "모든 작업이 완료되었습니다."