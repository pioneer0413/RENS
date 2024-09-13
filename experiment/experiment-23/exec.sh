#!/bin/bash
: << 'EOF'
    parser.add_argument('-d', '--dataset_type', type=str, required=False, default='cifar10', choices=['cifar10','cifar100'])
    parser.add_argument('-n', '--noise_type', type=str, required=False, default='gaussian', choices=['gaussian', 'snp', 'uniform', 'poisson'])
    parser.add_argument('--train_dataset_ratio', type=restricted_float, required=False, default=1.0) # train_dataset_size
    parser.add_argument('--test_dataset_ratio', type=restricted_float, required=False, default=1.0) # test_dataset_size
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64) # batch_size
    parser.add_argument('-e', '--epoch', type=int, required=False, default=50) # epoch
    parser.add_argument('-p', '--pretrained', type=str, required=False, default=None)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    parser.add_argument('--intensity_threshold', type=restricted_float, required=False, default=0.5)
    parser.add_argument('--margin', type=restricted_float, required=False, default=0.0)
    parser.add_argument('--sigma_reduction', type=int, required=False, default=0)
    parser.add_argument('--label_balance', action='store_true', default=False)
    parser.add_argument('--username', type=str, required=False, default=None)
    parser.add_argument('--output_path_meta', type=str, required=False, default=path_result_meta)
    parser.add_argument('--output_path_image', type=str, required=False, default=path_result_image)
    parser.add_argument('--output_path_model', type=str, required=False, default=path_result_model)
    parser.add_argument('--output_path_loss', type=str, required=False, default=path_result_loss)
    parser.add_argument('--output_path_accuracy', type=str, required=False, default=path_result_accuracy)
    parser.add_argument('--verbose', action='store_true', default=False, help="Enable verbose mode")
    parser.add_argument('--memo', type=str, required=False, default=None)
EOF
# 데이터셋 배열을 선언합니다.
datasets=("cifar10")

# 노이즈 유형 배열을 선언합니다.
noise_types=("gaussian" "snp" "uniform" "poisson")

# 0으로 분류할 비율을 지정합니다.
intensity_threshold=("0.6" "0.7" "0.8" "0.9")

# trim 비율을 지정합니다.
margins=("0.025" "0.05" "0.075")

# sigma_reduction 값을 지정합니다.
sigma_reduction=("0" "1" "2" "3" "4" "5") #0 : 균등분포, 1이상 : 정규분포

balanceTF=("")
#balanceTF=("--label_balance" "")

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
            for margin in "${margins[@]}"
            do
                for sigma in "${sigma_reduction[@]}"
                do
                    for balance in "${balanceTF[@]}"
                    do
                        for i in {1..10}
                        do
                            echo "실행 중: python evaluation_exp23.py -d $dataset -n $noise --intensity_threshold $intensity --margin $margin -e $epoch $early $lr --sigma_reduction $sigma $balance --username mwkim"
                            python evaluation_exp23.py -d $dataset -n $noise --intensity_threshold $intensity --margin $margin -e $epoch $early $lr --sigma_reduction $sigma $balance --username mwkim --memo "CHOOSE of margin & sigma_reduction"
                        done
                    done
                done
            done
        done
    done
done

echo "모든 작업이 완료되었습니다."