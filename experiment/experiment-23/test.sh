# 데이터셋 배열을 선언합니다.
datasets=("cifar10")

# 노이즈 유형 배열을 선언합니다.
noise_types=("gaussian" "snp" "uniform" "poisson")

# 0으로 분류할 비율을 지정합니다.
intensity_threshold=("0.7" "0.8" "0.9")

# trim 비율을 지정합니다.
margins=("0.01" "0.025" "0.05")

# sigma_reduction 값을 지정합니다.
sigma_reduction=("0" "3") #0 : 균등분포, 1이상 : 정규분포

balanceTF=("T" "F")

# 에포크 수를 지정합니다.
epoch=1000

# 조기종료 여부를 지정합니다.
early='--early_stopping'

# 학습률 조정 여부를 지정합니다.
lr='--lr_scheduler'

# 중첩된 루프를 사용하여 모든 조합에 대해 명령을 실행합니다.

echo "intensity_threshold noise_type margins sigma_reduction balanceTF"
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
                            echo "$intensity $noise $margin $sigma $balance"
                        done
                    done
                done
            done
            sleep 5
            echo "재개 5초전"
            sleep 5
        done
    done
done