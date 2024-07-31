
for i in {1..100}
do
    echo "Iteration: ${i}"
    
    # CIFAR-10, AWGN
    echo "Running [AWGN, CIFAR-10]"
    python evaluation_exp20.py \
    -d cifar10 \
    -n gaussian \
    -p ../../result/exp20/model/008_exp20_model_cifar10_gaussian_24-07-31_13-58-25.weights \
    --early_stopping \
    --lr_scheduler \
    --username hwkang;

    # CIFAR-10, SnP
    echo "Running [SnP, CIFAR-10]"
    python evaluation_exp20.py \
    -d cifar10 \
    -n snp \
    -p ../../result/exp20/model/009_exp20_model_cifar10_snp_24-07-31_14-00-09.weights \
    --early_stopping \
    --lr_scheduler \
    --username hwkang;
    
    # CIFAR-10, Uniform
    echo "Running [Uniform, CIFAR-10]"
    python evaluation_exp20.py \
    -d cifar10 \
    -n uniform \
    -p ../../result/exp20/model/010_exp20_model_cifar10_uniform_24-07-31_14-01-05.weights \
    --early_stopping \
    --lr_scheduler \
    --username hwkang;
    
    # CIFAR-10, Poisson
    echo "Running [Poisson, CIFAR-10]"
    python evaluation_exp20.py \
    -d cifar10 \
    -n poisson \
    -p ../../result/exp20/model/011_exp20_model_cifar10_poisson_24-07-31_14-03-02.weights \
    --early_stopping \
    --lr_scheduler \
    --username hwkang;

    #

    # CIFAR-100, AWGN
    echo "Running [AWGN, CIFAR-100]"
    python evaluation_exp20.py \
    -d cifar100 \
    -n gaussian \
    -p ../../result/exp20/model/012_exp20_model_cifar100_gaussian_24-07-31_14-05-36.weights \
    --early_stopping \
    --lr_scheduler \
    --username hwkang;
    
    # CIFAR-100, SnP
    echo "Running [SnP, CIFAR-100]"
    python evaluation_exp20.py \
    -d cifar100 \
    -n snp \
    -p ../../result/exp20/model/013_exp20_model_cifar100_snp_24-07-31_14-07-04.weights \
    --early_stopping \
    --lr_scheduler \
    --username hwkang;
    
    # CIFAR-100, Uniform
    echo "Running [Uniform, CIFAR-100]"
    python evaluation_exp20.py \
    -d cifar100 \
    -n uniform \
    -p ../../result/exp20/model/014_exp20_model_cifar100_uniform_24-07-31_14-07-57.weights \
    --early_stopping \
    --lr_scheduler \
    --username hwkang;
    
    # CIFAR-100, Poisson
    echo "Running [Poisson, CIFAR-100]"
    python evaluation_exp20.py \
    -d cifar100 \
    -n poisson \
    -p ../../result/exp20/model/015_exp20_model_cifar100_poisson_24-07-31_14-10-00.weights \
    --early_stopping \
    --lr_scheduler \
    --username hwkang;
done