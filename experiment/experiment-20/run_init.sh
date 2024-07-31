python evaluation_exp20.py \
-d cifar10 \
-n gaussian \
-e 100 \
--early_stopping \
--lr_scheduler \
--username hwkang \
--verbose;

python evaluation_exp20.py \
-d cifar10 \
-n snp \
-e 100 \
--early_stopping \
--lr_scheduler \
--username hwkang \
--verbose;

python evaluation_exp20.py \
-d cifar10 \
-n uniform \
-e 100 \
--early_stopping \
--lr_scheduler \
--username hwkang \
--verbose;

python evaluation_exp20.py \
-d cifar10 \
-n poisson \
-e 100 \
--early_stopping \
--lr_scheduler \
--username hwkang \
--verbose;


python evaluation_exp20.py \
-d cifar100 \
-n gaussian \
-e 100 \
--early_stopping \
--lr_scheduler \
--username hwkang \
--verbose;

python evaluation_exp20.py \
-d cifar100 \
-n snp \
-e 100 \
--early_stopping \
--lr_scheduler \
--username hwkang \
--verbose;

python evaluation_exp20.py \
-d cifar100 \
-n uniform \
-e 100 \
--early_stopping \
--lr_scheduler \
--username hwkang \
--verbose;

python evaluation_exp20.py \
-d cifar100 \
-n poisson \
-e 100 \
--early_stopping \
--lr_scheduler \
--username hwkang \
--verbose
