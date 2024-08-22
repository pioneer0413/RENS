# Noise training / Noise test

# X / X (baseline) # 0
python regular_exp09.py -d cifar10 -m resnet50 -o sgd -n gaussian -b 256 -e 200 --early_stopping --lr_scheduler --username hwkang --notes "X/X"

# O / X # 1
python regular_exp09.py -d cifar10 -m resnet50 -o sgd -n gaussian -b 256 -e 200 --noise_training --early_stopping --lr_scheduler --username hwkang --notes "O/X"

# X / O # 2~4
python regular_exp09.py -d cifar10 -m resnet50 -o sgd -n gaussian -b 256 -e 200 --noise_test --noise_intensity 0.25 --early_stopping --lr_scheduler --username hwkang --notes "X/O[25]"
python regular_exp09.py -d cifar10 -m resnet50 -o sgd -n gaussian -b 256 -e 200 --noise_test --noise_intensity 0.50 --early_stopping --lr_scheduler --username hwkang --notes "X/O[50]"
python regular_exp09.py -d cifar10 -m resnet50 -o sgd -n gaussian -b 256 -e 200 --noise_test --noise_intensity 0.75 --early_stopping --lr_scheduler --username hwkang --notes "X/O[75]"

# O / O # 5~7
python regular_exp09.py -d cifar10 -m resnet50 -o sgd -n gaussian -b 256 -e 200 --noise_training --noise_test --noise_intensity 0.25 --early_stopping --lr_scheduler --username hwkang --notes "O/O[25]"
python regular_exp09.py -d cifar10 -m resnet50 -o sgd -n gaussian -b 256 -e 200 --noise_training --noise_test --noise_intensity 0.50 --early_stopping --lr_scheduler --username hwkang --notes "O/O[50]"
python regular_exp09.py -d cifar10 -m resnet50 -o sgd -n gaussian -b 256 -e 200 --noise_training --noise_test --noise_intensity 0.75 --early_stopping --lr_scheduler --username hwkang --notes "O/O[75]"
