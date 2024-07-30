python evaluation.py -p ../../result/exp15/model/015_cifar10_gaussian_24-07-24_17-11-03.pt -n gaussian -s 1e-3
echo ""
python evaluation.py -p ../../result/exp15/model/016_cifar10_snp_24-07-24_17-15-57.pt -n snp -s 1e-3
echo ""
python evaluation.py -p ../../result/exp15/model/017_cifar10_uniform_24-07-24_17-20-50.pt -n uniform -s 1e-3
echo ""
python evaluation.py -p ../../result/exp15/model/018_cifar10_poisson_24-07-24_17-25-45.pt -n poisson -s 1e-3