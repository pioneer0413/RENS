# Input size variation
echo "Input size variation"

python evaluation_exp10.py \
--encoding_type delta \
--input_size 784 ;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 784 ;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 784 ;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 784 ;

python evaluation_exp10.py \
--encoding_type delta \
--input_size 1024 ;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 1024 ;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 1024 ;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 1024 ;

python evaluation_exp10.py \
--encoding_type delta \
--input_size 36864 ;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 36864 ;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 36864 ;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 36864 ;

python evaluation_exp10.py \
--encoding_type delta \
--input_size 307200 ;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 307200 ;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 307200 ;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 307200 ;

# The number of step variation
echo "The number of step variation"

python evaluation_exp10.py \
--encoding_type delta \
--input_size 1024 \
--num_step 25;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 1024 \
--num_step 25;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 1024 \
--num_step 25;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 1024 \
--num_step 25;

python evaluation_exp10.py \
--encoding_type delta \
--input_size 1024 \
--num_step 50;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 1024 \
--num_step 50;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 1024 \
--num_step 50;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 1024 \
--num_step 50;

python evaluation_exp10.py \
--encoding_type delta \
--input_size 1024 \
--num_step 100;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 1024 \
--num_step 100;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 1024 \
--num_step 100;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 1024 \
--num_step 100;

python evaluation_exp10.py \
--encoding_type delta \
--input_size 1024 \
--num_step 1000;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 1024 \
--num_step 1000;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 1024 \
--num_step 1000;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 1024 \
--num_step 1000;

# The number of iter variation
echo "The number of iter variation"

python evaluation_exp10.py \
--encoding_type delta \
--input_size 1024 \
--num_step 50 \
--num_iter 1000;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 1024 \
--num_step 50 \
--num_iter 1000;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 1024 \
--num_step 50 \
--num_iter 1000;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 1024 \
--num_step 50 \
--num_iter 1000;

python evaluation_exp10.py \
--encoding_type delta \
--input_size 1024 \
--num_step 50 \
--num_iter 2000;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 1024 \
--num_step 50 \
--num_iter 2000;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 1024 \
--num_step 50 \
--num_iter 2000;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 1024 \
--num_step 50 \
--num_iter 2000;

python evaluation_exp10.py \
--encoding_type delta \
--input_size 1024 \
--num_step 50 \
--num_iter 4000;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 1024 \
--num_step 50 \
--num_iter 4000;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 1024 \
--num_step 50 \
--num_iter 4000;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 1024 \
--num_step 50 \
--num_iter 4000;

python evaluation_exp10.py \
--encoding_type delta \
--input_size 1024 \
--num_step 50 \
--num_iter 8000;
python evaluation_exp10.py \
--encoding_type ttfs \
--input_size 1024 \
--num_step 50 \
--num_iter 8000;
python evaluation_exp10.py \
--encoding_type stime \
--input_size 1024 \
--num_step 50 \
--num_iter 8000;
python evaluation_exp10.py \
--encoding_type srate \
--input_size 1024 \
--num_step 50 \
--num_iter 8000;

echo "Complete!"