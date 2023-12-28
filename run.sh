

python run.py --dataset takatak --n_interest 2 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_takatak'

if [ $? -ne 0 ]; then
  echo "takatak interest failed 2"
  exit 1
fi

python run.py --dataset takatak --n_interest 4 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_takatak'

if [ $? -ne 0 ]; then
  echo "takatak interest failed 2"
  exit 1
fi

python run.py --dataset takatak --n_interest 1 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_takatak'

if [ $? -ne 0 ]; then
  echo "takatak interest failed 1"
  exit 1
fi


python run.py --dataset wechat --n_interest 4 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_wechat'
