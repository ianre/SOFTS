
# set variable
$model_name = 'SOFTS'

# first forecast

python -u run.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --root_path "C:\\Users\\ianre\\Desktop\\coda\\ianre\\myforks\\SOFTS\\data\\traffic" `
  --data_path traffic.csv `
  --model_id traffic_96_96 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 3 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --d_model 512 `
  --d_core 128 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0003 `
  --train_epochs 50 `
  --patience 50 `
  --lradj cosine `
  --des 'Exp' `
  --itr 1

# second forecast

python -u run.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --root_path "C:\\Users\\ianre\\Desktop\\coda\\ianre\\myforks\\SOFTS\\data\\traffic" `
  --data_path traffic.csv `
  --model_id traffic_96_192 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 192 `
  --e_layers 3 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --d_model 512 `
  --d_core 128 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0003 `
  --train_epochs 50 `
  --patience 50 `
  --lradj cosine `
  --des 'Exp' `
  --itr 1

# third forecast

python -u run.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --root_path "C:\\Users\\ianre\\Desktop\\coda\\ianre\\myforks\\SOFTS\\data\\traffic" `
  --data_path traffic.csv `
  --model_id traffic_96_336 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 336 `
  --e_layers 3 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --d_model 512 `
  --d_core 128 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0003 `
  --train_epochs 50 `
  --patience 50 `
  --lradj cosine `
  --des 'Exp' `
  --itr 1

# fourth forecast

python -u run.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --root_path "C:\\Users\\ianre\\Desktop\\coda\\ianre\\myforks\\SOFTS\\data\\traffic" `
  --data_path traffic.csv `
  --model_id traffic_96_720 `
  --model $model_name `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 720 `
  --e_layers 3 `
  --enc_in 862 `
  --dec_in 862 `
  --c_out 862 `
  --d_model 512 `
  --d_core 128 `
  --d_ff 512 `
  --batch_size 16 `
  --learning_rate 0.0003 `
  --train_epochs 50 `
  --patience 50 `
  --lradj cosine `
  --des 'Exp' `
  --itr 1