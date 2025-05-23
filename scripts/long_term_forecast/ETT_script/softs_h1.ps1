# PowerShell translation of your bash script
# define the model name
$model_name = 'SOFTS'

# first run
python -u run.py `
  --is_training 1 `
  --root_path "C:\\Users\\ianre\\Desktop\\coda\\ianre\\myforks\\SOFTS\\data\\ETT-small" `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_96 `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 1 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --d_model 256 `
  --d_core 256 `
  --d_ff 256 `
  --learning_rate 0.0003 `
  --lradj cosine `
  --train_epochs 20 `
  --patience 3 `
  --des 'Exp' `
  --itr 1

# second run
python -u run.py `
  --is_training 1 `
  --root_path "C:\\Users\\ianre\\Desktop\\coda\\ianre\\myforks\\SOFTS\\data\\ETT-small" `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_192 `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 192 `
  --e_layers 1 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --d_model 256 `
  --d_core 256 `
  --d_ff 256 `
  --learning_rate 0.0003 `
  --lradj cosine `
  --train_epochs 20 `
  --patience 3 `
  --des 'Exp' `
  --itr 1

# third run
python -u run.py `
  --is_training 1 `
  --root_path "C:\\Users\\ianre\\Desktop\\coda\\ianre\\myforks\\SOFTS\\data\\ETT-small" `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_336 `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 336 `
  --e_layers 1 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --d_model 256 `
  --d_core 256 `
  --d_ff 512 `
  --learning_rate 0.0003 `
  --lradj cosine `
  --train_epochs 20 `
  --patience 3 `
  --des 'Exp' `
  --itr 1

# fourth run
python -u run.py `
  --is_training 1 `
  --root_path "C:\\Users\\ianre\\Desktop\\coda\\ianre\\myforks\\SOFTS\\data\\ETT-small" `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_720 `
  --model $model_name `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --pred_len 720 `
  --e_layers 1 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --d_model 256 `
  --d_core 256 `
  --d_ff 512 `
  --learning_rate 0.0003 `
  --lradj cosine `
  --train_epochs 20 `
  --patience 3 `
  --des 'Exp' `
  --itr 1
