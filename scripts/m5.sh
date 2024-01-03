
seq_len=336
model_name=PDF

root_path_name=./m5-forecasting-accuracy/
data_path_name=sales_train_evaluation.csv
model_id_name=m5
data_name=m5
random_seed=2021

for pred_len in 28
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.5 \
      --fc_dropout 0.2 \
      --kernel_list 3 11 15 \
      --period 30 \
      --patch_len 8 \
      --stride 8 \
      --des 'Exp' \
      --patience 20 \
      --train_epochs 10 \
      --itr 1 --batch_size 32768 --learning_rate 0.00015
      # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
