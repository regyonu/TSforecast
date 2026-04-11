#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_list=(RNN LSTM GRU)

for model_name in "${model_list[@]}"
do
  for pred_len in 24 36 48 60
  do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/illness/ \
      --data_path national_illness.csv \
      --model_id "ili_36_$pred_len" \
      --model "$model_name" \
      --data custom \
      --features M \
      --seq_len 36 \
      --pred_len "$pred_len" \
      --e_layers 2 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --batch_size 32 \
      --learning_rate 0.001 \
      --itr 1
  done
done
