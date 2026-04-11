#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_list=(RNN LSTM GRU)

for model_name in "${model_list[@]}"
do
  for pred_len in 96 192 336 720
  do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id "Exchange_96_$pred_len" \
      --model "$model_name" \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len "$pred_len" \
      --e_layers 2 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --itr 1
  done
done
