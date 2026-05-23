#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# =========================
# BASE CONFIG (FAIR SETUP)
# =========================
ROOT_PATH=./dataset/ETT-small/
DATA=ETTh2
DATA_PATH=ETTh2.csv
SEQ_LEN=96

ENC_IN=7
DEC_IN=7
C_OUT=7
E_LAYERS=2

D_MODEL=128
D_FF=128

DES="Benchmark"

# =========================
# MODELS
# =========================
models=("LSTM" "GRU" "RNN")

# =========================
# HORIZONS
# =========================
pred_lens=(96 192 336 720)

# =========================
# MAIN LOOP
# =========================
for model_name in "${models[@]}"; do
  for pred_len in "${pred_lens[@]}"; do

      model_id="${DATA}_${SEQ_LEN}_${pred_len}_${model_name}"

      echo "====================================="
      echo "Model: $model_name"
      echo "Horizon: $pred_len"
      echo "====================================="

      python -u run.py \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model_id $model_id \
        --model $model_name \
        --data $DATA \
        --features M \
        --seq_len $SEQ_LEN \
        --pred_len $pred_len \
        --e_layers $E_LAYERS \
        --enc_in $ENC_IN \
        --dec_in $DEC_IN \
        --c_out $C_OUT \
        --des $DES \
        --d_model $D_MODEL \
        --d_ff $D_FF \
        --itr 1

  done
done
