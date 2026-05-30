#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# =========================
# DATA
# =========================
DATA=ETTh2
ROOT_PATH=./dataset/ETT-small/
DATA_PATH=ETTh2.csv

# =========================
# FORECAST SETTINGS
# =========================
SEQ_LEN=96
PRED_LENS=(96 192 336 720)

ENC_IN=7
DEC_IN=7
C_OUT=7

# =========================
# MODEL (shared)
# =========================
D_MODEL=256
N_HEADS=8
E_LAYERS=2
D_LAYERS=1
D_FF=512
FACTOR=3
DROPOUT=0.1
EMBED="timeF"

# =========================
# TRAINING
# =========================
TRAIN_EPOCHS=20
BATCH_SIZE=32
PATIENCE=5
LR=0.0001
LOSS="MSE"
LRADJ="type1"
NUM_WORKERS=2
ITR=1

# =========================
# MODELS
# =========================
models=("Transformer" "iTransformer")

# =========================
# RUN
# =========================
for model_name in "${models[@]}"; do
  for pred_len in "${PRED_LENS[@]}"; do

    model_id="${DATA}_${SEQ_LEN}_${pred_len}"

    echo "====================================="
    echo "Model: $model_name | Horizon: $pred_len"
    echo "====================================="

    python -u run.py \
      --is_training 1 \
      --model_id $model_id \
      --model $model_name \
      --data $DATA \
      --root_path $ROOT_PATH \
      --data_path $DATA_PATH \
      --features M \
      --seq_len $SEQ_LEN \
      --pred_len $pred_len \
      --enc_in $ENC_IN \
      --dec_in $DEC_IN \
      --c_out $C_OUT \
      --d_model $D_MODEL \
      --n_heads $N_HEADS \
      --e_layers $E_LAYERS \
      --d_layers $D_LAYERS \
      --d_ff $D_FF \
      --factor $FACTOR \
      --embed $EMBED \
      --dropout $DROPOUT \
      --batch_size $BATCH_SIZE \
      --train_epochs $TRAIN_EPOCHS \
      --patience $PATIENCE \
      --learning_rate $LR \
      --loss $LOSS \
      --lradj $LRADJ \
      --num_workers $NUM_WORKERS \
      --des "Benchmark" \
      --itr $ITR

  done
done
