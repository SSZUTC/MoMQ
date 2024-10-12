#!/bin/bash
export HF_ENDPOINT='https://hf-mirror.com'
# export HF_HOME='/mnt/model_cache/'
# export HF_HOME='../../cache'
# export WANDB_PROJECT='nl2sql_0729_moe_fewshot'
export WANDB_PROJECT='nl2sql_1012'
# export WANDB_PROJECT='nl2sql_0730'
export CUDA_VISIBLE_DEVICES=0,1
REPORT_TO="wandb"
RESUME=False

run_training() {
    local DATA=$1
    local OUTPUT=$2
        # python sft.py \
    accelerate launch --num_processes 2 --main_process_port 52262 --config_file $CONFIG sft.py \
        --save_only_model True \
        --resume False \
        --model_name_or_path $MODEL \
        --data_path $DATA \
        --eval_data_path $EVAL_DATA \
        --output_dir $OUTPUT \
        --num_train_epochs $EPOCH \
        --per_device_train_batch_size $BZ \
        --load_best_model_at_end False\
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $ACC_STEP \
        --save_strategy "epoch" \
        --eval_strategy "epoch" \
        --eval_steps $EVAL_STEP \
        --save_steps $EVAL_STEP \
        --save_total_limit 1 \
        --learning_rate $LR \
        --weight_decay 0.1 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --log_level "debug" \
        --logging_steps 10 \
        --report_to $REPORT_TO \
        --model_max_length 8192 \
        --gradient_checkpointing True \
        --predict_with_generate True \
        --include_inputs_for_metrics True \
        --torch_compile True \
        --group_by_length True \
        --model_type $MODEL_TYPE \
        --dataloader_num_workers 8 \
        --seed 42 \
        --bf16
}
# ====================dense======================
CONFIG="config/zero2.yaml"
MODEL_TYPE="auto"
MODEL="/home/linzhisheng/.cache/modelscope/hub/qwen/Qwen2___5-Coder-1___5B-Instruct"

DATA="data/1012/train_spider_chat.json"
EVAL_DATA="data/1012/eval_spider_chat.json"

# learning config
LR=5e-5
EVAL_STEP=100
EPOCH=3
BZ=8
ACC_STEP=1

OUTPUT="output/dense/1012/dense_spider_1.5b_"$LR"_"$BZ"_"$EPOCH
run_training $DATA $OUTPUT


