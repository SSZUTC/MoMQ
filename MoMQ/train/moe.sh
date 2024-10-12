#!/bin/bash
export HF_ENDPOINT='https://hf-mirror.com'
# export HF_HOME='/mnt/model_cache/'
export HF_HOME='../../cache'
# export WANDB_PROJECT='nl2sql_0729_moe_fewshot'
export WANDB_PROJECT='nl2sql_0906_moe'
# export WANDB_PROJECT='nl2sql_0730'
# export CUDA_VISIBLE_DEVICES=0,1,2,3
REPORT_TO="none"
RESUME=False

run_training() {
    local DATA=$1
    local OUTPUT=$2
        # python sft.py \
    accelerate launch --num_processes 4 --main_process_port 52262 --config_file $CONFIG sft_moe.py \
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
        --save_strategy "steps" \
        --eval_strategy "steps" \
        --eval_steps $EVAL_STEP \
        --save_steps $EVAL_STEP \
        --save_total_limit 50 \
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
        --use_moe_lora $USE_MOE_LORA \
        --use_lora $USE_LORA \
        --lora_r $LORA_R \
        --lora_alpha $((LORA_R * 2)) \
        --torch_compile True \
        --group_by_length True \
        --model_type $MODEL_TYPE \
        --use_moe_expert $USE_MOE_EXPERT \
        --moe_intermediate_size $moe_intermediate_size \
        --num_experts $num_experts \
        --lora_route_type $lora_route_type \
        --moe_lora_target_modules "${moe_lora_target_modules[@]}" \
        --output_router_logits $output_router_logits \
        --enable_dialect_router $enable_dialect_router \
        --dialect_router_loss_coef $dialect_router_loss_coef \
        --num_experts_per_tok $num_experts_per_tok \
        --dialect_num $dialect_num \
        --enable_label_smooth $enable_label_smooth \
        --smooth_factor $smooth_factor \
        --share_expert_num $share_expert_num \
        --train_dialects_num_map "$train_dialects_num_map" \
        --eval_dialects_num_map "$eval_dialects_num_map" \
        --dataloader_num_workers 8 \
        --hard_dialect_router $hard_dialect_router \
        --use_in_group_balance $use_in_group_balance \
        --seed 42 \
        --bf16
}
# ====================dense======================
CONFIG="config/zero2.yaml"
# CONFIG="config/zero1.yaml"
MODEL_TYPE="qwen"

# DATA="data/0725/train_mysql_pg_cypher_ngql.json"
# EVAL_DATA="data/0725/eval_mysql_pg_cypher_ngql.json"
DATA="data/0902/train_mysql_pg_sqlite_final.json"
EVAL_DATA="data/0902/eval_mysql_pg_spider_final_test.json"

LR=1e-5

# moe config
USE_LORA=False
USE_MOE_LORA=True
USE_MOE_EXPERT=False
moe_intermediate_size=64
lora_route_type='token'
moe_lora_target_modules=("down_proj")
# lora_target_modules=("q_proj" "k_proj" "v_proj" "o_proj" "down_proj")
enable_dialect_router=True
output_router_logits=True
router_aux_loss_coef=0.001
dialect_router_loss_coef=0.01
hard_dialect_router=False
use_in_group_balance=False
enable_label_smooth=False
smooth_factor=0
# mysql pg sqlite
dialect_num=3

# learning config
EVAL_STEP=1000
EPOCH=5
BZ=2
ACC_STEP=1

MODEL="/mnt/model_cache/Qwen2-1___5B-Instruct/"
MODEL="/mnt/model_cache/Qwen1___5-14B-Chat/"
MODEL="Qwen/Qwen2-7B-Instruct"
MODEL="/mnt/model_cache/CodeQwen1___5-7B-Chat"




BZ=2
ACC_STEP=1

LORA_R=128
LR=1e-5
num_experts=36
num_experts_per_tok=2
share_expert_num=2
dialect_router_loss_coef=0.01
dialect_num=3

OUTPUT="output/moe/0902/codeqwen_moe_"$LORA_R"_"$LR"_"$BZ"_expert_"$num_experts"_"$num_experts_per_tok"_"$lora_route_type"_dialect="$enable_dialect_router"_"$output_router_logits"_"$dialect_router_loss_coef"_"$smooth_factor"_"$share_expert_num"_"$train_dialects_num_map
run_training $DATA $OUTPUT

num_experts=48
OUTPUT="output/moe/0902/codeqwen_moe_"$LORA_R"_"$LR"_"$BZ"_expert_"$num_experts"_"$num_experts_per_tok"_"$lora_route_type"_dialect="$enable_dialect_router"_"$output_router_logits"_"$dialect_router_loss_coef"_"$smooth_factor"_"$share_expert_num"_"$train_dialects_num_map
run_training $DATA $OUTPUT
