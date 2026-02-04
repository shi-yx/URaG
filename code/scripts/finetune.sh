#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 deepspeed --master_port=29520 src/training/train.py \
    --lora_enable True \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --proj_layer_path './output/proj_layer_3B.pth' \
    --data_path './train_data/data_demo.json' \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/test_retrieval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --min_pixels $((128 * 28 * 28)) \
    --max_pixels $((1024 * 28 * 28)) \
    --learning_rate 5e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 8