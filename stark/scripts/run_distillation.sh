#!/bin/bash
# teacher prune
python run_distillation.py \
    --model_type kd \
    --teacher_type finetune \
    --teacher_model_path ../Glee/outputs/cls_tuning_rte_2_32_3e-05/ckpt-1780-2022-12-03-23:23:47/ \
    --student_model_path outputs/prune/ft_rte/ckpt-sparsity/ \
    --task_name rte \
    --data_type combined \
    --data_aug False \
    --selection_metric acc \
    --max_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --weight_decay 1e-2 \
    --log_interval 10 \
    --num_train_epochs 10 \
    --num_patience_epochs 5 \
    --warmup_proportion 0.1 \
    --max_grad_norm 5.0 \
    --seed 776 \
    --do_distill \
    --student_sparsity 70


# prompt teacher prune w.prompt
CUDA_VISIBLE_DEVICES=1 python run_distillation.py \
    --model_type kd \
    --teacher_type prompt \
    --teacher_model_path ../Glee/outputs/prompt_tuning_rte_2_32_3e-05/ckpt-2623-2022-12-03-23:28:12/ \
    --student_model_path outputs/prune/ft_prompt_rte/ckpt-sparsity/ \
    --task_name rte \
    --data_type combined \
    --data_aug False \
    --selection_metric acc \
    --max_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --weight_decay 1e-2 \
    --log_interval 10 \
    --num_train_epochs 10 \
    --num_patience_epochs 5 \
    --warmup_proportion 0.1 \
    --max_grad_norm 5.0 \
    --seed 776 \
    --do_distill \
    --student_sparsity 70