# !/bin/sh

CUDA_VISIBLE_DEVICES=1  python Glee/run_cls.py \
        --model_type prompt_tuning \
        --model_name_or_path ../bert-base-uncased \
        --task_name rte \
        --data_type prompted \
        --template "{cls} {text_a} question: {text_b} ? the answer: {mask} . {sep}" \
        --verbalizer "verbalizers/rte.verbalizer" \
        --max_length 128 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 32 \
        --learning_rate 3e-5 \
        --weight_decay 0.0 \
        --log_interval 1 \
        --num_train_epochs 5 \
        --num_patience_epochs 5 \
        --warmup_proportion 0.1 \
        --max_grad_norm 1.0 \
        --seed 776 \
        --selection_metric acc \
        --do_train \
        --do_test \
        --model_suffix rte

CUDA_VISIBLE_DEVICES=2 python Glee/run_cls.py \
        --model_type cls_tuning \
        --model_name_or_path ../bert-base-uncased \
        --task_name rte \
        --data_type combined \
        --template "" \
        --verbalizer "" \
        --max_length 128 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 32 \
        --learning_rate 3e-5 \
        --weight_decay 0.0 \
        --log_interval 1 \
        --num_train_epochs 5 \
        --num_patience_epochs 5 \
        --warmup_proportion 0.1 \
        --max_grad_norm 1.0 \
        --seed 776 \
        --selection_metric acc \
        --do_train \
        --do_test \
        --model_suffix rte