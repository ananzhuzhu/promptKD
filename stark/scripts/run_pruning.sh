python run_pruning.py \
    --model_type ft \
    --model_path ../Glee/outputs/cls_tuning_rte_2_32_3e-05/ckpt-1780-2022-12-03-23:23:47/ \
    --task_name rte \
    --data_type combined \
    --template "" \
    --verbalizer "" \
    --max_length 128 \
    --per_device_eval_batch_size 32


CUDA_VISIBLE_DEVICES=1 python run_pruning.py \
    --model_type ft_prompt \
    --model_path ../Glee/outputs/prompt_tuning_rte_2_32_3e-05/ckpt-2623-2022-12-03-23:28:12/ \
    --task_name rte \
    --data_type prompted \
    --template "{cls} {text_a} question: {text_b} ? the answer: {mask} . {sep}" \
    --verbalizer "verbalizers/rte.verbalizer" \
    --max_length 128 \
    --per_device_eval_batch_size 32
