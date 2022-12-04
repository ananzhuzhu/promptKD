python TinyBERT/data_augmentation.py \
    --pretrained_bert_model ../bert-base-uncased \
    --glove_embs ../glove/glove.6B.300d.txt \
    --glue_dir  stark/datasets \
    --task_name rte


CUDA_VISIBLE_DEVICES=1 python TinyBERT/data_augmentation.py \
    --pretrained_bert_model ../bert-base-uncased \
    --glove_embs ../glove/glove.6B.300d.txt \
    --glue_dir stark/datasets \
    --task_name mnli


CUDA_VISIBLE_DEVICES=2 python TinyBERT/data_augmentation.py \
    --pretrained_bert_model ../bert-base-uncased \
    --glove_embs ../glove/glove.6B.300d.txt \
    --glue_dir stark/datasets \
    --task_name qqp