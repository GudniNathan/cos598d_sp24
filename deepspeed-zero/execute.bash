git pull

export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE
export WORLD_SIZE=4

deepspeed --num_gpus 4 \
  run_glue.py \
  --deepspeed \
  --deepspeed_config  deepspeed_bsz4096_adam_config.json \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name RTE \
  --do_train \
  --do_eval \
  --data_dir ~/glue_data/RTE \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 64 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 10 \
  --output_dir /tmp/RTE/ \
  --overwrite_output_dir \
  --master_addr localhost \
  --master_port 8002 \
  --world_size 4
