git pull

export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE
export WORLD_SIZE=4

torchrun --nnodes 1 --nproc_per_node 4 run_glue.py \
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
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --world_size 4
