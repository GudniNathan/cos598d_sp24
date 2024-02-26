git pull

export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE
export WORLD_SIZE=4

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 64 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 4 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --local_rank $LOCAL_RANK \
  --master_addr "10.10.1.2" \
  --master_port 8888 \
  --world_size $WORLD_SIZE
