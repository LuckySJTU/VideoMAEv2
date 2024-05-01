#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=$((12000 + $RANDOM % 20000))  # Randomly set master_port to avoid port conflicts
export OMP_NUM_THREADS=1  # Control the number of threads
export CPATH=/home/yxwang/cuda/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/home/yxwang/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/yxwang/cuda/lib64/
#export PATH=/home/sxsong/my_cuda/cuda-11.6/bin:$PATH
export LDFLAGS=-L/home/yxwang/cuda/lib64/
export NCCL_P2P_DISABLE=1

OUTPUT_DIR='/home/yxwang/videoMAE/checkpoints/ft_post_pretraining15'  # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='/home/yxwang/Dataset/LRW/videoMAE'  # The data list folder. the folder has three files: train.csv, val.csv, test.csv
# finetune data list file follows the following format
# for the video data line: video_path, label
# for the rawframe data line: frame_folder_path, total_frames, label
MODEL_PATH='/home/yxwang/videoMAE/checkpoints/post_pretraining/checkpoint-99.pth'  # Model for initializing parameters

JOB_NAME=$1  # the job name of the slurm task
PARTITION=${PARTITION:-"RTX4090"}  # Name of the partition
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-8}  # Number of GPUs
GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # Number of GPUs in each node
CPUS_PER_TASK=${CPUS_PER_TASK:-10}  # Number of CPU cores allocated, number of tasks equal to the number of GPUs used
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:2}  # Other training args

# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --mem=990G \
        --kill-on-bad-exit=1 \
        python run_class_finetuning.py \
        --model vit_giant_patch14_224 \
        --data_set LRW \
        --nb_classes 500 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 1 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 8 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 0 \
        --epochs 30 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}
