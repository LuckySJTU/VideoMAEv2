source /home/yxwang/.bashrc
conda activate videomae
export CPATH=/home/yxwang/cuda/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/home/yxwang/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/yxwang/cuda/lib64/
#export PATH=/home/sxsong/my_cuda/cuda-11.6/bin:$PATH
export LDFLAGS=-L/home/yxwang/cuda/lib64/
export NCCL_P2P_DISABLE=1

nvcc --list-gpu-arch
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
        --nnodes=1 --node_rank=0 \
        run_class_finetuning.py \
        --model vit_giant_patch14_224 \
        --nb_classes 500 \
        --data_set LRW \
        --data_path "/home/yxwang/Dataset/LRW/videoMAE" \
        --log_dir "/home/yxwang/videoMAE/checkpoints/ft_post_pretraining13" \
        --output_dir "/home/yxwang/videoMAE/checkpoints/ft_post_pretraining13" \
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 8 \
        --opt adamw \
        --lr 1e-2 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 0 \
        --epochs 50 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --finetune "/home/yxwang/videoMAE/checkpoints/post_pretraining/checkpoint-99.pth" \
        --dist_eval \
        --enable_deepspeed
