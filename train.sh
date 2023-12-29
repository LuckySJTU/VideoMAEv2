export NCCL_P2P_DISABLE=1
OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    run_mae_pretraining.py \
    --batch_size 12 \
    --epochs 100 \
    --model pretrain_videomae_giant_patch14_224 \
    --decoder_depth 4 \
    --mask_type tube \
    --mask_ratio 0.9 \
    --decoder_mask_type run_cell \
    --decoder_mask_ratio 0.5 \
    --lr 1e-4 \
    --finetune "./vit_g_hybrid_pt_1200e.pth" \
    --data_path "/home/yxwang/Dataset/LRS2/videoMAE.csv" \
    --num_sample 4 \
    --num_workers 10 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --clip_grad 0.02 \
    --log_dir "./checkpoints/post_pretraining" \
    --output_dir "./checkpoints/post_pretraining" \
    --with_checkpoint \
    --save_ckpt_freq 5
