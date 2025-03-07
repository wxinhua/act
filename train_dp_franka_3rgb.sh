########
# /media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/ai_station_data
# h5_franka  h5_franka_single  h5_simulation  h5_songling  h5_tiangong  h5_ur

# exp type: franka_3rgb, franka_1rgb, ur_1rgb, songling_3rgb, tiangong_1rgb, sim

# export TRANSFORMERS_CACHE=/nfsroot/DATA/IL_Research/wk/huggingface_model
# export HF_HOME=/nfsroot/DATA/IL_Research/wk/huggingface_model
# export TORCH_HOME=/nfsroot/DATA/IL_Research/wk/torch_model

#### baidu
export HF_HOME=/media/users/wk/huggingface_model
export TORCH_HOME=/media/users/wk/torch_model

# --batch_size_train 24 --batch_size_val 24 \
# place_in_bread_on_plate_1
# multi_task_2
# --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
CUDA_VISIBLE_DEVICES=0 python3 train_algo.py \
        --task_name place_in_bread_on_plate_1 \
        --camera_names camera_left camera_right camera_top \
        --ckpt_dir ./ckpt_dir/DroidDiffusion_camlrt/241112/table/franka_3rgb_place_in_bread_on_plate_1_lr1e5_batch24_chunk16 \
        --exp_type franka_3rgb \
        --agent_class DroidDiffusion \
        --batch_size_train 32 --batch_size_val 32 \
        --chunk_size 16 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --lr_backbone 1e-4 \
        --backbone 'resnet50' \
        --act_norm_class norm1 \
        --lr_scheduler CosineLR \
        --pool_class 'SpatialSoftmax' --use_data_aug \
        --num_steps 75000 --eval_every 75001 --validate_every 250 --save_every 30000 \
        --use_wandb \
        --wandb_name DroidDiffusion_franka_3rgb_camlrt_place_in_bread_on_plate_1_lr1e5_batch24_chunk16 \

