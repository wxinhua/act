########
# /media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/ai_station_data
# h5_franka  h5_franka_single  h5_simulation  h5_songling  h5_tiangong  h5_ur

# exp type: franka_3rgb, franka_1rgb, ur_1rgb, songling_3rgb, tiangong_1rgb, sim

# export TRANSFORMERS_CACHE=/nfsroot/DATA/IL_Research/wk/huggingface_model
# export HF_HOME=/nfsroot/DATA/IL_Research/wk/huggingface_model
# export TORCH_HOME=/nfsroot/DATA/IL_Research/wk/torch_model


#### baidu
# export HF_HOME=/media/users/wk/huggingface_model
# export TORCH_HOME=/media/users/wk/torch_model

# multi_task_2
# open_top_white_drawer

# --batch_size_train 24 --batch_size_val 24 \
# --num_steps 20 --eval_every 21 --validate_every 10 --save_every 20 \
CUDA_VISIBLE_DEVICES=0 python3 train_algo.py \
        --task_name ur_put_steamed_bun_on_the_steamer_100 \
        --camera_names camera_left \
        --ckpt_dir /media/wxh/de27d578-5aeb-4158-9716-f0ec70911dbb/dp_ckpt/ur_put_steamed_bun_on_the_steamer_100\
        --exp_type ur_std_station_1 \
        --agent_class DroidDiffusion \
        --batch_size_train 24 --batch_size_val 24 \
        --chunk_size 16 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --lr_backbone 1e-4 \
        --backbone 'resnet18' \
        --act_norm_class norm1 \
        --lr_scheduler CosineLR \
        --pool_class 'SpatialSoftmax' --use_data_aug \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
        --use_wandb \
        --wandb_name DP_ur_put_steamed_bun_on_the_steamer_100_lr1e4_batch24_chunk16 \










