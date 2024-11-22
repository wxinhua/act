########
# /media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/ai_station_data
# h5_franka  h5_franka_single  h5_simulation  h5_songling  h5_tiangong  h5_ur

# exp type: franka_3rgb, franka_1rgb, ur_1rgb, songling_3rgb, tiangong_1rgb, sim

# export TRANSFORMERS_CACHE=/nfsroot/DATA/IL_Research/wk/huggingface_model
# export HF_HOME=/nfsroot/DATA/IL_Research/wk/huggingface_model
# export TORCH_HOME=/nfsroot/DATA/IL_Research/wk/torch_model

# CUDA_VISIBLE_DEVICES=0 python3 train_algo.py \
#         --task_name multi_task_2 \
#         --camera_names camera_left camera_right camera_top \
#         --ckpt_dir ./ckpt_dir/DroidDiffusion_camlrt/241112/table/franka_3rgb_multi_task_2_lr1e5_batch24_chunk16 \
#         --exp_type franka_3rgb \
#         --agent_class DroidDiffusion \
#         --batch_size_train 24 --batch_size_val 24 \
#         --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
#         --lr 1e-4 --lr_backbone 1e-4 \
#         --backbone 'resnet18' \
#         --act_norm_class norm1 \
#         --lr_scheduler CosineLR \
#         --pool_class 'SpatialSoftmax' --use_data_aug \
#         --num_steps 20 --eval_every 21 --validate_every 10 --save_every 10 \
#         --use_wandb \
#         --wandb_name DroidDiffusion_franka_3rgb_camlrt_multi_task_2_lr1e5_batch24_chunk16 \

# --batch_size_train 24 --batch_size_val 24 \
CUDA_VISIBLE_DEVICES=0 python3 train_algo.py \
        --task_name pick_plate_from_plate_rack \
        --camera_names camera_left camera_right \
        --ckpt_dir ./ckpt_dir/DroidDiffusion_camlrt/241112/table/franka_3rgb_pick_plate_from_plate_rack_lr1e5_batch24_chunk16 \
        --exp_type franka_3rgb \
        --agent_class DroidDiffusion \
        --batch_size_train 24 --batch_size_val 24 \
        --chunk_size 16 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --lr_backbone 1e-4 \
        --backbone 'resnet18' \
        --act_norm_class norm1 \
        --lr_scheduler CosineLR \
        --pool_class 'SpatialSoftmax' --use_data_aug \
        --num_steps 100000 --eval_every 100001 --validate_every 500 --save_every 50000 \
        --use_wandb \
        --wandb_name DroidDiffusion_franka_3rgb_camlrt_pick_plate_from_plate_rack_lr1e5_batch24_chunk16 \

