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


# CUDA_VISIBLE_DEVICES=0 python3 train_algo.py \
#         --task_name multi_task_2 \
#         --camera_names camera_top \
#         --ckpt_dir ./ckpt_dir/ACT_camt/241112/table/ur_1rgb_multi_task_2_lr1e5_batch24_chunk50 \
#         --exp_type ur_1rgb \
#         --agent_class ACT \
#         --batch_size_train 24 --batch_size_val 24 \
#         --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
#         --lr 5e-5 --kl_weight 10 \
#         --backbone 'resnet18' \
#         --act_norm_class norm2 \
#         --lr_scheduler CosineLR \
#         --num_steps 20 --eval_every 21 --validate_every 10 --save_every 10 \
#         --use_wandb \
#         --wandb_name ACT_ur_1rgb_camlrt_multi_task_2_lr1e5_batch24_chunk50 \

# CUDA_VISIBLE_DEVICES=0 python3 train_algo.py \
#         --task_name multi_task_2 \
#         --camera_names camera_top \
#         --ckpt_dir ./ckpt_dir/ACT_camt/241112/table/ur_1rgb_multi_task_2_lr1e5_batch24_chunk50 \
#         --exp_type ur_1rgb \
#         --agent_class ACT \
#         --batch_size_train 24 --batch_size_val 24 \
#         --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
#         --lr 5e-5 --kl_weight 10 --use_lang \
#         --backbone 'resnet18' \
#         --act_norm_class norm2 \
#         --lr_scheduler CosineLR \
#         --num_steps 20 --eval_every 21 --validate_every 10 --save_every 10 \
#         --use_wandb \
#         --wandb_name ACT_ur_1rgb_camlrt_multi_task_2_lr1e5_batch24_chunk50 \

############ for benchmark!
# --num_steps 100000 --eval_every 100001 --validate_every 500 --save_every 50000 \
CUDA_VISIBLE_DEVICES=0 python3 train_algo.py \
        --task_name ur_put_steamed_bun_on_the_steamer_100 \
        --camera_names camera_front\
        --ckpt_dir /media/wxh/de27d578-5aeb-4158-9716-f0ec70911dbb/act_ckpt/camera_front \
        --exp_type ur_std_station_1 \
        --agent_class ACT \
        --batch_size_train 24 --batch_size_val 24 \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --kl_weight 10 \
        --backbone 'resnet18' \
        --act_norm_class norm2 \
        --lr_scheduler CosineLR \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 10000 \
        --use_wandb \
        --wandb_name ACT_ur_put_steamed_bun_on_the_steamer_100_lr1e4_batch24_chunk50 \
        --use_aug_data False \
        --aug_num 0 \




