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

# multi_task_2
# place_button

# --batch_size_train 24 --batch_size_val 24 \
# --num_steps 20 --eval_every 21 --validate_every 10 --save_every 20 \
# --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
CUDA_VISIBLE_DEVICES=0 python3 train_algo.py \
        --task_name place_button \
        --camera_names camera_top \
        --ckpt_dir ./ckpt_dir/DroidDiffusion_camt/241112/table/tiangong_1rgb_mode3_place_button_lr1e5_batch24_chunk16 \
        --exp_type tiangong_1rgb \
        --agent_class DroidDiffusion \
        --batch_size_train 48 --batch_size_val 48 \
        --chunk_size 16 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --lr_backbone 1e-4 \
        --backbone 'resnet50' \
        --act_norm_class norm1 \
        --lr_scheduler CosineLR \
        --tg_mode mode3 \
        --pool_class 'SpatialSoftmax' --use_data_aug \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
        --use_wandb \
        --wandb_name DroidDiffusion_tiangong_1rgb_mode3_camt_place_button_lr1e5_batch24_chunk16 \

