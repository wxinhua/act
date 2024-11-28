######################

# export HF_HOME=/media/users/wk/huggingface_model
# export TORCH_HOME=/media/users/wk/torch_model

# export HF_HOME=/home/ps/wk/cache_model/huggingface_model
# export TORCH_HOME=/home/ps/wk/cache_model/torch_model

export HF_HOME=/home/agilex/wk/cache_model/huggingface_model
export TORCH_HOME=/home/agilex/wk/cache_model/torch_model



# agent_best
# policy_last
# num_steps is used for training

python3 deploy_songling_algo.py \
        --task_name songling_3rgb_13_packbowl \
        --camera_names cam_high cam_left_wrist cam_right_wrist \
        --ckpt_dir /home/agilex/wk/benchmark_results/act/songling_3rgb_13_packbowl/ckpt \
        --exp_type songling_3rgb \
        --ckpt_name agent_best.ckpt \
        --agent_class ACT \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --kl_weight 10 \
        --backbone 'resnet18' \
        --act_norm_class norm2 \
        --temporal_agg --num_steps 0 \
