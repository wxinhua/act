######################

# export HF_HOME=/media/users/wk/huggingface_model
# export TORCH_HOME=/media/users/wk/torch_model

export HF_HOME=/home/ps/wk/cache_model/huggingface_model
export TORCH_HOME=/home/ps/wk/cache_model/torch_model

# agent_best
# policy_last
# num_steps is used for training

python3 deploy_algo.py \
        --task_name franka_3rgb_241021_insert_marker_1 \
        --camera_names left right top \
        --ckpt_dir /media/ps/wk/benchmark_results/act/franka_3rgb_241021_insert_marker_1/ckpt \
        --exp_type franka_3rgb \
        --ckpt_name agent_best.ckpt \
        --agent_class ACT \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --kl_weight 10 \
        --backbone 'resnet18' \
        --act_norm_class norm2 \
        --temporal_agg --num_steps 0 \
