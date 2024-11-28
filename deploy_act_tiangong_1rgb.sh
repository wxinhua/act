######################

# export HF_HOME=/media/users/wk/huggingface_model
# export TORCH_HOME=/media/users/wk/torch_model

# export HF_HOME=/home/ps/wk/cache_model/huggingface_model
# export TORCH_HOME=/home/ps/wk/cache_model/torch_model

export HF_HOME=/home/ps/wk/cache_model/huggingface_model
export TORCH_HOME=/home/ps/wk/cache_model/torch_model

# agent_best
# policy_last
# num_steps is used for training
# --ckpt_dir /media/ps/wk/benchmark_results/act/ur_1rgb_close_trash_can/ckpt \
# /media/ps/Extreme Pro_1/wk/benchmark_results/act/tiangong_1rgb_place_button

python3 deploy_algo.py \
        --task_name tiangong_1rgb_mode1_place_button_lr1e5_batch24_chunk50 \
        --camera_names left \
        --ckpt_dir /home/ps/wk/benchmark_results/act/tiangong_1rgb_mode1_place_button_lr1e5_batch24_chunk50 \
        --exp_type tiangong_1rgb \
        --ckpt_name agent_best.ckpt \
        --agent_class ACT \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --kl_weight 10 \
        --backbone 'resnet18' \
        --act_norm_class norm2 \
        --tg_mode mode1 \
        --temporal_agg --num_steps 0 \
