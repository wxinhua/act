######################

export HF_HOME=/media/users/wk/huggingface_model
export TORCH_HOME=/media/users/wk/torch_model

python3 deploy_algo.py \
        --camera_names camera_left camera_right camera_top \
        --ckpt_dir ./ckpt_dir/ACT_camlrt/241121/table/place_in_bread_on_plate_1_lr1e5_batch24_chunk50 \
        --exp_type franka_3rgb \
        --ckpt_name agent_last.ckpt \
        --agent_class ACT \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --kl_weight 10 \
        --backbone 'resnet18' \
        --act_norm_class norm2 \
        --temporal_agg \
