python3 train_franka.py \
        --cfg_path ./cfgs/act/config.yaml \
        --dataset_dir /alex.zhao/dataset/franka/data \
        --scene_name table_h5 \
        --task_name open_drawer_h5 \
        --camera_names camera_left camera_right \
        --ckpt_dir ./ckpt_dir/930/table/open_drawer_lr1e-5_batch24_chunk100 \
        --agent_class ACT \
        --chunk_size 100 \
        --batch_size_train 24 --batch_size_val 24 \
        --lr 1e-5 --kl_weight 10 \
        --lr_scheduler CosineLR \
        --num_epochs 10 \
        --save_epoch 1 \
        --wandb_name franka_open_drawer
# camera_name need set: top down left
