python3 train_aloha.py \
        --cfg_path ./cfgs/act/config.yaml \
        --dataset_dir /nfsroot/DATA/Robot_DATA/cobot_kitchen_dataset1/data \
        --scene_name kitchen_h5 \
        --task_name appleblueplate \
        --camera_names camera_front camera_left_wrist camera_right_wrist \
        --ckpt_dir ./ckpt_dir/930/kitchen/appleblueplate_lr2e-5_batch24_chunk100 \
        --agent_class ACT \
        --chunk_size 100 \
        --batch_size_train 24 --batch_size_val 24 \
        --lr 2e-5 --kl_weight 10 \
        --lr_scheduler CosineLR \
        --num_epochs 10 \
        --save_epoch 1 \
        --wandb_name mobile_aloha_kitchen_appleblueplate
# camera_name need set: top down left
