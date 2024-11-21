####################

# image_data size: torch.Size([1, 3, 480, 640])
# python3 train_franka.py \
#         --cfg_path ./cfgs/act/config_franka.yaml \
#         --dataset_dir /media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/real_franka \
#         --scene_name h5_data \
#         --task_name 240924_pick_bread_plate_1 \
#         --camera_names camera_top \
#         --ckpt_dir ./ckpt_dir/ACT_camt/241011/table/240924_pick_bread_plate_1_lr1e5_batch24_chunk50 \
#         --agent_class ACT \
#         --chunk_size 50 \
#         --batch_size_train 24 --batch_size_val 24 \
#         --lr 1e-5 --kl_weight 10 \
#         --lr_scheduler CosineLR \
#         --num_epochs 50 \
#         --save_epoch 10 \
#         --wandb_name 240924_pick_bread_plate_1_lr1e5_batch24_chunk50

# python3 run_franka_act_inference.py \
#         --cfg_path ./cfgs/gemo/config_franka.yaml \
#         --camera_names top \
#         --ckpt_dir /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camt/241011/table/240924_pick_bread_plate_1_lr1e5_batch24_chunk50 \
#         --ckpt_name agent_best.ckpt \
#         --agent_class ACT \
#         --chunk_size 50 \
#         --lr 1e-5 --kl_weight 10 \
#         --temporal_agg \


######################## 
# 241012_upright_blue_cup_1

# /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlr/241014/table/241012_upright_blue_cup_1_lr1e5_batch24_chunk50 \
# /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241015/table/241012_upright_blue_cup_1_lr1e5_batch24_chunk50 \

# /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241020/table/241012_upright_blue_cup_1_lr1e5_batch24_chunk50

# python3 run_franka_act_inference.py \
#         --cfg_path ./cfgs/act/config_franka.yaml \
#         --camera_names left right top \
#         --ckpt_dir /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241020/table/241012_upright_blue_cup_1_lr1e5_batch24_chunk50 \
#         --ckpt_name agent_last.ckpt \
#         --agent_class ACT \
#         --chunk_size 50 \
#         --lr 1e-5 --kl_weight 10 \
#         --temporal_agg \

######################## 
# 24101516_pick_bread_plate_1

# /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241020/table/24101516_pick_bread_plate_1_lr1e5_batch24_chunk50
# agent_epoch_40_seed_1.ckpt


# python3 run_franka_act_inference.py \
#         --cfg_path ./cfgs/act/config_franka.yaml \
#         --camera_names left right top \
#         --ckpt_dir /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241020/table/24101516_pick_bread_plate_1_lr1e5_batch24_chunk50 \
#         --ckpt_name agent_epoch_30_seed_1.ckpt \
#         --agent_class ACT \
#         --chunk_size 50 \
#         --lr 1e-5 --kl_weight 10 \
#         --temporal_agg \

####################
# /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241022/table
# 24101516_pick_bread_plate_1_lr1e5_batch24_chunk25
# 241015_pick_bread_plate_1_lr1e5_batch24_chunk10
# 241015_pick_bread_plate_1_lr1e5_batch24_chunk25
# 241015_pick_bread_plate_1_lr1e5_batch24_chunk50


# python3 run_franka_act_inference.py \
#         --cfg_path ./cfgs/act/config_franka.yaml \
#         --camera_names left right top \
#         --ckpt_dir /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241022/table/24101516_pick_bread_plate_1_lr1e5_batch24_chunk25 \
#         --ckpt_name agent_last.ckpt \
#         --agent_class ACT \
#         --chunk_size 25 \
#         --lr 1e-5 --kl_weight 10 \
#         --temporal_agg \

# python3 run_franka_act_inference.py \
#         --cfg_path ./cfgs/act/config_franka.yaml \
#         --camera_names left right top \
#         --ckpt_dir /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241022/table/241015_pick_bread_plate_1_lr1e5_batch24_chunk10 \
#         --ckpt_name agent_last.ckpt \
#         --agent_class ACT \
#         --chunk_size 10 \
#         --lr 1e-5 --kl_weight 10 \
#         --temporal_agg \

python3 run_franka_act_inference.py \
        --cfg_path ./cfgs/act/config_franka.yaml \
        --camera_names left right top \
        --ckpt_dir /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241022/table/241015_pick_bread_plate_1_lr1e5_batch24_chunk50 \
        --ckpt_name agent_last.ckpt \
        --agent_class ACT \
        --chunk_size 50 \
        --lr 1e-5 --kl_weight 10 \
        --temporal_agg \

# python3 run_franka_act_inference.py \
#         --cfg_path ./cfgs/act/config_franka.yaml \
#         --camera_names left right top \
#         --ckpt_dir /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241022/table/241015_pick_bread_plate_1_lr1e5_batch24_chunk50 \
#         --ckpt_name agent_last.ckpt \
#         --agent_class ACT \
#         --chunk_size 50 \
#         --lr 1e-5 --kl_weight 10 \
#         --temporal_agg \



# python3 run_franka_act_inference_new.py \
#         --cfg_path ./cfgs/act/config_franka.yaml \
#         --camera_names left right top \
#         --ckpt_dir /media/ps/wk/all_ckpt/action_frame_ckpt/ckpt_dir/ACT_camlrt/241022/table/241015_pick_bread_plate_1_lr1e5_batch24_chunk50 \
#         --ckpt_name agent_last.ckpt \
#         --agent_class ACT \
#         --chunk_size 50 \
#         --lr 1e-5 --kl_weight 10 \
#         --temporal_agg \
