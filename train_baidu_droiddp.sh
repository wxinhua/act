#!/bin/bash
export HF_HOME=/media/users/wk/huggingface_model
export TORCH_HOME=/media/users/wk/torch_model

# example:  

TASK_NAME=""
EXP_TYPE=""
DAY="251127"
TG_MODE="mode1"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --task_name) TASK_NAME="$2"; shift ;;
        --exp_type) EXP_TYPE="$2"; shift ;;
        --day) DAY="$2"; shift ;;
        --tg_mode) TG_MODE="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$TASK_NAME" ] || [ -z "$EXP_TYPE" ]; then
    echo "Error: Missing required parameters."
    echo "Usage: $0 --task_name <task_name> --exp_type <exp_type>"
    exit 1
fi

CMD=""
case $EXP_TYPE in
    franka_3rgb)
        CMD="python3 train_algo.py \
        --task_name $TASK_NAME \
        --camera_names camera_left camera_right camera_top \
        --ckpt_dir ./ckpt_dir/DroidDiffusion_camlrt/${DAY}/table/${EXP_TYPE}_${TASK_NAME}_lr1e4_batch48_chunk16 \
        --exp_type $EXP_TYPE \
        --agent_class DroidDiffusion \
        --batch_size_train 32 --batch_size_val 32 \
        --chunk_size 16 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --lr_backbone 1e-4 \
        --backbone resnet50 \
        --act_norm_class norm1 \
        --lr_scheduler CosineLR \
        --pool_class SpatialSoftmax --use_data_aug \
        --num_steps 75000 --eval_every 75001 --validate_every 250 --save_every 30000 \
        --use_wandb \
        --wandb_name DroidDiffusion_${EXP_TYPE}camlrt_${TASK_NAME}_lr1e4_batch48_chunk16"
        ;;
    franka_1rgb)
        CMD="python3 train_algo.py \
        --task_name $TASK_NAME \
        --camera_names camera_top \
        --ckpt_dir ./ckpt_dir/DroidDiffusion_camt/${DAY}/table/${EXP_TYPE}_${TASK_NAME}_lr1e4_batch48_chunk16 \
        --exp_type $EXP_TYPE \
        --agent_class DroidDiffusion \
        --batch_size_train 48 --batch_size_val 48 \
        --chunk_size 16 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --lr_backbone 1e-4 \
        --backbone resnet50 \
        --act_norm_class norm1 \
        --lr_scheduler CosineLR \
        --pool_class SpatialSoftmax --use_data_aug \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
        --use_wandb \
        --wandb_name DroidDiffusion_${EXP_TYPE}_camlrt_${TASK_NAME}_lr1e4_batch48_chunk16"
        ;;
    simulation_4rgb)
        CMD="python3 train_algo.py \
        --task_name $TASK_NAME \
        --camera_names camera_left_external camera_right_external camera_front_external \
        --ckpt_dir ./ckpt_dir/DroidDiffusion_camlrt/${DAY}/table/${EXP_TYPE}_${TASK_NAME}_lr1e4_batch48_chunk16 \
        --exp_type $EXP_TYPE \
        --agent_class DroidDiffusion \
        --batch_size_train 32 --batch_size_val 32 \
        --chunk_size 16 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --lr_backbone 1e-4 \
        --backbone resnet50 \
        --act_norm_class norm1 \
        --lr_scheduler CosineLR \
        --pool_class SpatialSoftmax --use_data_aug \
        --num_steps 75000 --eval_every 75001 --validate_every 250 --save_every 30000 \
        --use_wandb \
        --wandb_name DroidDiffusion_${EXP_TYPE}_camlrt_${TASK_NAME}_lr1e4_batch48_chunk16"
        ;;
    songling_3rgb)
        CMD="python3 train_algo.py \
        --task_name $TASK_NAME \
        --camera_names camera_front camera_left_wrist camera_right_wrist \
        --ckpt_dir ./ckpt_dir/DroidDiffusion_camlrt/${DAY}/table/${EXP_TYPE}_${TASK_NAME}_lr1e4_batch48_chunk16 \
        --exp_type $EXP_TYPE \
        --agent_class DroidDiffusion \
        --batch_size_train 32 --batch_size_val 32 \
        --chunk_size 16 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --lr_backbone 1e-4 \
        --backbone resnet50 \
        --act_norm_class norm1 \
        --lr_scheduler CosineLR \
        --pool_class SpatialSoftmax --use_data_aug \
        --num_steps 75000 --eval_every 75001 --validate_every 250 --save_every 30000 \
        --use_wandb \
        --wandb_name DroidDiffusion_${EXP_TYPE}_camlrt_${TASK_NAME}_lr1e4_batch48_chunk16"
        ;;
    ur_1rgb)
        CMD="python3 train_algo.py \
        --task_name $TASK_NAME \
        --camera_names camera_top \
        --ckpt_dir ./ckpt_dir/DroidDiffusion_camt/${DAY}/table/${EXP_TYPE}_${TASK_NAME}_lr1e4_batch48_chunk16 \
        --exp_type $EXP_TYPE \
        --agent_class DroidDiffusion \
        --batch_size_train 48 --batch_size_val 48 \
        --chunk_size 16 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --lr_backbone 1e-4 \
        --backbone resnet50 \
        --act_norm_class norm1 \
        --lr_scheduler CosineLR \
        --pool_class SpatialSoftmax --use_data_aug \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
        --use_wandb \
        --wandb_name DroidDiffusion_${EXP_TYPE}_camlrt_${TASK_NAME}_lr1e4_batch48_chunk16"
        ;;
    tiangong_1rgb)
        CMD="python3 train_algo.py \
        --task_name place_button \
        --camera_names camera_top \
        --ckpt_dir ./ckpt_dir/DroidDiffusion_camt/${DAY}/table/${EXP_TYPE}_${TG_MODE}_${TASK_NAME}_lr1e4_batch48_chunk16 \
        --exp_type $EXP_TYPE \
        --agent_class DroidDiffusion \
        --batch_size_train 48 --batch_size_val 48 \
        --chunk_size 16 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --lr_backbone 1e-4 \
        --backbone resnet50 \
        --act_norm_class norm1 \
        --lr_scheduler CosineLR \
        --tg_mode $TG_MODE \
        --pool_class SpatialSoftmax --use_data_aug \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
        --use_wandb \
        --wandb_name DroidDiffusion_${EXP_TYPE}_${TG_MODE}_camlrt_${TASK_NAME}_lr1e4_batch48_chunk16"
        ;;
    *)
        echo "Invalid TYPE: $TYPE"
        exit 1
        ;;
esac

echo $CMD
CUDA_VISIBLE_DEVICES=0 $CMD

echo mkdir -p /media/results/droiddp/${EXP_TYPE}
mkdir -p /media/results/droiddp/${EXP_TYPE}

echo cp -ri ./ckpt_dir/* /media/results/droiddp/${EXP_TYPE}/
cp -rf ./ckpt_dir/* /media/results/droiddp/${EXP_TYPE}/
