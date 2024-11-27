#!/bin/bash
export HF_HOME=/media/users/wk/huggingface_model
export TORCH_HOME=/media/users/wk/torch_model

# example:  

TASK_NAME=""
EXP_TYPE=""
DAY="251124"
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
        --ckpt_dir ./ckpt_dir/ACT_camlrt/${DAY}/table/${EXP_TYPE}_${TASK_NAME}_lr1e5_batch24_chunk50 \
        --exp_type $EXP_TYPE \
        --agent_class ACT \
        --batch_size_train 48 --batch_size_val 48 \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --kl_weight 10 \
        --backbone resnet18 \
        --act_norm_class norm2 \
        --lr_scheduler CosineLR \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
        --use_wandb \
        --wandb_name ACT_${EXP_TYPE}camlrt_${TASK_NAME}_lr1e5_batch24_chunk50"
        ;;
    franka_1rgb)
        CMD="python3 train_algo.py \
        --task_name $TASK_NAME \
        --camera_names camera_top \
        --ckpt_dir ./ckpt_dir/ACT_camt/${DAY}/table/${EXP_TYPE}_${TASK_NAME}_lr1e5_batch24_chunk50 \
        --exp_type $EXP_TYPE \
        --agent_class ACT \
        --batch_size_train 48 --batch_size_val 48 \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --kl_weight 10 --use_lang \
        --backbone resnet18 \
        --act_norm_class norm2 \
        --lr_scheduler CosineLR \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
        --use_wandb \
        --wandb_name ACT_${EXP_TYPE}_camlrt_${TASK_NAME}_lr1e5_batch24_chunk50"
        ;;
    simulation_4rgb)
        CMD="python3 train_algo.py \
        --task_name $TASK_NAME \
        --camera_names camera_left_external camera_right_external camera_front_external \
        --ckpt_dir ./ckpt_dir/ACT_camlrt/${DAY}/table/${EXP_TYPE}_${TASK_NAME}_lr1e5_batch24_chunk50 \
        --exp_type $EXP_TYPE \
        --agent_class ACT \
        --batch_size_train 24 --batch_size_val 24 \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 5e-5 --kl_weight 10 --use_lang \
        --backbone resnet18 \
        --act_norm_class norm2 \
        --lr_scheduler CosineLR \
        --num_steps 20 --eval_every 21 --validate_every 10 --save_every 10 \
        --use_wandb \
        --wandb_name ACT_${EXP_TYPE}_camlrt_${TASK_NAME}_lr1e5_batch24_chunk50"
        ;;
    songling_3rgb)
        CMD="python3 train_algo.py \
        --task_name $TASK_NAME \
        --camera_names camera_front camera_left_wrist camera_right_wrist \
        --ckpt_dir ./ckpt_dir/ACT_camlrt/${DAY}/table/${EXP_TYPE}_${TASK_NAME}_lr1e5_batch24_chunk50 \
        --exp_type $EXP_TYPE \
        --agent_class ACT \
        --batch_size_train 48 --batch_size_val 48 \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --kl_weight 10 \
        --backbone resnet18 \
        --act_norm_class norm2 \
        --lr_scheduler CosineLR \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
        --use_wandb \
        --wandb_name ACT_${EXP_TYPE}_camlrt_${TASK_NAME}_lr1e5_batch24_chunk50"
        ;;
    ur_1rgb)
        CMD="python3 train_algo.py \
        --task_name $TASK_NAME \
        --camera_names camera_top \
        --ckpt_dir ./ckpt_dir/ACT_camt/${DAY}/table/${EXP_TYPE}_${TASK_NAME}_lr1e5_batch24_chunk50 \
        --exp_type $EXP_TYPE \
        --agent_class ACT \
        --batch_size_train 48 --batch_size_val 48 \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --kl_weight 10 \
        --backbone resnet18 \
        --act_norm_class norm2 \
        --lr_scheduler CosineLR \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
        --use_wandb \
        --wandb_name ACT_${EXP_TYPE}_camlrt_${TASK_NAME}_lr1e5_batch24_chunk50"
        ;;
    tiangong_1rgb)
        CMD="python3 train_algo.py \
        --task_name place_button \
        --camera_names camera_top \
        --ckpt_dir ./ckpt_dir/ACT_camt/${DAY}/table/${EXP_TYPE}_${TG_MODE}_${TASK_NAME}_lr1e5_batch24_chunk50 \
        --exp_type $EXP_TYPE \
        --agent_class ACT \
        --batch_size_train 48 --batch_size_val 48 \
        --chunk_size 50 --hidden_dim 512 --dim_feedforward 3200 \
        --lr 1e-4 --kl_weight 10 \
        --backbone resnet18 \
        --act_norm_class norm2 \
        --lr_scheduler CosineLR \
        --tg_mode $TG_MODE \
        --num_steps 50000 --eval_every 50001 --validate_every 250 --save_every 25000 \
        --use_wandb \
        --wandb_name ACT_${EXP_TYPE}_${TG_MODE}_camlrt_${TASK_NAME}_lr1e5_batch24_chunk50"
        ;;
    *)
        echo "Invalid TYPE: $TYPE"
        exit 1
        ;;
esac

echo $CMD
CUDA_VISIBLE_DEVICES=0 $CMD

echo mkdir -p /media/results/act/${EXP_TYPE}
mkdir -p /media/results/act/${EXP_TYPE}

echo cp -ri ./ckpt_dir/* /media/results/act/${EXP_TYPE}/
cp -rf ./ckpt_dir/* /media/results/act/${EXP_TYPE}/
