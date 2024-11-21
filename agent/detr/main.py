# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)  # will be overridden
    parser.add_argument("--lr_backbone", default=1e-5, type=float)  # will be overridden
    # parser.add_argument("--batch_size", default=2, type=int)  # not used
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    # parser.add_argument("--epochs", default=300, type=int)  # not used
    parser.add_argument("--lr_drop", default=200, type=int)  # not used
    parser.add_argument("--clip_max_norm", default=0.1, type=float,  # not used
        help="gradient clipping max norm",
    )

    # Model parameters
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet18",
        type=str,  # will be overridden
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--camera_names",
        default=[],
        type=list,  # will be overridden
        help="A list of camera names",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=4,
        type=int,  # will be overridden
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,  # will be overridden
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,  # will be overridden
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,  # will be overridden
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,  # will be overridden
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries",
        default=400,
        type=int,  # will be overridden
        help="Number of query slots",
    )
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    # parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--agent_class', action='store', type=str, help='agent_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument("--temporal_agg", action="store_true")

    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class', required=False)
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim', required=False)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--action_dim', action='store', type=int, required=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='load_ckpt_path', required=False)
    parser.add_argument('--no_encoder', action='store_true')
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)
    
    # parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--no_sepe_backbone', action='store_true')
    parser.add_argument('--use_lang', action='store_true')
    parser.add_argument('--input_state_acthead', action='store_true')
    parser.add_argument('--ddp_port', action='store', type=str, default='12355')
    parser.add_argument('--use_ddp', action='store_true', default=False)

    # norm 1 for diffusion: ((x-min)/(max-min)) * 2 - 1
    # norm 2 for act: (x-mean)/std
    parser.add_argument('--act_norm_class', type=str, default='norm1')

    # for eval
    parser.add_argument('--ckpt_name', type=str, default='null')
    parser.add_argument('--raw_lang', type=str, default='null')

    # exp type: franka_3rgb, franka_1rgn, ur, songling, tiangong, sim
    parser.add_argument('--exp_type', type=str, default='franka_3rgb')

    # for droid_diffusion, 
    # parser.add_argument('--lr_backbone', default=1e-4, type=float, help='lr_backbone')
    parser.add_argument('--pool_class', type=str, default='null')
    parser.add_argument('--stsm_num_kp', type=int, default=512)
    parser.add_argument('--img_fea_dim', type=int, default=512)
    parser.add_argument('--cond_obs_dim', type=int, default=512)
    parser.add_argument('--num_noise_samples', type=int, default=8)

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    print('args**:',args)
    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

