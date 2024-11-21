#!/usr/bin/env python3
# from memory_profiler import profile

import warnings
import os
from pathlib import Path
import yaml
import argparse
import ast
import pickle
from copy import deepcopy
from tqdm import tqdm
import sys
sys.path.append('/home/ps/Dev/inrocs/')
import gc

import torch
import cv2
import numpy as np

import sys
import os
print('os.getcwd():',os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'action_frame'))

from utils import set_seed_everywhere
from agent.act import ACTPolicy

import time

def make_agent(args, agent_config):
    if args['agent_class'] == "ACT":
        agent_config['num_queries'] = args['chunk_size']
        agent_config['chunk_size'] = args['chunk_size']
        agent_config['camera_names'] = args['camera_names']
        agent_config['use_depth_image'] = args['use_depth_image']
        agent_config['use_robot_base'] = args['use_robot_base']
        print('agent_config:',agent_config)
        agent = ACTPolicy(agent_config)
    else:
        raise NotImplementedError
    return agent, agent_config


def make_optimizer(args, agent):
    if args['agent_class'] == 'ACT':
        optimizer = agent.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


class VLAIL:
    def __init__(self, stats, args=None):
        # self.work_dir = Path.cwd()
        self.work_dir = os.path.join(Path.cwd(),'action_frame')
        print(f"agent workspace: {self.work_dir}")

        self.args = args

        if self.args['agent_class'] == 'ACT':
            # cfg_path = os.path.join(self.work_dir, 'cfgs/act/config_franka.yaml')
            cfg_path = self.args['cfg_path']
            with open(cfg_path, 'r', encoding='utf-8') as fin:
                self.config = yaml.load(fin, Loader=yaml.SafeLoader)
            self.seed = self.config['seed']
            self.device = torch.device(self.config['device'])
            self.config['robot_infor']['use_robot_base'] = self.args['use_robot_base']
            self.config['robot_infor']['camera_names'] = self.args['camera_names']
            # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.args['config'] = self.config
            set_seed_everywhere(self.seed)

        self.agent, self.agent_config = make_agent(self.args, self.config['agent_config'])

        self.chunk_size = self.args['chunk_size']
        self.stats = stats

        # ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        ckpt_path = os.path.join(self.args['ckpt_dir'], self.args['ckpt_name'])
        loading_status = self.agent.deserialize(torch.load(ckpt_path))
        print(loading_status)
        self.agent.cuda()
        self.agent.eval()

        print(f'Loaded: {ckpt_path}')

    def eval_act(self, curr_image=None, curr_depth=None, qpos=None):

        # load policy and stats
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        cam_images = (curr_image / 255.0).astype(np.float32)
        curr_image = torch.from_numpy(cam_images).cuda().unsqueeze(0)
        # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        # if curr_depth is not None:
        #     curr_depth = torch.from_numpy(curr_depth / 255.0).float().cuda().unsqueeze(0)

        start_time = time.time()
        with torch.inference_mode():
            all_actions = self.agent(curr_image, curr_depth, qpos)

        inference_actions = all_actions.cpu().detach().numpy()

        end_time = time.time()
        print("model cost time: ", end_time - start_time)
        return inference_actions


def main():
    ckpt_dir = '/home/zz/Project/action_method/action_frame/ckpt_dir/930/kitchen/breadcook_lr2e-5_batch32_chunk50'
    ckpt_name = 'agent_epoch_0_seed_1.ckpt'
    chunk_size = 50
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    vla_method = VLAIL(ckpt_dir=ckpt_dir,
                       ckpt_name=ckpt_name,
                       chunk_size=chunk_size,
                       camera_names=camera_names,
                       agent_class='ACT')
    vla_method.eval_act()



# ssh agilex@10.11.15.67 agx
if __name__ == "__main__":

    main()