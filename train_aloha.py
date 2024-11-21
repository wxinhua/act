#!/usr/bin/env python3
from memory_profiler import profile

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
import gc

import wandb
import torch
import cv2
import numpy as np

from utils import set_seed_everywhere, compute_dict_mean, detach_dict, plot_history
from agent.act import ACTPolicy


import time

def make_agent(args, agent_config):
    if args['agent_class'] == "ACT":
        agent_config['lr'] = args['lr']
        agent_config['num_queries'] = args['chunk_size']
        agent_config['kl_weight'] = args['kl_weight']
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

def make_scheduler(arg, optimizer):
    if arg['lr_scheduler'] == 'MultiStepLR':
        lr_schedulers = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8], gamma=0.9)
    elif arg['lr_scheduler'] == 'CosineLR':
        lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg['num_epochs'])
    else:
        raise ValueError('Not supported or recognized lr_scheduler_type', arg['lr_scheduler'])
    return lr_schedulers

def forward_pass(data, agent, use_depth_image):
    # image_data, qpos_data, action_data, is_pad = data
    image_data, depth_data, qpos_data, action_data, is_pad = (data[0].cuda(), data[1].cuda(), data[2].cuda(),
                                                              data[3].cuda(), data[4].cuda())
    if use_depth_image:
        depth_data = depth_data.cuda()
    else:
        depth_data = None
    # image_depth_data = None
    return agent(image_data, depth_data, qpos_data, action_data, is_pad) # TODO remove None


class VLAIL:
    def __init__(self, args=None):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.args = args
        if args['agent_class'] == 'ACT':
            cfg_path = './cfgs/act/config_aloha.yaml'
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

        # total trajectories for training
        self.batch_size_train = self.args['batch_size_train']
        self.batch_size_val = self.args['batch_size_val']

        self.chunk_size = self.args['chunk_size']

    def execute(self, dataset_dir):
        dataset_dir = os.path.join(dataset_dir, self.args['scene_name'], self.args['task_name'])
        print('dataset_dir:',dataset_dir)
        if not self.args['eval_set']:
            wandb.init(project=self.args["wandb_name"], reinit=True,
                       name=f"{self.args['wandb_name']}_B{self.args['batch_size_train']}_{self.args['lr']}")
            wandb.config.update(self.args)

        if self.args['agent_class'] == "ACT":
            if self.args['init_all']:
                from dataset_load.dataset_aloha_initall import load_data
            else:
                from dataset_load.dataset_aloha import load_data

            train_dataloader, val_dataloader, stats, is_sim = load_data(dataset_dir,
                                                                        self.config['robot_infor'],
                                                                        self.batch_size_train,
                                                                        self.batch_size_val,
                                                                        chunk_size=self.chunk_size,
                                                                        cutoff=0)
            # save dataset stats
            if not os.path.isdir(self.args['ckpt_dir']):
                os.makedirs(self.args['ckpt_dir'])
            stats_path = os.path.join(self.args['ckpt_dir'], f'dataset_stats.pkl')
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)

            best_ckpt_info = self.train(train_dataloader, val_dataloader)

            best_epoch, min_val_loss, best_state_dict = best_ckpt_info

            # save best checkpoint
            ckpt_path = os.path.join(self.args['ckpt_dir'], f'agent_best.ckpt')
            torch.save(best_state_dict, ckpt_path)
            print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

        wandb.finish()

    # @profile
    def train(self, train_dataloader, val_dataloader):

        num_epochs = self.args['num_epochs']
        ckpt_dir = self.args['ckpt_dir']

        self.agent.cuda()
        optimizer = make_optimizer(self.args, self.agent)
        scheduler = make_scheduler(self.args, optimizer)
        train_history = []
        validation_history = []
        min_val_loss = np.inf
        best_ckpt_info = None

        run_step = 0

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch}')
            # validation
            with torch.inference_mode():
                self.agent.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, self.agent, self.args['use_depth_image'])
                    epoch_dicts.append(forward_dict)
                    # del data
                    # gc.collect()
                    if batch_idx > 1000:
                        break
                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(self.agent.serialize()))
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

            self.agent.train()
            # epoch_iterator = tqdm(train_dataloader, desc= "Training (Epoch %d)" % epoch)
            with tqdm(total=len(train_dataloader), ncols=100) as t:

                for batch_idx, data in enumerate(train_dataloader):
                    # training
                    forward_dict = forward_pass(data, self.agent, self.args['use_depth_image'])
                    # backward
                    loss = forward_dict['loss']
                    # save_dict = forward_dict['state_dict'].item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    save_forward_dict = detach_dict(forward_dict)
                    # print('save_forward_dict:',save_forward_dict)
                    train_history.append(save_forward_dict)
                    run_step += 1

                    t.set_description(desc="Epoch %d" % epoch)
                    t.set_postfix(steps=run_step, loss=loss.data.item(), lr=optimizer.state_dict()['param_groups'][0]['lr'])
                    t.update(1)
                    # sys.stdout.flush() # 在每次更新后手动刷新输出缓冲
                    wandb.log(save_forward_dict, step=run_step)  # not great, make training 1-2% slower
                    # del data
                    # del forward_dict
                    # del save_forward_dict
                    # gc.collect()

            scheduler.step()

            epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
            epoch_train_loss = epoch_summary['loss']

            wandb.log({
                "Training loss": epoch_train_loss,
                # "Validation loss": epoch_val_loss,
                "Epoch": epoch})

            print(f'Train loss: {epoch_train_loss:.5f}')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

            if epoch % self.args['save_epoch'] == 0:
                ckpt_path = os.path.join(ckpt_dir, f'agent_epoch_{epoch}_seed_{self.seed}.ckpt')
                torch.save(self.agent.serialize(), ckpt_path)
                plot_history(train_history, validation_history, epoch, ckpt_dir, self.seed)
                # success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)

        ckpt_path = os.path.join(ckpt_dir, f'agent_last.ckpt')
        torch.save(self.agent.serialize(), ckpt_path)

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f'agent_epoch_{best_epoch}_seed_{self.seed}.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Training finished:\nSeed {self.seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

        # save training curves
        plot_history(train_history, validation_history, num_epochs, ckpt_dir, self.seed)

        return best_ckpt_info

    def eval_bc(self, ckpt_name, val_dataloader, save_episode=True, num_rollouts=10):
        # load policy and stats
        ckpt_path = os.path.join(self.args['ckpt_dir'], ckpt_name)
        loading_status = self.agent.deserialize(torch.load(ckpt_path))
        print(loading_status)
        self.agent.eval()

        print(f'Loaded: {ckpt_path}')

        # save dataset stats
        stats_path = os.path.join(self.args['ckpt_dir'], f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        if self.args['agent_class'] == 'ACT':
            pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
            post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        else:
            raise NotImplementedError
        # load environment

        query_frequency = self.agent_config['num_queries']
        if temporal_agg:
            query_frequency = 1
            num_queries = self.agent_config['num_queries']

        max_timesteps = int(400 * 1)  # may increase for real-world tasks

        for batch_idx, data in enumerate(val_dataloader):
            forward_dict = forward_pass(data, self.agent, self.args['use_depth_image'])
            print('forward_dict:',forward_dict)


def main(args):

    vla_method = VLAIL(args)
    vla_method.execute(args['dataset_dir'])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', action='store', type=str, help='cfg_path', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', required=True)

    parser.add_argument('--scene_name', action='store', type=str, help='scene_name', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)

    parser.add_argument('--camera_names', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--agent_class', action='store', type=str, help='agent_class, capitalize', required=True)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--batch_size_train', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--batch_size_val', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr_scheduler', action='store', type=str, help='lr_scheduler', required=True)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image', default=False, required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base', default=False, required=False)

    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--init_all', action='store', type=bool, help='init-load-dataset',default=False, required=False)

    parser.add_argument('--save_epoch', action='store', type=int, default=1, help='save_every', required=False)
    parser.add_argument('--wandb_name', '--wandb_name', action='store', type=str, help='run name for logs', required=True)
    parser.add_argument('--eval_set', action='store_true')
    # parser.add_argument('--sim_aloha', action='store_true')

    main(vars(parser.parse_args()))