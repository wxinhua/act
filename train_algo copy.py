#!/usr/bin/env python3
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
import psutil
import wandb
import torch
import cv2
import numpy as np

from torchvision import transforms
import torchvision

import datetime
import logging

torchvision.disable_beta_transforms_warning()

os.environ['DEVICE'] = "cuda"
WANDB_ENTITY = 'zedwk' #None
WANDB_API_KEY = 'd8b6ecb1e0ea3f72b9d017d7868809be770b283e' #None
# os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from utils import set_seed_everywhere, compute_dict_mean, detach_dict, plot_history
from agent.act import ACTPolicy
from agent.droid_difffusion import DroidDiffusionPolicy

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def make_agent(args, agent_config, rank=None):
    if args['agent_class'] == "ACT":
        agent_config['num_queries'] = args['chunk_size']
        agent_config['chunk_size'] = args['chunk_size']
        agent_config['camera_names'] = args['camera_names']
        agent_config['use_depth_image'] = args['use_depth_image']
        agent_config['no_sepe_backbone'] = args['no_sepe_backbone']
        agent_config['use_lang'] = args['use_lang']
        # print('agent_config:',agent_config)
        agent = ACTPolicy(agent_config, rank)
    elif args['agent_class'] == "DroidDiffusion":
        agent_config['num_queries'] = args['chunk_size']
        agent_config['camera_names'] = args['camera_names']
        agent_config['pool_class'] = args['pool_class']
        agent_config['use_depth_image'] = args['use_depth_image']
        agent_config['use_lang'] = args['use_lang']
        # print('agent_config:',agent_config)
        agent = DroidDiffusionPolicy(agent_config, rank)
    else:
        raise NotImplementedError
    return agent, agent_config


def make_optimizer(args, agent):
    policy_class_list = ['ACT', 'DroidDiffusion']
    if args['agent_class'] in policy_class_list:
        if torch.cuda.device_count() > 1:
            optimizer = agent.module.configure_optimizers()
        else:
            optimizer = agent.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def make_scheduler(arg, optimizer):
    if arg['lr_scheduler'] == 'MultiStepLR':
        # [2,4,6,8]
        lr_schedulers = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 10000, 15000, 20000], gamma=0.9)
    elif arg['lr_scheduler'] == 'CosineLR':
        lr_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg['num_steps'])
    else:
        raise ValueError('Not supported or recognized lr_scheduler_type', arg['lr_scheduler'])
    return lr_schedulers

def forward_pass(data, agent, use_lang, use_raw_lang, use_depth_image):
    if use_raw_lang:
        image_data, depth_data, qpos_data, action_data, is_pad, lang_embed, lang_raw = data
    else:
        image_data, depth_data, qpos_data, action_data, is_pad = data
    
    # process = psutil.Process(os.getpid())
    # print_memory_info(process, 101)

    image_data, depth_data, qpos_data, action_data, is_pad = image_data.cuda(), depth_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    
    # print_memory_info(process, 102)

    if not use_lang:
        lang_embed = None
    if lang_embed is not None:
        lang_embed = lang_embed.cuda()
    
    if use_raw_lang:
        return agent(qpos_data, image_data, depth_data, action_data, is_pad, language_distilbert=lang_embed, lang_raw=lang_raw)
    
    # print_memory_info(process, 103)

    return agent(qpos_data, image_data, depth_data, action_data, is_pad, language_distilbert=lang_embed) # TODO remove None

def old_forward_pass(data, agent, use_lang, use_raw_lang, use_depth_image):
    if use_raw_lang:
        image_data, depth_data, qpos_data, action_data, is_pad, lang_embed, lang_raw = data
    else:
        image_data, depth_data, qpos_data, action_data, is_pad, lang_embed = data
    image_data, depth_data, qpos_data, action_data, is_pad = image_data.cuda(), depth_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()

    if not use_lang:
        lang_embed = None
    if lang_embed is not None:
        lang_embed = lang_embed.cuda()
    
    if use_raw_lang:
        return agent(qpos_data, image_data, depth_data, action_data, is_pad, language_distilbert=lang_embed, lang_raw=lang_raw)
    return agent(qpos_data, image_data, depth_data, action_data, is_pad, language_distilbert=lang_embed) # TODO remove None

def print_memory_info(process, idx):
    used_bytes = process.memory_info().rss
    used_MB = used_bytes / (1024*1024)
    print(f"{idx}. used_MB: {used_MB}")

class VLAIL:
    def __init__(self, args=None):
        self.work_dir = Path.cwd()

        self.args = args
        if not os.path.isdir(self.args['ckpt_dir']):
            os.makedirs(self.args['ckpt_dir'])

        self.set_log_file_handler()

        # print(f"workspace: {self.work_dir}")
        self.logger.info(f"workspace: {self.work_dir}")

        cfg_path = os.path.join('./cfgs', f'{args["agent_class"]}', f'config_{args["exp_type"]}.yaml')
        # print(f"cfg_path: {cfg_path}")
        self.logger.info(f"cfg_path: {cfg_path}")
        with open(cfg_path, 'r', encoding='utf-8') as fin:
            self.config = yaml.load(fin, Loader=yaml.SafeLoader)
        self.seed = self.config['seed']
        self.config['robot_infor']['camera_names'] = self.args['camera_names']
        if self.args['use_depth_image']:
            self.config['robot_infor']['camera_sensors'] = ['rgb_images', 'depth_images']
        else:
            self.config['robot_infor']['camera_sensors'] = ['rgb_images']
        
        for key in self.config['agent_config']:
            if key in self.args:
                self.config['agent_config'][key] = self.args[key]

        self.args['config'] = self.config
        set_seed_everywhere(self.seed)

        self.agent, agent_config = make_agent(self.args, self.config['agent_config'], self.args['rank'])
        self.config['agent_config'] = agent_config

        # total trajectories for training
        self.batch_size_train = self.args['batch_size_train']
        self.batch_size_val = self.args['batch_size_val']

        self.chunk_size = self.args['chunk_size']

    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)

        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = self.args['ckpt_dir']

        now_str = datetime.datetime.now().__str__().replace(' ','_')

        self.log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)

    def execute(self, task_name):
        # exp type: franka_3rgb, franka_1rgb, ur_1rgb, songling_3rgb, tiangong_1rgb, sim
        # if self.args['exp_type'] == 'franka_3rgb':
        #     from cfgs.constants_config import Franka_3rgb_TASK_CONFIGS as TASK_CONFIGS
        # elif self.args['exp_type'] == 'franka_1rgb':
        #     from cfgs.constants_config import Franka_1rgb_TASK_CONFIGS as TASK_CONFIGS
        # elif self.args['exp_type'] == 'ur_1rgb':
        #     from cfgs.constants_config import UR_1rgb_TASK_CONFIGS as TASK_CONFIGS
        # elif self.args['exp_type'] == 'songling_3rgb':
        #     from cfgs.constants_config import Songling_3rgb_TASK_CONFIGS as TASK_CONFIGS
        # elif self.args['exp_type'] == 'tiangong_1rgb':
        #     from cfgs.constants_config import Tiangong_1rgb_TASK_CONFIGS as TASK_CONFIGS
        # task_config = TASK_CONFIGS[task_name]

        # dataset_dir = task_config['dataset_dir']

        # sample_weights = task_config.get('sample_weights', None)
        # train_ratio > 1: random train_ratio num of traj
        # train_ratio -> [0, 1]: ratio of traj
        # train_ratio = task_config.get('train_ratio', [0.95])
        # name_filter = task_config.get('name_filter', lambda n: True)

        Franka_1rgb_DATA_DIR = '/media/data/h5_franka_1rgb'
        # Franka_3rgb_DATA_DIR = '/media/data/h5_franka_3rgb'
        Franka_3rgb_DATA_DIR = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_franka_3rgb'
        Songling_3rgb_DATA_DIR = '/media/data/h5_songling_3rgb'
        Tiangong_1rgb_DATA_DIR = '/media/data/h5_tiangong_1rgb'
        UR_1rgb_DATA_DIR = '/media/data/h5_ur_1rgb'

        if self.args['exp_type'] == 'franka_3rgb':
            robot_data_dir = Franka_3rgb_DATA_DIR
        elif self.args['exp_type'] == 'franka_1rgb':
            robot_data_dir = Franka_1rgb_DATA_DIR
        elif self.args['exp_type'] == 'ur_1rgb':
            robot_data_dir = UR_1rgb_DATA_DIR
        elif self.args['exp_type'] == 'songling_3rgb':
            robot_data_dir = Songling_3rgb_DATA_DIR
        elif self.args['exp_type'] == 'tiangong_1rgb':
            robot_data_dir = Tiangong_1rgb_DATA_DIR

        dataset_dir = os.path.join(robot_data_dir, self.args['task_name'])
        sample_weights = None
        name_filter = lambda n: True
        # print(f'dataset_dir: {dataset_dir}')
        # print(f'train_ratio: {train_ratio}')
        # print(f'sample_weights: {sample_weights}')

        if self.args['use_wandb']:
            wandb.init(entity=WANDB_ENTITY, project=self.args["wandb_name"], reinit=True,
                       name=f"{self.args['wandb_name']}", dir=self.args['ckpt_dir'], mode='offline')
            wandb.config.update(self.args)
            # print('use_wandb: True')
        
        # if self.args['exp_type'] == 'franka_3rgb':
        from dataset_load.dataset_multi_robot import load_data

        train_dataloader, val_dataloader, stats = load_data(dataset_dir, self.config['robot_infor'], self.batch_size_train, self.batch_size_val, chunk_size=self.chunk_size, use_depth_image=self.args['use_depth_image'], sample_weights=sample_weights, rank=self.args['rank'], use_data_aug=self.args['use_data_aug'], act_norm_class=self.args['act_norm_class'], use_raw_lang=self.args['use_raw_lang'], name_filter=name_filter, exp_type=self.args['exp_type'], logger=self.logger)

        # save dataset stats
        stats_path = os.path.join(self.args['ckpt_dir'], f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        best_ckpt_info = self.train(train_dataloader, val_dataloader)

        best_step = best_ckpt_info['step']
        min_val_loss = best_ckpt_info['min_val_loss']

        # save best checkpoint
        ckpt_path = os.path.join(self.args['ckpt_dir'], f'agent_best.ckpt')
        torch.save(best_ckpt_info, ckpt_path)
        
        ## test ckpt
        if self.args['rank'] == 0:
            # print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
            self.logger.info(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
            resume_ckpt_path = ckpt_path
            self.agent.cpu()
            self.load_ckpt(resume_ckpt_path)

        wandb.finish()

    def load_ckpt(self, resume_ckpt_path=None):
        if resume_ckpt_path is not None:
            # print(f"Rank: {self.args['rank']}, Load resume_ckpt_path!")
            self.logger.info(f"Rank: {self.args['rank']}, Load resume_ckpt_path!")
            checkpoint = torch.load(resume_ckpt_path, map_location=torch.device('cpu'))
            loading_status = self.agent.deserialize(checkpoint['nets'])
            # if self.args['use_ddp']:
            #     checkpoint = torch.load(resume_ckpt_path, map_location=torch.device('cpu'))
            #     if self.args['agent_class'] in load_policy_list:
            #         loading_status = self.agent.deserialize(checkpoint["nets"])
            #     else:
            #         loading_status = self.agent.load_state_dict(checkpoint["nets"])
            # else:
            #     loading_status = self.agent.deserialize(torch.load(resume_ckpt_path))

            # checkpoint = torch.load(config['resume_ckpt_path'], map_location=torch.device('cpu'))
            # status = policy.load_state_dict(checkpoint["nets"])
            # print('Loaded model')
            # loading_status = status
            curr_step = checkpoint["step"]
            # print(f"Rank: {self.args['rank']}, curr_step: {curr_step}, num_steps: {self.args['num_steps']}")
            # print(f"Rank: {self.args['rank']}, Resume policy from: {resume_ckpt_path}, Status: {loading_status}")
            self.logger.info(f"Rank: {self.args['rank']}, curr_step: {curr_step}, num_steps: {self.args['num_steps']}")
            self.logger.info(f"Rank: {self.args['rank']}, Resume policy from: {resume_ckpt_path}, Status: {loading_status}")

    def train(self, train_dataloader, val_dataloader, stats=None):

        num_steps = self.args['num_steps']
        ckpt_dir = self.args['ckpt_dir']
        rank = self.args['rank']

        curr_step = 0
        resume_ckpt_path = self.args['resume_ckpt_path']
        self.load_ckpt(resume_ckpt_path)
        
        if torch.cuda.device_count() > 1:
            print(f"=== set policy to {rank}")
            self.logger.info(f"=== set policy to {rank}")
            self.agent = self.agent.to(rank)
            self.agent = DDP(self.agent, device_ids=[rank], find_unused_parameters=True)
        else:
            self.agent.cuda()

        # self.agent.cuda()
        optimizer = make_optimizer(self.args, self.agent)
        scheduler = make_scheduler(self.args, optimizer)
        if self.args['resume_ckpt_path'] is not None:
            # loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
            # print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
            checkpoint = torch.load(self.args['resume_ckpt_path'])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(f"Rank: {rank}, Resume optimizer from: {self.args['resume_ckpt_path']}")
            print(f"Rank: {rank}, Resume scheduler from: {self.args['resume_ckpt_path']}")
            self.logger.info(f"Rank: {rank}, Resume optimizer from: {self.args['resume_ckpt_path']}")
            self.logger.info(f"Rank: {rank}, Resume scheduler from: {self.args['resume_ckpt_path']}")
        
        train_history = []
        validation_history = []
        min_val_loss = np.inf
        best_ckpt_info = None

        train_dataloader = repeater(train_dataloader, self.args['num_steps'], rank)
        loss = -1.0
        
        for step in tqdm(range(curr_step, self.args['num_steps'])):
            # if step % 50 == 0:
            #     gc.collect()

            if step % self.args['validate_every'] == 0:
                if rank == 0:
                    # print('validating')
                    self.logger.info('validating')

                val_start = time.time()
                with torch.inference_mode():
                    self.agent.eval()
                    validation_dicts = []
                    for batch_idx, data in enumerate(val_dataloader):
                        forward_dict = forward_pass(data, self.agent, self.args['use_lang'], use_raw_lang=self.args['use_raw_lang'], use_depth_image=self.args['use_depth_image'])

                        # if batch_idx % 50 == 0:
                        #     gc.collect()

                        validation_dicts.append(forward_dict)
                        if batch_idx > 3:
                            break

                    validation_summary = compute_dict_mean(validation_dicts)
                    validation_history.append(validation_summary)
                    # print('********')
                    # print(f"validation_history: {validation_history}")

                    epoch_val_loss = validation_summary['loss']
                    if epoch_val_loss < min_val_loss:
                        min_val_loss = epoch_val_loss
                        if torch.cuda.device_count() > 1:
                            nets_ckpt = self.agent.module.serialize()
                        else:
                            nets_ckpt = self.agent.serialize()
                        best_ckpt_info = {"step": step+1,
                            "nets": nets_ckpt,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "loss": loss,
                            "min_val_loss": min_val_loss}
                
                if rank == 0:
                    # for k in list(validation_summary.keys()):
                    #     validation_summary[f'val_{k}'] = validation_summary.pop(k)            
                    wandb.log(validation_summary, step=step)
                    # print(f'Val loss:   {epoch_val_loss:.5f}')
                    self.logger.info(f'Val loss:   {epoch_val_loss:.5f}')
                    val_end = time.time()
                    # print(f"val time: {val_end - val_start}")
                    self.logger.info(f"val time: {val_end - val_start}")
                    summary_string = ''
                    for k, v in validation_summary.items():
                        summary_string += f'val_{k}: {v.item():.3f} '
                    # print(summary_string)
                    self.logger.info(summary_string)
            
            self.agent.train()

            print(f"-------------------")
            process = psutil.Process(os.getpid())
            print_memory_info(process, 1)

            data = next(train_dataloader)

            print_memory_info(process, 6)

            forward_dict = forward_pass(data, self.agent, self.args['use_lang'], use_raw_lang=self.args['use_raw_lang'], use_depth_image=self.args['use_depth_image'])
            
            print_memory_info(process, 7)

            loss = forward_dict['loss'].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            save_forward_dict = detach_dict(forward_dict)
            train_history.append(save_forward_dict)
            # print('********')
            # print(f"train_history: {train_history}")

            scheduler.step()
            
            print_memory_info(process, 7)

            if step > 0 and step % self.args['save_every'] == 0:    
                if rank == 0:
                    wandb.log(forward_dict, step=step) # not great, make training 1-2% slower
                    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}.ckpt')
                    if torch.cuda.device_count() > 1:
                        nets_ckpt = self.agent.module.serialize()
                    else:
                        nets_ckpt = self.agent.serialize()
                    curr_ckpt_info = { "step" :step+1,
                            "nets": nets_ckpt,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "loss":loss}
                    torch.save(curr_ckpt_info, ckpt_path)
                    plot_history(train_history, validation_history, self.args['num_steps'], ckpt_dir, self.seed)
        
        if rank == 0:
            if torch.cuda.device_count() > 1:
                nets_ckpt = self.agent.module.serialize()
            else:
                nets_ckpt = self.agent.serialize()
            curr_ckpt_info = {"step": step+1,
                        "nets": nets_ckpt,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "loss": loss}
            ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
            torch.save(curr_ckpt_info, ckpt_path)

            # best_step, min_val_loss, best_state_dict = best_ckpt_info
            best_step = best_ckpt_info["step"]
            min_val_loss = best_ckpt_info["min_val_loss"]
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{self.seed}.ckpt')
            torch.save(best_ckpt_info, ckpt_path)
            # print(f'Training finished:\nSeed {self.seed}, val loss {min_val_loss:.6f} at step {best_step}')
            self.logger.info(f'Training finished:\nSeed {self.seed}, val loss {min_val_loss:.6f} at step {best_step}')
        
            # save training curves
            plot_history(train_history, validation_history, self.args['num_steps'], ckpt_dir, self.seed)
        
        ## test ckpt
        if rank == 0:
            resume_ckpt_path = ckpt_path
            if torch.cuda.device_count() > 1:
                self.agent = self.agent.module
            self.agent.cpu()
            self.load_ckpt(resume_ckpt_path)

        return best_ckpt_info


def train_VLAIL(rank, world_size, args):
    if torch.cuda.device_count() > 1:
        setup(rank, world_size, args['ddp_port'])
    
    args['rank'] = rank
    args['world_size'] = world_size

    vla_method = VLAIL(args)
    vla_method.execute(args['task_name'])

def repeater(data_loader, total_steps, rank=0):
    step = 0
    while step < total_steps:
        # Create a new iterator for each epoch to ensure proper shuffling and distribution
        iterator = iter(data_loader)
        for data in iterator:
            yield data
            step += 1
            if step >= total_steps:
                break

        # Since the DataLoader is exhausted, synchronize all processes here
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Optionally, log the completion of an epoch on rank 0
        if rank == 0:
            print(f"Completed full pass through the DataLoader at step {step}")

def setup(rank, world_size, ddp_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = ddp_port
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(args):
    world_size = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        mp.spawn(train_VLAIL, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_VLAIL(0, 1, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--camera_names', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--agent_class', action='store', type=str, help='agent_class, capitalize', required=True)
    
    parser.add_argument('--batch_size_train', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--batch_size_val', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr_scheduler', action='store', type=str, help='lr_scheduler', required=True)

    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image', default=False, required=False)

    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--wandb_name', '--wandb_name', action='store', type=str, help='run name for logs', required=True)
    parser.add_argument('--use_wandb', action='store_true')
    parser.set_defaults(use_wandb=False)

    parser.add_argument('--use_ddp', action='store_true', default=False)
    parser.add_argument('--ddp_port', action='store', type=str, default='12355')

    parser.add_argument('--use_raw_lang', action='store_true')
    parser.set_defaults(use_raw_lang=False)
    
    # norm 1 for diffusion: ((x-min)/(max-min)) * 2 - 1
    # norm 2 for act: (x-mean)/std
    parser.add_argument('--act_norm_class', type=str, default='norm2')

    # exp type: franka_3rgb, franka_1rgb, ur_1rgb, songling_3rgb, tiangong_1rgb, simulation_4rgb
    parser.add_argument('--exp_type', type=str, default='franka_3rgb')

    # ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')

    # visual encoder
    # efficientnet_b0film, efficientnet_b3film, efficientnet_b5film
    # resnet18, resnet34, resnet50
    # resnet18film, resnet34film, resnet50film need to be debuged
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--no_sepe_backbone', action='store_true')
    parser.add_argument('--use_lang', action='store_true')
    parser.add_argument('--input_state_acthead', action='store_true')

    # for droid_diffusion, 
    parser.add_argument('--lr_backbone', default=1e-4, type=float, help='lr_backbone')
    parser.add_argument('--pool_class', type=str, default='null')
    parser.add_argument('--stsm_num_kp', type=int, default=512)
    parser.add_argument('--img_fea_dim', type=int, default=512)
    parser.add_argument('--cond_obs_dim', type=int, default=512)
    parser.add_argument('--num_noise_samples', type=int, default=8)

    # for droid_diffusion
    # new version is always false, have aug data in dataset getitem
    parser.add_argument('--use_color_rand', action='store_true')
    parser.set_defaults(use_color_rand=False)

    # ori diffusion, adt1, adt2 is True, ori act is False
    parser.add_argument('--use_data_aug', action='store_true')
    parser.set_defaults(use_data_aug=False)


    main(vars(parser.parse_args()))