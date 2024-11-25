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
from torch.utils.data import DataLoader
from dataset_load.read_franka_h5 import ReadH5Files
import h5py

# torchvision.disable_beta_transforms_warning()

class Logger:
    def set_log_file_handler(self):
        self.logger = logging.getLogger(__name__)

        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)

        log_dir = './ckpt_dir/ACT_camlrt/241121/table/pick_plate_from_plate_rack_lr1e5_batch24_chunk50'

        now_str = datetime.datetime.now().__str__().replace(' ','_')

        self.log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)


def print_memory_info(process, idx):
    used_bytes = process.memory_info().rss
    used_MB = used_bytes / (1024*1024)
    print(f"{idx}. used_MB: {used_MB}")


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, robot_infor, norm_stats, episode_ids, episode_len, chunk_size, rank=None, use_data_aug=False, act_norm_class='norm2', use_raw_lang=False, use_depth_image=False, exp_type='franka_3rgb'):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.use_depth_image = use_depth_image
        self.min_depth = 0.25
        self.max_depth = 1.25
        self.robot_infor = robot_infor
        self.exp_type = exp_type

        if exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb', 'tiangong_1rgb']:
            self.qpos_arm_key = 'puppet'
            self.action_arm_key = 'puppet'
            self.ctl_elem_key = 'joint_position'
        elif exp_type in ['songling_3rgb']:
            self.qpos_arm_key = 'puppet'
            self.action_arm_key = 'master'
            self.ctl_elem_key = ['joint_position_left', 'joint_position_right']
        elif exp_type in ['simulation_4rgb']:
            self.qpos_arm_key = 'franka'
            self.action_arm_key = 'franka'
            self.ctl_elem_key = 'joint_position'

        # todo
        # self.path2string_dict = dict()
        # for dataset_path in self.dataset_path_list:
        #     with h5py.File(dataset_path, 'r') as root:
        #         if 'language_raw' in root.keys():
        #             lang_raw_utf = root['language_raw'][0].decode('utf-8')
        #         else:
        #             lang_raw_utf = 'None'
        #         self.path2string_dict[dataset_path] = lang_raw_utf
        #     root.close()
        
        print(f"len episode_ids: {len(self.episode_ids)}")
        print(f"first 10 episode_ids: {self.episode_ids[:10]}")
        print(f"len episode_len: {len(self.episode_len)}")
        print(f"sum episode_len: {sum(self.episode_len)}")
        print(f"len dataset_path_list: {len(self.dataset_path_list)}")
        # print(f"total dataset cumulative_len: {self.cumulative_len}")

        self.max_episode_len = max(episode_len)
        self.resize_images = True
        print('Initializing resize transformations')

        # (w, h) for cv2.resize
        self.new_size = (640, 480)

        self.augment_images = use_data_aug
        self.aug_trans = None
        print(f"cur dataset augment_images: {self.augment_images}")
        # augmentation
        if self.augment_images is True:
            print('Initializing transformations')
            # (h, w) for transforms aug
            original_size = (480, 640)
            ratio = 0.95
            self.aug_trans = [
                transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                transforms.Resize(original_size, antialias=True),
                transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
            ]

        self.rank = rank
        self.transformations = None
        self.act_norm_class = act_norm_class
        self.use_raw_lang = use_raw_lang

        self.tmp_cnt = 0

        ####
        # self.read_cnt = 0
        # self.image_dict = None
        # self.control_dict = None
        # self.base_dict = None
        # self.exe = None 
        # self.is_compress = None
        # self.start_ts = None

        self.__getitem__(0) # initialize self.is_sim and self.transformations
    
    def __len__(self):
        return 1000 * sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        # print(f"index: {index}")
        # print(f"self.cumulative_len: {self.cumulative_len}")
        # print(f"episode_id: {episode_id}")
        # print(f"start_ts: {start_ts}")
        return episode_id, start_ts

    def __getitem__(self, index):
        index = index % sum(self.episode_len)  # Ensure index wraps around
        episode_id, start_ts = self._locate_transition(index)

        dataset_path = self.dataset_path_list[episode_id]
        
        # print(f"=============")
        # print(f"xxxxxxxxxxxxxx")
        process = psutil.Process(os.getpid())
        # print_memory_info(process, 2)
        
        # ### todo

        with h5py.File(dataset_path, 'r') as root:

            image_dict = dict()
            for cam_name in self.robot_infor['camera_names']:
                image_dict[cam_name] = root['observations']['rgb_images'][cam_name][start_ts]

                cur_image = cv2.imdecode(image_dict[cam_name], cv2.IMREAD_COLOR)
                image_dict[cam_name] = cur_image
 
        root.close()
        
        # print_memory_info(process, 3)

        all_cam_images = []
        all_cam_depths = []
        for cam_name in self.robot_infor['camera_names']:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images1 = np.stack(all_cam_images, axis=0)
        # all_cam_images = torch.stack(all_cam_images, axis=0)
        ## right 
        all_cam_images1 = (all_cam_images1 / 255.0).astype(np.float32)

        # construct observations
        image_data = torch.from_numpy(all_cam_images1)

        # print_memory_info(process, 4)
        # memory leakage
        # image_data1 = image_data / 255.0

        print_memory_info(process, 5)

        return image_data


def get_files(dataset_dir, robot_infor):
    read_h5files = ReadH5Files(robot_infor)
    # dataset_dir = os.path.join(dataset_dir, 'success_episodes')
    files = []
    for trajectory_id in sorted(os.listdir(dataset_dir)):
        trajectory_dir = os.path.join(dataset_dir, trajectory_id)
        file_path = os.path.join(trajectory_dir, 'data/trajectory.hdf5')
        try:
            _, control_dict, base_dict, _, is_compress = read_h5files.execute(file_path, camera_frame=2)
            files.append(file_path)
        except Exception as e:
            print(e)
    return files

def get_norm_stats(train_dataset_path_list, val_dataset_path_list, robot_infor, exp_type):
    read_h5files = ReadH5Files(robot_infor)

    all_qpos_data = []
    all_action_data = []
    val_episode_len = []
    train_episode_len = []

    if exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb', 'tiangong_1rgb']:
        qpos_arm_key = 'puppet'
        action_arm_key = 'puppet'
        ctl_elem_key = 'joint_position'
    elif exp_type in ['songling_3rgb']:
        qpos_arm_key = 'puppet'
        action_arm_key = 'master'
        ctl_elem_key = ['joint_position_left', 'joint_position_right']
    elif exp_type in ['simulation_4rgb']:
        qpos_arm_key = 'franka'
        action_arm_key = 'franka'
        ctl_elem_key = 'joint_position'
    

    for list_id, cur_dataset_path_list in enumerate([train_dataset_path_list, val_dataset_path_list]):
        cur_episode_len = []
        for id, dataset_path in enumerate(cur_dataset_path_list):
            try:
                _, control_dict, base_dict, _, is_compress = read_h5files.execute(dataset_path, camera_frame=0)

                if isinstance(ctl_elem_key, list):
                    qpos_list = []
                    action_list = []
                    for ele in ctl_elem_key:
                        cur_qpos = control_dict[qpos_arm_key][ele][:]
                        cur_action = control_dict[action_arm_key][ele][:]
                        qpos_list.append(cur_qpos)
                        action_list.append(cur_action)
                    qpos = np.concatenate(qpos_list, axis=-1)
                    action = np.concatenate(action_list, axis=-1)
                else:
                    qpos = control_dict[qpos_arm_key][ctl_elem_key][:]
                    action = control_dict[action_arm_key][ctl_elem_key][:]

                # puppet_joint_position = control_dict['puppet']['joint_position'][:]
                
                # action = puppet_joint_position

                # for frame_id in range(action.shape[0]):
                #     trial_indices.append((id, frame_id))
                all_qpos_data.append(torch.from_numpy(qpos))
                all_action_data.append(torch.from_numpy(action))
                cur_episode_len.append(len(action))
            except Exception as e:
                print(e)
                print('filename:',dataset_path)
        if list_id == 0:
            train_episode_len = cur_episode_len
        else:
            val_episode_len = cur_episode_len

    num_episodes = len(all_action_data)

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)  # 该任务所有的轨迹帧按行合并[每条轨迹长度*轨迹数，x]

    # normalize action data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    qpos_min = all_qpos_data.min(dim=0).values.float()
    qpos_max = all_qpos_data.max(dim=0).values.float()

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(), "action_min": action_min.numpy() - eps, "action_max": action_max.numpy() + eps, "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(), "qpos_min": qpos_min.numpy() - eps, "qpos_max": qpos_max.numpy() + eps}

    return stats, train_episode_len, val_episode_len

def flatten_list(l):
    return [item for sublist in l for item in sublist]


def load_data(dataset_dir_l, robot_infor, batch_size_train, batch_size_val, chunk_size, use_depth_image=False, sample_weights=None, rank=None, use_data_aug=False, act_norm_class='norm2', use_raw_lang=False, name_filter=None, exp_type='franka_3rgb', logger=None):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]

    # [[task1_epi1, task1_epi2], [task2_epi1, task2_epi2]]
    train_dataset_path_list_list = []
    val_dataset_path_list_list = []
    for dataset_dir in dataset_dir_l:
        # hdf5_files_list = find_all_hdf5(dataset_dir) 
        # obtain train test split
        succ_dataset_dir = os.path.join(dataset_dir, 'success_episodes')
        train_dir = os.path.join(succ_dataset_dir, 'train')
        val_dir = os.path.join(succ_dataset_dir, 'val')
        hdf5_train_files_list = get_files(train_dir, robot_infor)
        hdf5_val_files_list = get_files(val_dir, robot_infor)
        train_dataset_path_list_list.append(hdf5_train_files_list)
        val_dataset_path_list_list.append(hdf5_val_files_list)

        # hdf5_files_list = get_files(dataset_dir, robot_infor)
        # dataset_path_list_list.append(hdf5_files_list)
    
    train_num_episodes_l = [len(cur_dataset_path_list) for cur_dataset_path_list in train_dataset_path_list_list]
    train_num_episodes_cumsum = np.cumsum(train_num_episodes_l)
    val_num_episodes_l = [len(cur_dataset_path_list) for cur_dataset_path_list in val_dataset_path_list_list]
    val_num_episodes_cumsum = np.cumsum(val_num_episodes_l)
    if rank == 0 or rank is None:
        # print(f"train_num_episodes_cumsum: {train_num_episodes_cumsum}")
        # print(f"val_num_episodes_cumsum: {val_num_episodes_cumsum}")
        logger.info(f"train_num_episodes_cumsum: {train_num_episodes_cumsum}")
        logger.info(f"val_num_episodes_cumsum: {val_num_episodes_cumsum}")

    train_dataset_path_fla_list = flatten_list(train_dataset_path_list_list)
    train_dataset_path_fla_list = [n for n in train_dataset_path_fla_list if name_filter(n)]
    val_dataset_path_fla_list = flatten_list(val_dataset_path_list_list)
    val_dataset_path_fla_list = [n for n in val_dataset_path_fla_list if name_filter(n)]

    train_episode_ids_l = []
    cur_num_episodes = 0
    for task_idx, _ in enumerate(train_dataset_path_list_list):
        # num episodes of task idx
        num_episodes_i = len(train_dataset_path_list_list[task_idx])
        if rank == 0 or rank is None:
            # print(f"Train: cur task_idx: {task_idx}; num_episodes_i: {num_episodes_i}")
            logger.info(f"Train: cur task_idx: {task_idx}; num_episodes_i: {num_episodes_i}")
        shuffled_episode_ids_i = np.random.permutation(num_episodes_i)
        train_episode_ids_l.append(shuffled_episode_ids_i)
        for i in range(num_episodes_i):
            shuffled_episode_ids_i[i] += cur_num_episodes
        cur_num_episodes += num_episodes_i

    val_episode_ids_l = []
    cur_num_episodes = 0
    for task_idx, _ in enumerate(val_dataset_path_list_list):
        # num episodes of task idx
        num_episodes_i = len(val_dataset_path_list_list[task_idx])
        if rank == 0 or rank is None:
            # print(f"Val: cur task_idx: {task_idx}; num_episodes_i: {num_episodes_i}")
            logger.info(f"Val: cur task_idx: {task_idx}; num_episodes_i: {num_episodes_i}")
        shuffled_episode_ids_i = np.random.permutation(num_episodes_i)
        val_episode_ids_l.append(shuffled_episode_ids_i)
        for i in range(num_episodes_i):
            shuffled_episode_ids_i[i] += cur_num_episodes
        cur_num_episodes += num_episodes_i
    
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    if rank == 0 or rank is None:
        # train_episode_ids: list []
        # val_episode_ids: list []
        # print(f"train_episode_ids: {train_episode_ids}")
        # print(f"val_episode_ids: {val_episode_ids}")
        # print(f"len train_episode_ids: {len(train_episode_ids)}")
        # print(f"len val_episode_ids: {len(val_episode_ids)}")
        logger.info(f"train_episode_ids: {train_episode_ids}")
        logger.info(f"val_episode_ids: {val_episode_ids}")
        logger.info(f"len train_episode_ids: {len(train_episode_ids)}")
        logger.info(f"len val_episode_ids: {len(val_episode_ids)}")


    if rank == 0 or rank is None:
        # print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
        logger.info(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
    
    norm_stats, train_episode_len, val_episode_len = get_norm_stats(train_dataset_path_fla_list, val_dataset_path_fla_list, robot_infor, exp_type)
    if rank == 0 or rank is None:
        # print(f"norm_stats: {norm_stats}")
        # all_episode_len: list []
        # print(f"len all_episode_len: {len(all_episode_len)}")
        # print(f"len train_episode_len: {len(train_episode_len)}")
        # print(f"len val_episode_len: {len(val_episode_len)}")
        logger.info(f"norm_stats: {norm_stats}")
        logger.info(f"len train_episode_len: {len(train_episode_len)}")
        logger.info(f"len val_episode_len: {len(val_episode_len)}")

    train_episode_len_l = []
    for cur_task_train_episode_ids in train_episode_ids_l:
        cur_task_train_episode_len_l = []
        for i in cur_task_train_episode_ids:
            cur_task_train_episode_len_l.append(train_episode_len[i])
        train_episode_len_l.append(cur_task_train_episode_len_l)
    # if rank == 0 or rank is None:
    #     # train_episode_len_l: list of list [[task1_ep1, task1_ep2], [task2_ep1, task2_ep2]]
    #     print(f"train_episode_len_l: {train_episode_len_l}")
    
    val_episode_len_l = []
    for cur_task_val_episode_ids in val_episode_ids_l:
        cur_task_val_episode_len_l = []
        for i in cur_task_val_episode_ids:
            cur_task_val_episode_len_l.append(val_episode_len[i])
        val_episode_len_l.append(cur_task_val_episode_len_l)
    # if rank == 0 or rank is None:
    #     # val_episode_len_l: list of list [[task1_ep1, task1_ep2], [task2_ep1, task2_ep2]]
    #     print(f"val_episode_len_l: {val_episode_len_l}")

    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if rank == 0 or rank is None:
        # train_episode_len: list []
        # val_episode_len: list []
        # print(f"train_episode_len: {train_episode_len}")
        # print(f"val_episode_len: {val_episode_len}")
        # print(f'Norm stats from: {dataset_dir_l}')
        logger.info(f'Norm stats from: {dataset_dir_l}')

    if torch.cuda.device_count() == 1:
        # batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
        # batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

        # construct dataset and dataloader
        train_dataset = EpisodicDataset(train_dataset_path_fla_list, robot_infor, norm_stats, train_episode_ids, train_episode_len, chunk_size, rank=rank, use_data_aug=use_data_aug, act_norm_class=act_norm_class, use_raw_lang=use_raw_lang, use_depth_image=use_depth_image, exp_type=exp_type)
        # val_dataset = EpisodicDataset(val_dataset_path_fla_list, robot_infor, norm_stats, val_episode_ids, val_episode_len, chunk_size, rank=rank, use_data_aug=use_data_aug, act_norm_class=act_norm_class, use_raw_lang=use_raw_lang, use_depth_image=use_depth_image, exp_type=exp_type)

        # train_num_workers = 0
        train_num_workers = 0 #4 #16
        # val_num_workers = 0
        val_num_workers = 0 #4 #16
        # print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
        logger.info(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
        # pin_memory=True
        # train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=False, num_workers=train_num_workers, prefetch_factor=2)
        # val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=False, num_workers=val_num_workers, prefetch_factor=2)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=1, prefetch_factor=1) # num_workers=8
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=18, prefetch_factor=1) # num_workers=8
        
    return train_dataloader, None, norm_stats

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

if __name__ == "__main__":
    # from dataset_load.dataset_multi_robot import load_data
    batch_size_train = 24 #24
    batch_size_val = 1 #24
    robot_data_dir = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_franka_3rgb'
    # ['camera_left', 'camera_right'],
    robot_infor = {
        'camera_sensors': ['rgb_images'],
        'camera_names': ['camera_left'],
        'arms': ['puppet'],
        'controls': ['joint_position', 'end_effector'],
        'use_robot_base': False,
    }
    dataset_dir = os.path.join(robot_data_dir, 'pick_plate_from_plate_rack')

    name_filter = lambda n: True
    sample_weights = None

    logger = Logger()
    logger.set_log_file_handler()

    train_dataloader, val_dataloader, stats = load_data(dataset_dir, robot_infor, batch_size_train, batch_size_val, chunk_size=50, use_depth_image=False, sample_weights=sample_weights, rank=0, use_data_aug=False, act_norm_class='norm2', use_raw_lang=False, name_filter=name_filter, exp_type='franka_3rgb', logger=logger.logger)

    # import time
    # train_dataloader = repeater(train_dataloader, 100000, 0)
    # for i in range(100000):
    #     print(f"=======================")
    #     process = psutil.Process(os.getpid())
    #     print_memory_info(process, i)

    #     data = next(train_dataloader)

        # time.sleep(1)
        
    for batch_idx, data in enumerate(train_dataloader):
        process = psutil.Process(os.getpid())
        print_memory_info(process, 100+batch_idx)



