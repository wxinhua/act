## Copyright (c) Meta Platforms, Inc. and affiliates
from memory_profiler import profile

import numpy as np
import torch
import os
import h5py
from torch.utils.data import DataLoader
from collections import defaultdict
import random
import glob
import gc
from dataset_load.read_franka_h5 import ReadH5Files
import time
import random
import copy
import torchvision.transforms as transforms
import cv2

class DatasetAloha(torch.utils.data.Dataset):
    def __init__(self, episode_ids, files, norm_stats, num_episodes, chunk_size, robot_infor, trial_lengths, cutoff, use_depth_image):
        super(DatasetAloha).__init__()
        self.episode_ids = episode_ids
        # self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.num_episodes = num_episodes
        self.is_sim = None
        self.verbose = True
        self.use_depth_image = use_depth_image
        self.min_depth = 0.25
        self.max_depth = 1.25
        self.cutoff = cutoff
        self.robot_infor = robot_infor
        self.chunk_size = chunk_size

        # files = get_files(self.dataset_dir)
        self.read_h5files = ReadH5Files(self.robot_infor)

        self.max_idx = trial_lengths
        self.trial_files = files
        print("TOTAL TRIALS", len(self.trial_files))
        print("CURRENT TRIALS", self.num_episodes)
        print("CURRENT idx", len(self.episode_ids))

        self.augment_images = False
        self.resize_images = True
        print('Initializing transformations')
        # (h, w)
        # original_size = self.image_data_shape
        # self.new_size = (480, 640)
        # (w, h) for cv2.resize
        self.new_size = (640, 480)
        ratio = 0.95
        self.resize_trans = transforms.Resize(self.new_size, antialias=True) # original_size
        self.aug_trans = [
            # transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
            # transforms.Resize(new_size, antialias=True), # original_size
            # transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
        ]

        self.__getitem__(0)
        
    def __len__(self):
        return len(self.episode_ids)

    # @profile
    def __getitem__(self, idx):
        # print('self.episode_ids:',self.episode_ids)
        trial_idx, start_ts = self.episode_ids[idx]
        trial_file = self.trial_files[trial_idx]

        image_dict, control_dict, base_dict, is_sim, is_compress = self.read_h5files.execute(
            trial_file, camera_frame=start_ts, use_depth_image=self.use_depth_image)
        # print('--train-load-one-h5file-time--:', time.time() - time3)
        puppet_joint_position = control_dict['puppet']['joint_position'][()]
        action = puppet_joint_position
        # print(f"=============")
        # print(f"0.0 action shape: {action.shape}")

        if len(base_dict) > 0 and self.robot_infor['use_robot_base']:
            base_action = base_dict['base_action'][:-self.cutoff]
            base_action = preprocess_base_action(base_action)
            action = np.concatenate([action, base_action], axis=-1)
        # else:
        #     dummy_base_action = np.zeros([action.shape[0], 2])  # (n, 2)
        #     action = np.concatenate([action, dummy_base_action], axis=-1)  # (n, 18)

        original_action_shape = action.shape
        # print(f"0. original_action_shape: {original_action_shape}")
        episode_len = original_action_shape[0] - self.cutoff

        # get observation at start_ts only
        qpos = action[start_ts]

        if is_sim:
            action = action[start_ts:]
            action_len = episode_len - start_ts
        else:
            action = action[max(0, start_ts - 1):]  # hack, to make timesteps more aligned
            action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned
        # print(f"0. action size: {action.shape}")

        if original_action_shape[0] < self.chunk_size:
            original_action_shape = list(original_action_shape)
            original_action_shape[0] = self.chunk_size
            original_action_shape = tuple(original_action_shape)
        
        if episode_len < self.chunk_size:
            episode_len = self.chunk_size

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        if self.cutoff > 0:
            padded_action[:action_len] = action[:-self.cutoff]
        else:
            padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        padded_action = padded_action[:self.chunk_size]
        # print(f"0. padded_action size: {padded_action.shape}")

        is_pad = is_pad[:self.chunk_size]
        # new axis for different cameras
        all_cam_images = []
        all_cam_depths = []
        # print('image_dict:',image_dict.keys())
        # for cam_name in self.camera_names:
        for cam_name in self.robot_infor['camera_names']:
            cur_img = image_dict[self.robot_infor['camera_sensors'][0]][cam_name]
            # cur_img = torch.from_numpy(cur_img)
            # print(f'1 cur_img size: {cur_img.size()}')
            # cur_img = torch.einsum('h w c -> c h w', cur_img)
            # print(f'2 cur_img size: {cur_img.size()}')
            # 1 cur_img size: (480, 640, 3) ([720, 1280, 3])
            # 2 cur_img size: (3, 480, 640) ([3, 720, 1280])
            if self.resize_images:
                # cur_img = self.resize_trans(cur_img)
                # 3 cur_img size: torch.Size([3, 480, 640])
                # print(f'3 cur_img size: {cur_img.size()}')
                cur_img = cv2.resize(cur_img, dsize=self.new_size)
                # 3 cur_img size: (640, 480, 3)
                # print(f'3 cur_img size: {cur_img.shape}')

            all_cam_images.append(cur_img) # rgb
            if self.use_depth_image:
                depth_image = image_dict[self.robot_infor['camera_sensors'][1]][cam_name] / 1000 # m
                depth_image_filtered = np.zeros_like(depth_image)
                depth_image_filtered[(depth_image >= self.min_depth) & (depth_image <= self.max_depth)] = depth_image[
                    (depth_image >= self.min_depth) & (depth_image <= self.max_depth)]
                # depth_image_filtered = np.expand_dims(depth_image_filtered, axis=2)

                # depth_image_filtered = torch.from_numpy(depth_image_filtered)
                if self.resize_images:
                    # depth_image_filtered = self.resize_trans(depth_image_filtered)
                    # print(f'3 depth_image_filtered size: {depth_image_filtered.size()}')
                    depth_image_filtered = cv2.resize(depth_image_filtered, dsize=self.new_size)
                    # print(f'3 depth_image_filtered size: {depth_image_filtered.shape}')

                all_cam_depths.append(depth_image_filtered)
            else:
                # all_cam_depths.append(np.zeros_like(image_dict[self.robot_infor['camera_sensors'][0]][cam_name]))
                # dummy_depth = torch.zeros(self.new_size)
                # 3 dummy_depth size: torch.Size([480, 640])
                # print(f'3 dummy_depth size: {dummy_depth.size()}')
                # dummy_depth = np.zeros(self.new_size)
                dummy_depth = np.zeros((480, 640))
                # 3 dummy_depth size: (480, 640)
                # print(f'3 dummy_depth size: {dummy_depth.shape}')

                all_cam_depths.append(dummy_depth)
        all_cam_images = np.stack(all_cam_images, axis=0)
        all_cam_depths = np.stack(all_cam_depths, axis=0)

        # all_cam_images = torch.stack(all_cam_images, axis=0)
        # all_cam_depths = torch.stack(all_cam_depths, axis=0)

        # all_cam_depths = np.zeros_like(all_cam_images)
        # construct observations
        all_cam_images = (all_cam_images / 255.0).astype(np.float32)
        # all_cam_images = (all_cam_images / 255.0).to(dtype=torch.float32)
        image_data = torch.from_numpy(all_cam_images)
        # image_data = all_cam_images

        all_cam_depths = all_cam_depths.astype(np.float32)
        # all_cam_depths = all_cam_depths.to(dtype=torch.float32)
        depth_data = torch.from_numpy(all_cam_depths)
        # depth_data = all_cam_depths
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        # self.image_data_shape = image_data.shape[2:]
        # image_data size: torch.Size([2, 3, 480, 640])
        # depth_data size: torch.Size([2, 480, 640])
        # print(f"image_data size: {image_data.size()}")
        # print(f"depth_data size: {depth_data.size()}")
        # depth_data = torch.einsum('k h w c -> k c h w', depth_data)

        if self.augment_images:
            for transform in self.aug_trans:
                image_data = transform(image_data)
        
        # print(f"image_data size: {image_data.size()}")

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # del image_dict
        # del control_dict
        # del base_dict
        # gc.collect()

        # print(f"=============+++++++++++")
        # print(f"image_data size: {image_data.size()}")
        # print(f"depth_data size: {depth_data.size()}")
        # print(f"qpos_data size: {qpos_data.size()}")
        # print(f"action_data size: {action_data.size()}")
        # print(f"is_pad size: {is_pad.size()}")

        return image_data, depth_data, qpos_data, action_data, is_pad


def get_files(dataset_dir, robot_infor):
    read_h5files = ReadH5Files(robot_infor)
    dataset_dir = os.path.join(dataset_dir, 'success_episodes')
    files = []
    for trajectory_id in sorted(os.listdir(dataset_dir)):
        trajectory_dir = os.path.join(dataset_dir, trajectory_id)
        file_path = os.path.join(trajectory_dir, 'data/trajectory.hdf5')
        try:
            _, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(file_path, camera_frame=2)
            files.append(file_path)
        except Exception as e:
            print(e)
    return files


def get_norm_stats(files, robot_infor, cutoff, train_ratio=0.9):
    read_h5files = ReadH5Files(robot_infor)

    all_action_data = []
    all_episode_len = []

    trial_indices = []

    for id, filename in enumerate(files):
        # Check each file to see how many entires it has
        # time1 = time.time()
        try:
            _, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(filename, camera_frame=0)

            if cutoff > 0:
                puppet_joint_position = control_dict['puppet']['joint_position'][:-cutoff]
                action = puppet_joint_position

                if len(base_dict) > 0 and robot_infor['use_robot_base']:
                    base_action = base_dict['base_action'][:-cutoff]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([action, base_action], axis=-1)
                else:
                    action = action[:-cutoff]
                    # dummy_base_action = np.zeros([action.shape[0], 2])  # (n, 2)
                    # action = np.concatenate([action, dummy_base_action], axis=-1)  # (n, 18)
            else:
                puppet_joint_position = control_dict['puppet']['joint_position'][:]
                action = puppet_joint_position

                if len(base_dict) > 0 and robot_infor['use_robot_base']:
                    base_action = base_dict['base_action'][:]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([action, base_action], axis=-1)
                else:
                    action = action[:]
                    # dummy_base_action = np.zeros([action.shape[0], 2])  # (n, 2)
                    # action = np.concatenate([action, dummy_base_action], axis=-1)  # (n, 18)

            for frame_id in range(action.shape[0]):
                trial_indices.append((id, frame_id))
            all_action_data.append(torch.from_numpy(action))
            all_episode_len.append(len(action))
        except Exception as e:
            print(e)
            print('filename:',filename)

    num_episodes = len(all_action_data)
    random.shuffle(trial_indices)
    print('trial_indices:',len(trial_indices))
    train_trial_indices = trial_indices[:int(train_ratio * len(trial_indices))]
    val_trial_indices = trial_indices[int(train_ratio * len(trial_indices)):]

    train_num_episodes = int(num_episodes * train_ratio)
    val_num_episodes = num_episodes - train_num_episodes

    all_action_data = torch.cat(all_action_data, dim=0)  # 该任务所有的轨迹帧按行合并[每条轨迹长度*轨迹数，18]

    trial_lengths = sum(all_episode_len)
    print('trial_lengths:',trial_lengths)
    assert (len(train_trial_indices) + len(val_trial_indices)) == trial_lengths

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps, "action_max": action_max.numpy() + eps,
             }

    # del all_qpos_data, all_action_data, all_episode_len
    # gc.collect()

    return (stats, trial_lengths, train_trial_indices, val_trial_indices,
            train_num_episodes, val_num_episodes)


def load_data(dataset_dir, robot_infor, batch_size_train, batch_size_val, chunk_size, cutoff=0, use_depth_image=False):
    # obtain train test split
    train_ratio = 0.99  # change as needed
    files = get_files(dataset_dir, robot_infor)
    # obtain normalization stats for qpos and action
    (norm_stats, trial_lengths, train_trial_indices, val_trial_indices,
     train_num_episodes, val_num_episodes) = get_norm_stats(files, robot_infor, cutoff, train_ratio)
    print('----get-norm-stats-finished----')
    # construct dataset and dataloader
    train_dataset = DatasetAloha(train_trial_indices, files, norm_stats, train_num_episodes, chunk_size, robot_infor, trial_lengths, cutoff, use_depth_image)
    val_dataset = DatasetAloha(val_trial_indices, files, norm_stats, val_num_episodes, chunk_size, robot_infor, trial_lengths, cutoff, use_depth_image)

    # num_workers=18
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=2, prefetch_factor=1) # num_workers=8
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=18, prefetch_factor=1) # num_workers=8

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    # print('base_action1:',base_action)
    base_action = smooth_base_action(base_action)
    return base_action
