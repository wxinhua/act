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
from dataset_load.read_aloha_h5 import ReadH5Files
import time

class DatasetAloha(torch.utils.data.Dataset):
    def __init__(self, episode_ids, files, norm_stats, num_episodes, chunk_size, robot_infor,
                 trial_lengths, cutoff):
        super(DatasetAloha).__init__()
        self.episode_ids = episode_ids
        # self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.num_episodes = num_episodes
        self.is_sim = None
        self.verbose = True
        self.cutoff = cutoff
        self.robot_infor = robot_infor
        self.chunk_size = chunk_size

        # files = get_files(self.dataset_dir)
        self.read_h5files = ReadH5Files(self.robot_infor)

        self.max_idx = trial_lengths
        self.trial_files = files
        print("TOTAL TRIALS", len(self.trial_files))
        self.trial_files = self.trial_files[:num_episodes]

        assert self.num_episodes == len(self.trial_files)  ## sanity check that all files are loaded, remove if needed

        print('TOTAL TRIALS = num_episodes = ', len(self.trial_files))

        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    # @profile
    def __getitem__(self, idx):
        # print('idx:',idx)
        trial_idx, start_ts = self.episode_ids[idx]
        # print('trial_idx:',trial_idx)
        # print('start_ts:',start_ts)

        trial_file = self.trial_files[trial_idx]
        image_dict, control_dict, base_dict, is_sim, is_compress = self.read_h5files.execute(trial_file,
                                                                                             camera_frame=start_ts)
        # print('--train-load-one-h5file-time--:', time.time() - time3)
        master_joint_position_left = control_dict['master']['joint_position_left'][()]
        master_joint_position_right = control_dict['master']['joint_position_right'][()]
        action = np.concatenate([master_joint_position_left, master_joint_position_right], axis=-1)  # (n, 18)

        if len(base_dict) > 0 and self.robot_infor['use_robot_base']:
            base_action = base_dict['base_action'][:-self.cutoff]
            base_action = preprocess_base_action(base_action)
            action = np.concatenate([action, base_action], axis=-1)
        # else:
        #     dummy_base_action = np.zeros([action.shape[0], 2])  # (n, 2)
        #     action = np.concatenate([action, dummy_base_action], axis=-1)  # (n, 18)

        original_action_shape = action.shape
        episode_len = original_action_shape[0] - self.cutoff

        # get observation at start_ts only
        puppet_joint_position_left_ts = control_dict['puppet']['joint_position_left'][start_ts]
        puppet_joint_position_right_ts = control_dict['puppet']['joint_position_right'][start_ts]
        qpos = np.concatenate([puppet_joint_position_left_ts, puppet_joint_position_right_ts], axis=-1)
        if is_sim:
            action = action[start_ts:]
            action_len = episode_len - start_ts
        else:
            action = action[max(0, start_ts - 1):]  # hack, to make timesteps more aligned
            action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        if self.cutoff > 0:
            padded_action[:action_len] = action[:-self.cutoff]
        else:
            padded_action[:action_len] = action

        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        padded_action = padded_action[:self.chunk_size]
        is_pad = is_pad[:self.chunk_size]

        # new axis for different cameras
        all_cam_images = []
        all_cam_depths = []

        # print('image_dict:',image_dict.keys())
        # for cam_name in self.camera_names:
        for cam_name in self.robot_infor['camera_names']:
            all_cam_images.append(image_dict[self.robot_infor['camera_sensors'][0]][cam_name]) # rgb
            all_cam_depths.append([image_dict[self.robot_infor['camera_sensors'][1]][cam_name]])
        all_cam_images = np.stack(all_cam_images, axis=0)
        all_cam_depths = np.stack(all_cam_depths, axis=0)

        # construct observations
        all_cam_images = (all_cam_images / 255.0).astype(np.float32)
        image_data = torch.from_numpy(all_cam_images)

        all_cam_depths = all_cam_depths.astype(np.float32)
        depth_data = torch.from_numpy(all_cam_depths)
        # image_data = torch.from_numpy(all_cam_images/255).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        depth_data = torch.einsum('k h w c -> k c h w', depth_data)

        # normalize image and change dtype to float
        # image_data = image_data / 255.0

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # del image_dict
        # del control_dict
        # del base_dict
        # gc.collect()
        return image_data, depth_data, qpos_data, action_data, is_pad


def get_files(dataset_dir):
    # dataset_dir_path = glob.glob("{}".format(dataset_dir))[0]
    dataset_dir = os.path.join(dataset_dir, 'success_episodes')
    files = []
    for trajectory_id in sorted(os.listdir(dataset_dir)):
        trajectory_dir = os.path.join(dataset_dir, trajectory_id)
        file_path = os.path.join(trajectory_dir, 'data/trajectory.hdf5')
        print('file_path:',file_path)
        files.append(file_path)
    return files


def get_norm_stats(files, num_episodes, robot_infor, cutoff, train_ratio=0.9):
    read_h5files = ReadH5Files(robot_infor)

    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    train_trial_indices = []
    val_trial_indices = []

    for id, filename in enumerate(files):
        # Check each file to see how many entires it has
        # time1 = time.time()
        _, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(filename, camera_frame=0)
        # time2 = time.time()
        # print('read_h5file_time = ', time2 - time1)
        if cutoff > 0:
            puppet_joint_position_left = control_dict['puppet']['joint_position_left'][:-cutoff]
            puppet_joint_position_right = control_dict['puppet']['joint_position_right'][:-cutoff]
            qpos = np.concatenate([puppet_joint_position_left, puppet_joint_position_right], axis=-1)

            master_joint_position_left = control_dict['master']['joint_position_left'][:-cutoff]
            master_joint_position_right = control_dict['master']['joint_position_right'][:-cutoff]
            action = np.concatenate([master_joint_position_left, master_joint_position_right], axis=-1)  # (n, 18)

            if len(base_dict) > 0 and robot_infor['use_robot_base']:
                base_action = base_dict['base_action'][:-cutoff]
                base_action = preprocess_base_action(base_action)
                action = np.concatenate([action, base_action], axis=-1)
            else:
                action = action[:-cutoff]
                # dummy_base_action = np.zeros([action.shape[0], 2])  # (n, 2)
                # action = np.concatenate([action, dummy_base_action], axis=-1)  # (n, 18)
        else:
            puppet_joint_position_left = control_dict['puppet']['joint_position_left'][:]
            puppet_joint_position_right = control_dict['puppet']['joint_position_right'][:]
            qpos = np.concatenate([puppet_joint_position_left, puppet_joint_position_right], axis=-1)

            master_joint_position_left = control_dict['master']['joint_position_left'][:]
            master_joint_position_right = control_dict['master']['joint_position_right'][:]
            action = np.concatenate([master_joint_position_left, master_joint_position_right], axis=-1)  # (n, 18)

            if len(base_dict) > 0 and robot_infor['use_robot_base']:
                base_action = base_dict['base_action'][:]
                base_action = preprocess_base_action(base_action)
                action = np.concatenate([action, base_action], axis=-1)
            else:
                action = action[:]
                # dummy_base_action = np.zeros([action.shape[0], 2])  # (n, 2)
                # action = np.concatenate([action, dummy_base_action], axis=-1)  # (n, 18)

        for frame_id in range(qpos.shape[0]):
            if id in train_indices:
                train_trial_indices.append((id, frame_id))
            elif id in val_indices:
                val_trial_indices.append((id, frame_id))
            else:
                continue
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)  # 该任务所有的轨迹帧按行合并[每条轨迹长度*轨迹数，16]
    all_action_data = torch.cat(all_action_data, dim=0)  # 该任务所有的轨迹帧按行合并[每条轨迹长度*轨迹数，18]

    trial_lengths = sum(all_episode_len)

    assert (len(train_trial_indices) + len(val_trial_indices)) == trial_lengths

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps, "action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    # del all_qpos_data, all_action_data, all_episode_len
    # gc.collect()

    return stats, trial_lengths, train_trial_indices, val_trial_indices


def load_data(dataset_dir, robot_infor, batch_size_train, batch_size_val, chunk_size, cutoff=0):
    # obtain train test split
    train_ratio = 0.99  # change as needed
    files = get_files(dataset_dir)
    num_episodes = len(files)
    # obtain normalization stats for qpos and action
    norm_stats, trial_lengths, train_trial_indices, val_trial_indices = get_norm_stats(files,
                                                                                       num_episodes,
                                                                                       robot_infor,
                                                                                       cutoff, train_ratio)
    print('----get-norm-stats-finished----')
    # construct dataset and dataloader
    train_dataset = DatasetAloha(train_trial_indices, files, norm_stats, num_episodes, chunk_size,
                                 robot_infor, trial_lengths, cutoff)
    val_dataset = DatasetAloha(val_trial_indices, files, norm_stats, num_episodes, chunk_size,
                               robot_infor, trial_lengths, cutoff)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False,
                                  num_workers=8, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=8,
                                prefetch_factor=1)

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
