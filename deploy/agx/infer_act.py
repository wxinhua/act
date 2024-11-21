

import torch
import numpy as np
import pickle
import argparse
import yaml
from einops import rearrange
import rospy
import time
import threading
import math
import collections
import cv2

import sys
import os
sys.path.append(os.getcwd())
from utils import set_seed_everywhere, compute_dict_mean, detach_dict, plot_history
from infer import VLAIL

from pathlib import Path
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
print('CURRENT_DIR:',CURRENT_DIR)
sys.path.append(CURRENT_DIR)

from ros_set import RosOperator

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

class InferACT():
    def __init__(self, args):
        # self.infer_work_dir = Path.cwd()

        # self.infer_work_dir = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
        #
        # print(f"Infer workspace: {self.infer_work_dir}")
        ros_set_path = os.path.join(CURRENT_DIR, 'ros_set_config.yaml')

        with open(ros_set_path, 'r', encoding='utf-8') as fin:
            self.ros_set_config = yaml.load(fin, Loader=yaml.SafeLoader)

        self.use_depth_image = args.use_depth_image
        self.chunk_size = args.chunk_size
        self.use_actions_interpolation = args.use_actions_interpolation
        self.use_robot_base = args.use_robot_base
        self.pos_lookahead_step = args.pos_lookahead_step

        self.episode_len = args.max_publish_step
        self.temporal_agg = args.temporal_agg
        self.state_dim = args.state_dim

        self.ros_set_config['use_depth_image'] = self.use_depth_image
        self.ros_set_config['use_robot_base'] = self.use_robot_base
        print('self.ros_set_config:',self.ros_set_config)

        self.ros_operator = RosOperator(argparse.Namespace(**self.ros_set_config))

        self.ckpt_dir = args.ckpt_dir
        self.ckpt_name = args.ckpt_name
        self.camera_names = args.camera_names

        # save dataset stats
        stats_path = os.path.join(self.ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        self.val_infer = VLAIL(ckpt_dir=self.ckpt_dir, ckpt_name=self.ckpt_name,
                               chunk_size=self.chunk_size, camera_names=self.camera_names,
                               stats=self.stats)

    def actions_interpolation(self, pre_action, actions):
        steps = np.concatenate((np.array(self.ros_set_config['arm_steps_length']),
                                np.array(self.ros_set_config['arm_steps_length'])), axis=0)
        pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        result = [pre_action]
        post_action = post_process(actions[0])
        # print("pre_action:", pre_action[7:])
        # print("actions_interpolation1:", post_action[:, 7:])
        max_diff_index = 0
        max_diff = -1
        for i in range(post_action.shape[0]):
            diff = 0
            for j in range(pre_action.shape[0]):
                if j == 6 or j == 13:
                    continue
                diff += math.fabs(pre_action[j] - post_action[i][j])
            if diff > max_diff:
                max_diff = diff
                max_diff_index = i

        for i in range(max_diff_index, post_action.shape[0]):
            step = max([math.floor(math.fabs(result[-1][j] - post_action[i][j]) / steps[j]) for j in
                        range(pre_action.shape[0])])
            inter = np.linspace(result[-1], post_action[i], step + 2)
            result.extend(inter[1:])
        while len(result) < self.chunk_size + 1:
            result.append(result[-1])
        result = np.array(result)[1:self.chunk_size + 1]
        # print("actions_interpolation2:", result.shape, result[:, 7:])
        result = pre_process(result)
        result = result[np.newaxis, :]
        return result

    def get_image(self, observation, camera_names):
        curr_images = []
        for cam_name in camera_names:
            curr_image = observation['images'][cam_name]
            rgb_image_encode = cv2.imencode(".jpg", curr_image)[1]
            curr_image = cv2.imdecode(rgb_image_encode, cv2.IMREAD_COLOR)
            curr_image = rearrange(curr_image, 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image

    def get_depth_image(self, observation, camera_names):
        curr_images = []
        for cam_name in camera_names:
            curr_images.append(observation['images_depth'][cam_name])
        curr_image = np.stack(curr_images, axis=0)
        # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image

    def inference_process(self, t, pre_action):
        global inference_lock
        global inference_actions
        global inference_timestep
        print_flag = True
        pre_pos_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        pre_action_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]

        rate = rospy.Rate(self.ros_set_config['publish_rate'])
        while True and not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                rate.sleep()
                continue
            print_flag = True
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
             puppet_arm_left, puppet_arm_right, robot_base) = result
            obs = collections.OrderedDict()
            image_dict = dict()

            image_dict[self.camera_names[0]] = img_front
            image_dict[self.camera_names[1]] = img_left
            image_dict[self.camera_names[2]] = img_right

            obs['images'] = image_dict

            if self.use_depth_image:
                image_depth_dict = dict()
                image_depth_dict[self.camera_names[0]] = img_front_depth
                image_depth_dict[self.camera_names[1]] = img_left_depth
                image_depth_dict[self.camera_names[2]] = img_right_depth
                obs['images_depth'] = image_depth_dict

            obs['qpos'] = np.concatenate(
                (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            obs['qvel'] = np.concatenate(
                (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
            obs['effort'] = np.concatenate(
                (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
            if self.use_robot_base:
                obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
                obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
            else:
                obs['base_vel'] = [0.0, 0.0]

            print('obs[qpos]:',obs['qpos'])
            qpos = pre_pos_process(obs['qpos'])

            # 当前图像curr_image获取图像
            curr_image = self.get_image(obs, self.camera_names)
            curr_depth = None
            if self.use_depth_image:
                curr_depth = self.get_depth_image(obs, self.camera_names)

            start_time = time.time()
            inference_actions = self.val_infer.eval_act(curr_image, curr_depth, qpos)
            end_time = time.time()
            print("model cost time: ", end_time - start_time)

            inference_lock.acquire()
            # inference_actions = all_actions.cpu().detach().numpy()
            if pre_action is None:
                pre_action = obs['qpos']
            # print("obs['qpos']:", obs['qpos'][7:])
            if self.use_actions_interpolation:
                inference_actions = actions_interpolation(pre_action, inference_actions)
            inference_timestep = t
            inference_lock.release()
            break

    def model_inference(self, save_episode=True):
        global inference_lock
        global inference_actions
        global inference_timestep
        global inference_thread
        set_seed_everywhere(1000)

        max_publish_step = self.episode_len

        pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']

        # 发布基础的姿态
        left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                 -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
        right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656,
                  -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
        left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                 -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
        right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156,
                  -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]

        self.ros_operator.puppet_arm_publish_continuous(left0, right0)
        input("Enter any key to continue :")
        self.ros_operator.puppet_arm_publish_continuous(left1, right1)
        action = None
        # 推理
        with torch.inference_mode():
            while True and not rospy.is_shutdown():
                # 每个回合的步数
                t = 0
                max_t = 0
                rate = rospy.Rate(self.ros_set_config['publish_rate'])
                if self.temporal_agg:
                    all_time_actions = np.zeros([max_publish_step, max_publish_step + self.chunk_size, self.state_dim])
                while t < max_publish_step and not rospy.is_shutdown():
                    # start_time = time.time()
                    # query policy
                    if t >= max_t:
                        pre_action = action
                        inference_thread = threading.Thread(target=self.inference_process,
                                                            args=(t, pre_action))
                        inference_thread.start()
                        inference_thread.join()
                        inference_lock.acquire()
                        if inference_actions is not None:
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + self.pos_lookahead_step
                            if self.temporal_agg:
                                all_time_actions[[t], t:t + self.chunk_size] = all_actions
                        inference_lock.release()
                    if self.temporal_agg:
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if self.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % self.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % self.chunk_size]

                    action = post_process(raw_action[0])
                    print('action:',action)
                    left_action = action[:7]  # 取7维度
                    right_action = action[7:14]
                    self.ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread
                    if self.use_robot_base:
                        vel_action = action[14:16]
                        self.ros_operator.robot_base_publish(vel_action)
                    t += 1

                    rate.sleep()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', required=True)

    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)
    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)

    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=50, required=False)
    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--camera_names', nargs='+', help='<Required> Set flag', required=True)

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    infer_act = InferACT(args)

    infer_act.model_inference(save_episode=True)


if __name__ == '__main__':
    main()




