import warnings
import datetime
import os
import sys
sys.path.append('/home/ps/Dev/inrocs/')
# sys.path.append('..')
sys.path.append('/home/ps/Dev/inrocs/action_frame')
import threading
import time
from pathlib import Path
import pickle
import numpy as np
import tqdm
import tyro
from pynput import keyboard
import torch
import cv2
from einops import rearrange

from robot_env.franka_env import robot_env

import argparse
import sys
import os
sys.path.append(os.getcwd())
from action_frame.utils import set_seed_everywhere, compute_dict_mean, detach_dict, plot_history
from action_frame.infer import VLAIL

# inference_thread = None
# inference_lock = threading.Lock()
# inference_actions = None
# inference_timestep = None
preparing = True

class InferACT():
    def __init__(self, args_input):

        self.ckpt_dir = args_input['ckpt_dir']
        self.ckpt_name = args_input['ckpt_name']
        self.camera_names = args_input['camera_names']
        self.chunk_size = args_input['chunk_size']

        self.episode_len = args_input['episode_len']
        self.temporal_agg = args_input['temporal_agg']
        self.state_dim = args_input['state_dim']

        self.pos_lookahead_step = args_input['pos_lookahead_step']

        stats_path = os.path.join(self.ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        # self.val_infer = VLAIL(ckpt_dir=self.ckpt_dir, ckpt_name=self.ckpt_name, chunk_size=self.chunk_size, camera_names=self.camera_names, stats=self.stats)
        self.val_infer = VLAIL(stats=self.stats, args=args_input)

        self.chunk_size = 15 # org25

        self.chunk_size = 18 # org50


        self.action_process_input = lambda s_qpos: (s_qpos - self.stats['action_mean']) / self.stats['action_std']
        self.action_process_output = lambda action: action * self.stats['action_std'] + self.stats['action_mean']

    def on_press(self, key):
        global preparing
        try:
            if key == keyboard.Key.enter:
                preparing = False

        except AttributeError:
            pass


    def start_keyboard_listener(self):
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()


    def print_color(self, *args, color=None, attrs=(), **kwargs):
        import termcolor

        if len(args) > 0:
            args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
        print(*args, **kwargs)


    def get_image(self, obs):
        # (w, h) for cv2.resize
        img_new_size = (640, 480) #(480, 640)
        curr_images = []
        self.resize_images = True
        for cam_name in self.camera_names:
            curr_image = obs['images'][cam_name]
            print('curr_image:',curr_image.shape)
            # rgb_image_encode = cv2.imencode(".jpg", curr_image)[1]
            rgb_image_encode = curr_image
            curr_image = cv2.imdecode(rgb_image_encode, cv2.IMREAD_COLOR)

            # if cam_name == 'top':
            if self.resize_images:
                # from 640 1280 -> 480 640
                # 1 curr_image: (720, 1280, 3)
                # 2 curr_image: (640, 480, 3)
                # print('1 curr_image:',curr_image.shape)
                curr_image = cv2.resize(curr_image, dsize=img_new_size)
                # print('2 curr_image:',curr_image.shape)

            # cv2.imshow("image", curr_image)
            # cv2.waitKey()

            curr_image = rearrange(curr_image, 'h w c -> c h w')
            # curr_image: (3, 480, 640)
            # print('curr_image:',curr_image.shape)
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image

    def get_qpos(self, obs):
        qpos = obs['qpos']
        return qpos

    def get_norm_stats(self, stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        return stats

    def inference_process(self, t, pre_action):
        global inference_lock
        global inference_actions
        global inference_timestep
        print_flag = True

        obs = robot_env.get_obs()

        qpos = self.get_qpos(obs)
        qpos_input = self.action_process_input(qpos)

        curr_image = self.get_image(obs)
        curr_depth = None

        start_time = time.time()
        inference_actions = self.val_infer.eval_act(curr_image, curr_depth, qpos_input)
        
        end_time = time.time()
        print("model cost time: ", end_time - start_time)

        inference_lock.acquire()
        # inference_actions = all_actions.cpu().detach().numpy()
        if pre_action is None:
            pre_action = obs['qpos']

        inference_timestep = t
        inference_lock.release()

    def model_inference(self, ):
        global inference_lock
        global inference_actions
        global inference_timestep
        global inference_thread
        set_seed_everywhere(1000)

        max_publish_step = self.episode_len

        print("Going to start position")

        print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))
        os.system("spd-say start")

        ###
        # warm up
        for i in range(25):
            obs = robot_env.get_obs()
        # print(data)
        t = 0
        max_t = 0

        action = None
        # 推理
        with torch.inference_mode():
            t = 0
            max_t = 0
            if self.temporal_agg:
                all_time_actions = np.zeros([max_publish_step, max_publish_step + self.chunk_size, self.state_dim])

            while t < max_publish_step:
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

                action_pred = self.action_process_output(raw_action[0])
                obs = robot_env.step(action_pred)

                t += 1

    ###############################################
    ###############################################
    ###############################################
    def infer_set(self, obs):
        qpos = self.get_qpos(obs)
        qpos_input = self.action_process_input(qpos)

        curr_image = self.get_image(obs)
        curr_depth = None

        start_time = time.time()
        inference_actions = self.val_infer.eval_act(curr_image, curr_depth, qpos_input)
        print('self.chunk_size:',self.chunk_size)
        print('inference_actions:',inference_actions.shape)
        inference_actions = inference_actions[:,:self.chunk_size,:]
        end_time = time.time()
        print("model cost time: ", end_time - start_time)
        return inference_actions

    def execute(self, ):
        ###
        listener_thread = threading.Thread(target=self.start_keyboard_listener, daemon=True)
        listener_thread.start()
        ###
        print("Going to start position")

        self.print_color("\nReady for Start 🚀🚀🚀", color="green", attrs=("bold",))
        os.system("espeak start")

        # warm up
        obs = robot_env.get_obs()
        print('***obs***:', obs)

        ###
        print("enter enter to go")
        global preparing
        while preparing:
            ...
        preparing = True
        ###
        for i in range(2):
            obs = robot_env.get_obs()

        max_publish_step = self.episode_len
        # self.chunk_size = 35
        # 推理
        with torch.inference_mode():
            t = 0
            max_t = 0
            if self.temporal_agg:
                all_time_actions = np.zeros([max_publish_step, max_publish_step + self.chunk_size, self.state_dim])

            # qpos_history = torch.zeros((1, max_publish_step, self.state_dim)).cuda()

            for t in range(max_publish_step):
                if t % self.chunk_size == 0:
                    all_actions = self.infer_set(obs)
                if self.temporal_agg:
                    print('all_action:',all_actions.shape)
                    all_time_actions[[t], t:t + self.chunk_size] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    print('actions_for_curr_step:',actions_for_curr_step.shape)
                    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = exp_weights[:, np.newaxis]
                    raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)


                    # k = 0.01
                    # exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    # exp_weights = exp_weights / exp_weights.sum()
                    # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    # raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % self.chunk_size]

                action_pred = self.action_process_output(raw_action[0])
                obs = robot_env.step(action_pred)

            # while t < max_publish_step:
            #     # query policy
            #     if t >= max_t:
            #         inference_actions = self.infer_set(obs)
            #         if inference_actions is not None:
            #             all_actions = inference_actions
            #             max_t = t + self.pos_lookahead_step
            #             if self.temporal_agg:
            #                 all_time_actions[[t], t:t + self.chunk_size] = all_actions
            #     if self.temporal_agg:
            #         actions_for_curr_step = all_time_actions[:, t]
            #         actions_populated = np.all(actions_for_curr_step != 0, axis=1)
            #         actions_for_curr_step = actions_for_curr_step[actions_populated]
            #         k = 0.01
            #         exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            #         exp_weights = exp_weights / exp_weights.sum()
            #         exp_weights = exp_weights[:, np.newaxis]
            #         raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
            #     else:
            #         if self.pos_lookahead_step != 0:
            #             raw_action = all_actions[:, t % self.pos_lookahead_step]
            #         else:
            #             raw_action = all_actions[:, t % self.chunk_size]
            #
            #     action_pred = self.action_process_output(raw_action[0])
            #     obs = robot_env.step(action_pred)
            #     t += 1


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', action='store', type=str, help='cfg_path', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', required=True)

    parser.add_argument('--agent_class', action='store', type=str, help='agent_class, capitalize', required=True)

    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)

    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    # parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)


    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=50, required=False)
    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image', default=False, required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--camera_names', nargs='+', help='<Required> Set flag', required=True)

    args = parser.parse_args()
    return args

def main():
    # ckpt_dir = "/home/pc/Project/zhen/inrocs_franka/action_frame/ckpt_dir/open_drawer_lr1e-5_batch24_chunk100"
    # ckpt_name = "agent_best.ckpt"
    # camera_names = ["left", "right"]

    args = get_arguments()
    args_input = vars(args)

    # chunk_size = 100

    episode_len = 10000
    # temporal_agg = True
    state_dim = 8
    pos_lookahead_step = 0

    args_input['episode_len'] = episode_len
    args_input['state_dim'] = state_dim
    args_input['pos_lookahead_step'] = pos_lookahead_step

    infer_act = InferACT(args_input)

    # infer_act.model_inference()
    infer_act.execute()

if __name__ == '__main__':
    main()
