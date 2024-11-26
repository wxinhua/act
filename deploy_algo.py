import warnings
import datetime
import os
import sys
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
import argparse
import yaml
from PIL import Image

# Get the current working directory
current_directory = os.getcwd()
# Append the current working directory to sys.path
sys.path.append(current_directory)

from agent.act import ACTPolicy
from agent.droid_difffusion import DroidDiffusionPolicy

# inference_thread = None
# inference_lock = threading.Lock()
# inference_actions = None
# inference_timestep = None
preparing = True

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

class InferVLAIL():
    def __init__(self, args=None):
        self.work_dir = Path.cwd()

        self.args = args
        ckpt_dir = self.args['ckpt_dir']
        ckpt_name = self.args['ckpt_name']
        if not os.path.isdir(self.args['ckpt_dir']):
            print(f"ckpt_dir {ckpt_dir} does not exist!!")
        
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.dataset_stats = pickle.load(f)

        cfg_path = os.path.join('./cfgs', f'{args["agent_class"]}', f'config_{args["exp_type"]}.yaml')
        with open(cfg_path, 'r', encoding='utf-8') as fin:
            self.config = yaml.load(fin, Loader=yaml.SafeLoader)
        self.config['robot_infor']['camera_names'] = self.args['camera_names']
        if self.args['use_depth_image']:
            self.config['robot_infor']['camera_sensors'] = ['rgb_images', 'depth_images']
        else:
            self.config['robot_infor']['camera_sensors'] = ['rgb_images']

        for key in self.config['agent_config']:
            if key in self.args:
                self.config['agent_config'][key] = self.args[key]

        self.args['config'] = self.config
        self.agent, agent_config = make_agent(self.args, self.config['agent_config'], 0)
        self.config['agent_config'] = agent_config
        self.chunk_size = self.args['chunk_size']

        load_ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.load_ckpt(load_ckpt_path)

        self.agent.cuda()

        self.exp_type = self.args['exp_type']
        if self.exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb', 'tiangong_1rgb']:
            self.qpos_arm_key = 'puppet'
            self.action_arm_key = 'puppet'
            self.ctl_elem_key = 'joint_position'
        elif self.exp_type in ['songling_3rgb']:
            self.qpos_arm_key = 'puppet'
            self.action_arm_key = 'master'
            self.ctl_elem_key = ['joint_position_left', 'joint_position_right']
        elif self.exp_type in ['simulation_4rgb']:
            self.qpos_arm_key = 'franka'
            self.action_arm_key = 'franka'
            self.ctl_elem_key = 'joint_position'
        
        if self.args['use_lang']:
            raw_lang = self.args['raw_lang']
            raw_lang = 'place the lid of the toaster on the table, grab the bread on the right and place it in the toaster.'
            encoded_input = tokenizer(raw_lang, return_tensors='pt').to('cuda')
            outputs = lang_model(**encoded_input)
            encoded_lang = outputs.last_hidden_state.sum(1).squeeze().unsqueeze(0)
            # [1, 768]
            print(f'encoded_lang size: {encoded_lang.size()}')
            self.encoded_lang = encoded_lang.float()
        else:
            self.encoded_lang = None

    def qpos_pre_process(self, robot_state_value, stats):
        tmp = robot_state_value
        tmp = (tmp - stats['qpos_mean']) / stats['qpos_std']
        return tmp

    def action_post_process(self, action, stats):
        act_norm_class = self.args['act_norm_class']
        print(f'act_norm_class: {act_norm_class}')
        if act_norm_class == 'norm1':
            action = ((action + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        elif act_norm_class == 'norm2':
            action = action * stats['action_std'] + stats['action_mean']
        
        return action

    def load_ckpt(self, load_ckpt_path=None):
        # if load_ckpt_path is not None:
        print(f"Load load_ckpt_path!")
        checkpoint = torch.load(load_ckpt_path, map_location=torch.device('cpu'))
        loading_status = self.agent.deserialize(checkpoint['nets'])
        curr_step = checkpoint["step"]
        print(f"Resume policy from: {load_ckpt_path}, Status: {loading_status}, Curr_step: {curr_step}")

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
        all_cam_images = []
        self.resize_images = True
        for cam_name in self.args['camera_names']:
            curr_image = obs['images'][cam_name]
            print(f'{cam_name} curr_image:',curr_image.shape)
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

            if self.exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb', 'simulation_4rgb']:
                curr_image = curr_image[:, :, ::-1]

            # img_dir = '/home/ps/wk/tmp'
            # if self.tmp_cnt < 10:
            #     img = Image.fromarray((curr_image).astype(np.uint8))
            #     img.save(os.path.join(img_dir, str(self.tmp_cnt)+'_rgb.png'))
            # self.tmp_cnt += 1

            # cv2.imshow(f"{cam_name} image", curr_image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # curr_image = rearrange(curr_image, 'h w c -> c h w')
            # curr_image: (3, 480, 640)
            # print('curr_image:',curr_image.shape)
            all_cam_images.append(curr_image)
        all_cam_images = np.stack(all_cam_images, axis=0)
        # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        all_cam_images = (all_cam_images / 255.0).astype(np.float32)

        return all_cam_images

    def get_qpos(self, obs):
        # incros/incros/robot_env/franka_env.py
        # incros/incros/robot_env/ur_env.py
        qpos = obs['qpos']
        return qpos

    def get_model_input(self, obs, rand_crop_resize):
        qpos = self.get_qpos(obs)
        input_qpos = self.qpos_pre_process(qpos, self.dataset_stats)
        
        self.tmp_cnt = 0
        input_image = self.get_image(obs)
        input_depth = None
        # input_qpos: (8,)
        # input_image: (3, 480, 640, 3)
        print(f"input_qpos: {input_qpos.shape}")
        print(f"input_image: {input_image.shape}")
        # print(f"input_depth: {input_depth.shape}")

        qpos_data = torch.from_numpy(input_qpos).float()
        image_data = torch.from_numpy(input_image)
        # k is the number of camera
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        
        qpos_data = qpos_data.unsqueeze(0)
        image_data = image_data.unsqueeze(0)
        qpos_data = qpos_data.cuda()
        image_data = image_data.cuda()
        # use_data_aug True in training
        if rand_crop_resize:
            print('rand crop resize is used!')
            original_size = image_data.shape[-2:]
            ratio = 0.95
            image_data = image_data[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                        int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
            image_data = image_data.squeeze(0)
            print(f"1. image_data size: {image_data.size()}")
            resize_transform = transforms.Resize(original_size, antialias=True)
            image_data = resize_transform(image_data)
            image_data = image_data.unsqueeze(0)
            print(f"2. image_data size: {image_data.size()}")

        return qpos_data, image_data, input_depth
        
    def execute(self):
        ###
        listener_thread = threading.Thread(target=self.start_keyboard_listener, daemon=True)
        listener_thread.start()
        ###
        print("Going to start position")

        self.print_color("\nReady for Start 🚀🚀🚀", color="green", attrs=("bold",))
        os.system("espeak start")

        if self.exp_type == 'franka_3rgb':
            sys.path.append('/home/ps/Dev/inrocs/')
            from robot_env.franka_env import robot_env
        elif self.exp_type == 'ur_1rgb':
            sys.path.append("/home/ps/work_sapce_hqr/inrocs")
            from robot_env.ur_env import robot_env

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

        max_timesteps = self.args['episode_len']

        temporal_agg = self.args['temporal_agg']
        chunk_size = self.args['chunk_size']
        state_dim = self.config['agent_config']['state_dim']
        action_dim = self.config['agent_config']['action_dim']
        rand_crop_resize = self.args['use_data_aug']

        if temporal_agg:
            # all_time_actions = np.zeros([max_timesteps, max_timesteps + chunk_size, action_dim])
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+chunk_size, action_dim]).cuda()
            query_frequency = 1
        else:
            query_frequency = chunk_size

        # 推理
        with torch.inference_mode():
            for t in range(max_timesteps):
                if t % query_frequency == 0:
                    qpos_data, image_data, input_depth = self.get_model_input(obs, rand_crop_resize)
                    
                if self.args['agent_class'] == 'ACT':
                    if t % query_frequency == 0:
                        all_actions = self.agent(qpos_data, image_data, input_depth, language_distilbert=self.encoded_lang)
                        # all_actions = all_actions.cpu().numpy()
                    
                    if temporal_agg:
                        infer_chunk = chunk_size
                        print(f"all_actions: {all_actions.size()}")
                        print(f"all_time_actions: {all_time_actions.size()}")
                        print(f"t: {t}, chunk_size:{chunk_size}")
                        # all_time_actions[[t], t:t+chunk_size] = all_actions
                        all_time_actions[[t], t:t+infer_chunk] = all_actions[:,:infer_chunk]
                        
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif self.args['agent_class'] == 'DroidDiffusion':                   
                    if t % query_frequency == 0:
                        all_actions = self.agent(qpos_data, image_data, input_depth, language_distilbert=self.encoded_lang)
                        print(f"======= t :{t}, all_actions size: {all_actions.size()}")
                    
                    temporal_agg = False
                    if temporal_agg:
                        print("temporal_agg:",temporal_agg)
                        infer_chunk = chunk_size #8 #num_queries #1 #5 #num_queries #10
                        print(f"all_actions: {all_actions.size()}")
                        print(f"all_time_actions: {all_time_actions.size()}")
                        print(f"t: {t}, num_queries:{num_queries}")
                        # all_time_actions[[t], t:t+num_queries] = all_actions
                        all_time_actions[[t], t:t+infer_chunk] = all_actions[:,:infer_chunk]
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                        print(f"t: {t}, raw_action: {raw_action.size()}")

                raw_action = raw_action.squeeze(0).cpu().numpy()
                action_pred = self.action_post_process(raw_action, self.dataset_stats)
                print(f"action_pred size: {action_pred.shape}")
                obs = robot_env.step(action_pred)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--camera_names', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', required=True)
    parser.add_argument('--agent_class', action='store', type=str, help='agent_class, capitalize', required=True)
    
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', default=0)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image', default=False, required=False)

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


    parser.add_argument('--episode_len', action='store', type=int, help='episode_len', default=5000, required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation', default=False, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step', default=0, required=False)

    parser.add_argument('--raw_lang', type=str, default='null')

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    args_input = vars(args)
    infer_model = InferVLAIL(args_input)

    infer_model.execute()

if __name__ == '__main__':
    main()
