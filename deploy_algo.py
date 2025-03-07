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

from dataset_load.read_franka_h5 import ReadH5Files
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
        if self.args['exp_type'] == 'tiangong_1rgb':
            # default is mode1
            if self.args['tg_mode'] == 'mode1':
                self.config['agent_config']['state_dim'] = 26
                self.config['agent_config']['action_dim'] = 26
            elif self.args['tg_mode'] == 'mode2':
                self.config['agent_config']['state_dim'] = 26
                self.config['agent_config']['action_dim'] = 18
            elif self.args['tg_mode'] == 'mode3':
                self.config['agent_config']['state_dim'] = 18
                self.config['agent_config']['action_dim'] = 18
            elif self.args['tg_mode'] == 'mode4':
                self.config['agent_config']['state_dim'] = 14
                self.config['agent_config']['action_dim'] = 14
            elif self.args['tg_mode'] in ['mode5', 'mode6', 'mode7', 'mode8']:
                self.config['robot_infor']['arms'] = ['puppet', 'master']
                self.config['robot_infor']['controls'] = ['joint_position']
                self.config['agent_config']['state_dim'] = 16
                self.config['agent_config']['action_dim'] = 16

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
        # if self.exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb', 'tiangong_1rgb']:
        #     self.qpos_arm_key = 'puppet'
        #     self.action_arm_key = 'puppet'
        #     self.ctl_elem_key = 'joint_position'
        # elif self.exp_type in ['songling_3rgb']:
        #     self.qpos_arm_key = 'puppet'
        #     self.action_arm_key = 'master'
        #     self.ctl_elem_key = ['joint_position_left', 'joint_position_right']
        # elif self.exp_type in ['simulation_4rgb']:
        #     self.qpos_arm_key = 'franka'
        #     self.action_arm_key = 'franka'
        #     self.ctl_elem_key = 'joint_position'
        
        
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

    def get_image(self, obs, show_img=False):
        # (w, h) for cv2.resize
        img_new_size = (640, 480) #(480, 640)
        all_cam_images = []
        self.resize_images = True
        for cam_name in self.args['camera_names']:
            if self.exp_type == 'tiangong_1rgb':
                curr_image = obs['images'][cam_name]
            else:
                # for franka_3rgb, ur_1rgb
                curr_image = obs['images'][cam_name]
            print(f'{cam_name} curr_image:',curr_image.shape)
            # rgb_image_encode = cv2.imencode(".jpg", curr_image)[1]
            rgb_image_encode = curr_image
            if self.exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb', 'simulation_4rgb', 'tiangong_1rgb']:
                curr_image = cv2.imdecode(rgb_image_encode, cv2.IMREAD_COLOR)
            else:
                curr_image = rgb_image_encode

            
            # if cam_name == 'top':
            if self.resize_images:
                # from 640 1280 -> 480 640
                # 1 curr_image: (720, 1280, 3)
                # 2 curr_image: (640, 480, 3)
                # print('1 curr_image:',curr_image.shape)
                curr_image = cv2.resize(curr_image, dsize=img_new_size)
                # print('2 curr_image:',curr_image.shape)

            if show_img:
                rgb_image = curr_image[:, :, ::-1]
                cv2.imshow(f"{cam_name} image", rgb_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.exp_type in ['franka_3rgb', 'franka_1rgb', 'ur_1rgb', 'simulation_4rgb']:
                curr_image = curr_image[:, :, ::-1]

            # img_dir = '/home/ps/wk/tmp'
            # if self.tmp_cnt < 10:
            #     img = Image.fromarray((curr_image).astype(np.uint8))
            #     img.save(os.path.join(img_dir, str(self.tmp_cnt)+'_rgb.png'))
            # self.tmp_cnt += 1            

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
        if self.exp_type == 'tiangong_1rgb':
            # [self.left_jpos, self.left_hand, self.right_jpos, self.right_hand]
            qpos = obs['qpos']
            left_jpos = qpos[:7]
            left_hand_jpos = qpos[7:13]
            right_jpos = qpos[13:20]
            right_hand_jpos = qpos[20:26]
            # mode1: input 26, output 26;
            # mode2: input 26, output 14+4=18: Index Finger食指, Thumb拇指
            # mode3: input 14+4=18, output 14+4=18: Index Finger食指, Thumb拇指
            # mode4: input 14, output 14: only contron arm
            if self.args['tg_mode'] == 'mode1':
                qpos = np.concatenate((left_hand_jpos, right_hand_jpos, left_jpos, right_jpos))
            elif self.args['tg_mode'] == 'mode2':
                qpos = np.concatenate((left_hand_jpos, right_hand_jpos, left_jpos, right_jpos))
            elif self.args['tg_mode'] == 'mode3':
                qpos = np.concatenate((left_hand_jpos[[3,4]], right_hand_jpos[3:4], left_jpos, right_jpos))
            elif self.args['tg_mode'] == 'mode4':
                qpos = np.concatenate((left_jpos, right_jpos))
            elif self.args['tg_mode'] in ['mode5', 'mode6', 'mode7', 'mode8']:
                ### 16 dim joint_position
                qpos = obs['qpos']
        else:
            # for franka_3rgb, ur_1rgb
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
        # tiangong mode4: input_qpos: (14,)
        # tiangong mode4: input_image: (1, 480, 640, 3)
        # print(f"input_qpos: {input_qpos.shape}")
        # print(f"input_image: {input_image.shape}")
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
        
    def process_tiangong_action(self, all_actions):
        # mode1: input 26, output 26;
        # mode2: input 26, output 14+4=18: Index Finger食指, Thumb拇指
        # mode3: input 14+4=18, output 14+4=18: Index Finger食指, Thumb拇指
        # mode4: input 14, output 14: only contron arm
        # action should be: [self.left_jpos, self.left_hand, self.right_jpos, self.right_hand]
        # all_actions: [b, t, dim]
        if self.args['tg_mode'] == 'mode1':
            left_hand_jpos = all_actions[:, :, :6]
            right_hand_jpos = all_actions[:, :, 6:12]
            left_jpos = all_actions[:, :, 12:19]
            right_jpos = all_actions[:, :, 19:26]
            all_actions = np.concatenate((left_jpos, left_hand_jpos, right_jpos, right_hand_jpos), axis=-1)
        elif self.args['tg_mode'] in ['mode2', 'mode3']:
            left_finger_jpos = all_actions[:, :, :2]
            right_finger_jpos = all_actions[:, :, 2:4]
            left_jpos = all_actions[:, :, 4:11]
            right_jpos = all_actions[:, :, 11:18]
            left_hand_jpos = np.array([0., 0., 0., 0., 0., 1.,])
            right_hand_jpos = np.array([0., 0., 0., 0., 0., 1.,])
            for i in range(4):
                left_hand_jpos[i] = left_finger_jpos[0]
                right_hand_jpos[i] = right_finger_jpos[0]
            left_hand_jpos[4] = left_finger_jpos[1]
            right_hand_jpos[4] = right_finger_jpos[1]
            left_hand_jpos = np.expand_dims(left_hand_jpos, axis=0)
            right_hand_jpos = np.expand_dims(right_hand_jpos, axis=0)
            all_actions = np.concatenate((left_jpos, left_hand_jpos, right_jpos, right_hand_jpos), axis=-1)
        elif self.args['tg_mode'] == 'mode4':
            left_hand_jpos = np.array([0., 0., 0., 0., 0., 1.,])
            right_hand_jpos = np.array([0., 0., 0., 0., 0., 1.,])
            left_jpos = all_actions[:, :, :7]
            right_jpos = all_actions[:, :, 7:14]
            left_hand_jpos = np.expand_dims(left_hand_jpos, axis=0)
            right_hand_jpos = np.expand_dims(right_hand_jpos, axis=0)
            all_actions = np.concatenate((left_jpos, left_hand_jpos, right_jpos, right_hand_jpos), axis=-1)
        
        return all_actions

    def init_tiangong(self, robot_env, h5_file):

        robot_infor = {'camera_names': ['camera_top'],
                        'camera_sensors': ['rgb_images'],
                        'arms': ['master', 'puppet'],
                        'controls': ['joint_position', 'end_effector']}

        read_h5files = ReadH5Files(robot_infor)

        image_dict, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(h5_file)
        end_effect = control_dict['puppet']['end_effector']
        joint_position = control_dict['puppet']['joint_position']

        left_hand_jpos = end_effect[0, :6]
        right_hand_jpos = end_effect[0, 6:12]
        left_arm_jpos = joint_position[0, :7]
        right_arm_jpos = joint_position[0, 7:14]

        prepare_left = [-0.176873, 0.103522, -1.334014, -0.1800, 1.2640, -0.01137, 0.2419, 1.0]
        prepare_right = [0.55696, -0.043007, 1.26522, 0.94075, -0.58208, -0.10169, 0.11724, 1.0]
        prepare_left[:7] = left_arm_jpos
        prepare_right[:7] = right_arm_jpos
        robot_env.move_to_target(prepare_left + prepare_right)

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
        elif self.exp_type == 'tiangong_1rgb':
            # sys.path.append('/home/ps/code_lei/inrocs')
            # # os.environ["ROS_MASTER_URI"] = "http://192.168.41.1:11311"
            # os.environ["ROS_MASTER_URI"] = "http://192.168.41.13:11311"
            # os.environ["ROS_IP"] = "192.168.41.55"
            # from robot_env.tiangong_env_5hz_wk import  TiangongEnv
            # robot_env = TiangongEnv(
            #     camera_topic="/camera/color/image_raw",
            #     left_arm_ctrl_topic="/human_arm_ctrl_left",
            #     right_arm_ctrl_topic="/human_arm_ctrl_right",
            #     left_arm_state_topic="/human_arm_state_left",
            #     right_arm_state_topic="/human_arm_state_right",
            #     left_hand_topic="/inspire_hand/ctrl/left_hand",
            #     right_hand_topic="/inspire_hand/ctrl/right_hand"
            # )
            from inrocs.robot_env.tianyi_env import tianyi_env as robot_env
            # qpos = tianyi_env.get_obs_full()['qpos']

            # print(qpos)

            # qpos[7] = 1.0
            # tianyi_env.step_full(qpos)

            # # tianyi_env.reset_to_home()
            if self.args['tg_mode'] in ['mode1', 'mode2', 'mode3', 'mode4']:
                robot_env.reset_to_parepre()
            elif self.args['tg_mode'] in ['mode5', 'mode6', 'mode7', 'mode8']:
                robot_env.reset_to_parepre_left()

            # traj_list = ['/home/ps/wk/benchmark_results/tiangong_1122_traj.hdf5',
            # '/home/ps/wk/benchmark_results/tiangong_1122_traj_2.hdf5', 
            # '/home/ps/wk/benchmark_results/tiangong_place_button_traj.hdf5' ]

            # h5_file = '/home/ps/wk/benchmark_results/tiangong_place_button_traj.hdf5'
            # h5_file = '/home/ps/wk/benchmark_results/tiangong_place_button_traj.hdf5'
            # self.init_tiangong(robot_env, h5_file)

        # warm up
        import time
        time.sleep(2)
        if self.exp_type == 'tiangong_1rgb':
            if self.args['tg_mode'] in ['mode1', 'mode2', 'mode3', 'mode4']:
                obs = robot_env.get_obs_full()
            else:
                obs = robot_env.get_obs()
        else:
            obs = robot_env.get_obs()
        print('***obs***:', obs)

        ###
        print("enter enter to go")
        global preparing
        while preparing:
            # ...
            # obs = robot_env.get_obs()
            if self.exp_type == 'tiangong_1rgb':
                if self.args['tg_mode'] in ['mode1', 'mode2', 'mode3', 'mode4']:
                    obs = robot_env.get_obs_full()
                else:
                    obs = robot_env.get_obs()
            else:
                obs = robot_env.get_obs()
            input_image = self.get_image(obs, show_img=True)
            # print('***obs***:', obs)

        preparing = True
        ###
        for i in range(2):
            if self.exp_type == 'tiangong_1rgb':
                if self.args['tg_mode'] in ['mode1', 'mode2', 'mode3', 'mode4']:
                    obs = robot_env.get_obs_full()
                else:
                    obs = robot_env.get_obs()
            else:
                obs = robot_env.get_obs()

        max_timesteps = self.args['episode_len']

        temporal_agg = self.args['temporal_agg']
        chunk_size = self.args['chunk_size']
        state_dim = self.config['agent_config']['state_dim']
        action_dim = self.config['agent_config']['action_dim']
        rand_crop_resize = self.args['use_data_aug']

        if temporal_agg:
            # all_time_actions = np.zeros([max_timesteps, max_timesteps + chunk_size, action_dim])
            all_time_actions = np.zeros([max_timesteps, max_timesteps+chunk_size, action_dim])
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
                        all_actions = all_actions.cpu().numpy()
                        # print(f"1 all_actions: {all_actions.shape}")

                        if self.exp_type == 'tiangong_1rgb':
                            all_actions = self.process_tiangong_action(all_actions)
                    
                    if temporal_agg:
                        infer_chunk = chunk_size
                        # print(f"all_actions: {all_actions.shape}")
                        # print(f"all_time_actions: {all_time_actions.shape}")
                        print(f"t: {t}, chunk_size:{chunk_size}")
                        # all_time_actions[[t], t:t+chunk_size] = all_actions
                        all_time_actions[[t], t:t+infer_chunk] = all_actions[:,:infer_chunk]
                        
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif self.args['agent_class'] == 'DroidDiffusion':                   
                    if t % query_frequency == 0:
                        all_actions = self.agent(qpos_data, image_data, input_depth, language_distilbert=self.encoded_lang)
                        print(f"======= t :{t}, all_actions size: {all_actions.size()}")
                        all_actions = all_actions.cpu().numpy()

                        if self.exp_type == 'tiangong_1rgb':
                            all_actions = self.process_tiangong_action(all_actions)
                    
                    temporal_agg = False
                    if temporal_agg:
                        print("temporal_agg:",temporal_agg)
                        infer_chunk = chunk_size #8 #num_queries #1 #5 #num_queries #10
                        print(f"all_actions: {all_actions.shape}")
                        print(f"all_time_actions: {all_time_actions.shape}")
                        print(f"t: {t}, num_queries:{num_queries}")
                        # all_time_actions[[t], t:t+num_queries] = all_actions
                        all_time_actions[[t], t:t+infer_chunk] = all_actions[:,:infer_chunk]
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                        print(f"t: {t}, raw_action: {raw_action.shape}")

                # raw_action = raw_action.squeeze(0).cpu().numpy()
                raw_action = raw_action[0]
                action_pred = self.action_post_process(raw_action, self.dataset_stats)
                print(f"action_pred size: {action_pred.shape}")
                if self.exp_type == 'tiangong_1rgb':
                    # robot_env.act(action_pred)
                    if self.args['tg_mode'] in ['mode1', 'mode2', 'mode3']:
                        robot_env.step_full(action_pred)
                        obs = robot_env.get_obs_full()
                        time.sleep(0.2)
                    if self.args['tg_mode'] in ['mode4']:
                        prepare_left = [-0.176873, 0.103522, -1.334014, -0.1800, 1.2640, -0.01137, 0.2419, 1.0]
                        prepare_right = [0.55696, -0.043007, 1.26522, 0.94075, -0.58208, -0.10169, 0.11724, 1.0]
                        left_arm_jpos = action_pred[12:19]
                        right_arm_jpos = action_pred[19:26]
                        prepare_left[:7] = left_arm_jpos
                        prepare_right[:7] = right_arm_jpos
                        robot_env.move_to_target(prepare_left + prepare_right)
                        obs = robot_env.get_obs_full()
                        time.sleep(0.1)
                    elif self.args['tg_mode'] in ['mode5', 'mode6', 'mode7', 'mode8']:
                        ######## 16 dim
                        robot_env.step(action_pred)
                        # time.sleep(0.1)
                        obs = robot_env.get_obs()
                else:
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


    parser.add_argument('--episode_len', action='store', type=int, help='episode_len', default=10000, required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation', default=False, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step', default=0, required=False)

    parser.add_argument('--raw_lang', type=str, default='null')

    # mode1: input 26, output 26; 
    # mode2: input 26, output 14+2=16: Index Finger食指, Thumb拇指->average
    # mode3: input 14+2=16, output 14+2=16: Index Finger食指, Thumb拇指->average
    # mode4: input 14, output 14: only contron arm
    parser.add_argument('--tg_mode', type=str, default='mode1')

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    args_input = vars(args)
    infer_model = InferVLAIL(args_input)

    infer_model.execute()

if __name__ == '__main__':
    main()
