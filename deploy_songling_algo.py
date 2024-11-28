
import torch
import numpy as np
import os
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
from pynput import keyboard
import threading
import torch

from pathlib import Path
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
print('CURRENT_DIR:',CURRENT_DIR)
sys.path.append(CURRENT_DIR)

from utils import set_seed_everywhere, compute_dict_mean, detach_dict, plot_history
# from agent_method import VLAIL
from ros_set import RosOperator
from agent.act import ACTPolicy
from agent.droid_difffusion import DroidDiffusionPolicy

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None
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
    def __init__(self, args):
        # self.infer_work_dir = Path.cwd()

        # self.infer_work_dir = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
        #
        # print(f"Infer workspace: {self.infer_work_dir}")
        self.work_dir = Path.cwd()
        self.args = args

        ros_set_path = os.path.join(CURRENT_DIR, 'ros_set_config.yaml')

        with open(ros_set_path, 'r', encoding='utf-8') as fin:
            self.ros_set_config = yaml.load(fin, Loader=yaml.SafeLoader)

        self.use_depth_image = args['use_depth_image']
        self.chunk_size = args['chunk_size']
        self.use_actions_interpolation = args['use_actions_interpolation']
        self.use_robot_base = args['use_robot_base']
        self.pos_lookahead_step = args['pos_lookahead_step']

        self.episode_len = args['max_publish_step']
        self.temporal_agg = args['temporal_agg']
        self.state_dim = args['state_dim']

        self.ros_set_config['use_depth_image'] = self.use_depth_image
        self.ros_set_config['use_robot_base'] = self.use_robot_base
        print('self.ros_set_config:',self.ros_set_config)

        self.ros_operator = RosOperator(argparse.Namespace(**self.ros_set_config))

        self.ckpt_dir = args['ckpt_dir']
        self.ckpt_name = args['ckpt_name']
        self.camera_names = args['camera_names']

        # save dataset stats
        stats_path = os.path.join(self.ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
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

        load_ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_name)
        self.load_ckpt(load_ckpt_path)

        self.agent.cuda()

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
        # tmp = (tmp - stats['qpos_mean']) / stats['qpos_std']
        tmp = (tmp - stats['action_mean']) / stats['action_std']
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

    def get_image(self, observation, show_img=False):
        # (w, h) for cv2.resize
        img_new_size = (640, 480) #(480, 640)
        curr_images = []
        self.resize_images = True
        for cam_name in self.args['camera_names']:
            curr_image = observation['images'][cam_name]
            rgb_image_encode = cv2.imencode(".jpg", curr_image)[1]
            curr_image = cv2.imdecode(rgb_image_encode, cv2.IMREAD_COLOR)

            print(f'{cam_name} curr_image:',curr_image.shape)

            if self.resize_images:
                curr_image = cv2.resize(curr_image, dsize=img_new_size)
            
            if show_img:
                rgb_image = curr_image[:, :, ::-1]
                cv2.imshow(f"{cam_name} image", rgb_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # curr_image = curr_image[:, :, ::-1]

            # curr_image = rearrange(curr_image, 'h w c -> c h w')
            curr_images.append(curr_image)
        all_cam_images = np.stack(curr_images, axis=0)
        # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        all_cam_images = (all_cam_images / 255.0).astype(np.float32)

        return all_cam_images

    def get_depth_image(self, observation, camera_names):
        curr_images = []
        for cam_name in camera_names:
            curr_images.append(observation['images_depth'][cam_name])
        curr_image = np.stack(curr_images, axis=0)
        # curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image

    def generate_obs(self, ros_results):
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth, puppet_arm_left, puppet_arm_right, robot_base) = ros_results
        
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
        
        return obs

    def inference_process(self, t, pre_action, rand_crop_resize):
        global inference_lock
        global inference_actions
        global inference_timestep
        print_flag = True
        # pre_pos_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        # pre_action_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]

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
            
            obs = self.generate_obs(result)
            #######

            print('obs[qpos]:',obs['qpos'])
            # qpos = pre_pos_process(obs['qpos'])
            qpos = self.qpos_pre_process(obs['qpos'], self.stats)

            # 当前图像curr_image获取图像
            curr_image = self.get_image(obs)
            curr_depth = None
            if self.use_depth_image:
                curr_depth = self.get_depth_image(obs, self.camera_names)

            start_time = time.time()

            qpos_data = torch.from_numpy(qpos).float()
            image_data = torch.from_numpy(curr_image)
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
            
            with torch.inference_mode():
                inference_actions = self.agent(qpos_data, image_data, curr_depth, language_distilbert=self.encoded_lang)

            # inference_actions = self.agent.eval_act(curr_image, curr_depth, qpos)
            end_time = time.time()
            print("model cost time: ", end_time - start_time)

            inference_lock.acquire()
            # inference_actions = all_actions.cpu().detach().numpy()
            if pre_action is None:
                pre_action = obs['qpos']
            # print("obs['qpos']:", obs['qpos'][7:])
            if self.use_actions_interpolation:
                inference_actions = self.actions_interpolation(pre_action, inference_actions)
            inference_timestep = t
            inference_actions = inference_actions.cpu().numpy()
            inference_lock.release()
            break

    def model_inference(self, save_episode=True):
        global inference_lock
        global inference_actions
        global inference_timestep
        global inference_thread
        set_seed_everywhere(1000)

        ###
        listener_thread = threading.Thread(target=self.start_keyboard_listener, daemon=True)
        listener_thread.start()
        ###
        print("Going to start position")

        self.print_color("\nReady for Start 🚀🚀🚀", color="green", attrs=("bold",))
        os.system("espeak start")

        # warm up
        import time
        time.sleep(2)
        obs = self.ros_operator.get_frame()
        # print('***obs***:', obs)


        max_publish_step = self.episode_len


        # pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        # post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']

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
        # input("Enter any key to continue :")
        self.ros_operator.puppet_arm_publish_continuous(left1, right1)
        action = None

        ###
        print("enter enter to go")
        global preparing
        rate = rospy.Rate(self.ros_set_config['publish_rate'])
        print_flag = True
        while preparing and not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                rate.sleep()
                continue
            # print(f"result: {result}")
            obs = self.generate_obs(result)
            input_image = self.get_image(obs, show_img=True)
        preparing = True
        ###
        for i in range(2):
            result = self.ros_operator.get_frame()

        action_dim = self.config['agent_config']['action_dim']
        rand_crop_resize = self.args['use_data_aug']

        # 推理
        with torch.inference_mode():
            while True and not rospy.is_shutdown():
                # 每个回合的步数
                t = 0
                max_t = 0
                rate = rospy.Rate(self.ros_set_config['publish_rate'])
                if self.temporal_agg:
                    all_time_actions = np.zeros([max_publish_step, max_publish_step + self.chunk_size, action_dim])
                while t < max_publish_step and not rospy.is_shutdown():
                    # start_time = time.time()
                    # query policy
                    if t >= max_t:
                        pre_action = action
                        inference_thread = threading.Thread(target=self.inference_process, args=(t, pre_action, rand_crop_resize))
                        inference_thread.start()
                        inference_thread.join()
                        inference_lock.acquire()
                        if inference_actions is not None:
                            infer_chunk = self.chunk_size
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + self.pos_lookahead_step
                            if self.temporal_agg:
                                all_time_actions[[t], t:t + infer_chunk] = all_actions[:,:infer_chunk]
                        inference_lock.release()
                    if self.temporal_agg:
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if self.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % self.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % self.chunk_size]

                    # raw_action = raw_action.squeeze(0).cpu().numpy()
                    raw_action = raw_action[0]
                    # action = post_process(raw_action[0])
                    action = self.action_post_process(raw_action, self.stats)
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
    parser.add_argument('--exp_type', type=str, default='songling_3rgb')

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

    # parser.add_argument('--episode_len', action='store', type=int, help='episode_len', default=10000, required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation', default=False, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step', default=0, required=False)

    parser.add_argument('--raw_lang', type=str, default='null')

    ###################
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)

    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    args_input = vars(args)
    infer_act = InferVLAIL(args_input)

    infer_act.model_inference(save_episode=True)


if __name__ == '__main__':
    main()




