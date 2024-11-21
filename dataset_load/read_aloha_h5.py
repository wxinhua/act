import h5py
import os
import cv2
import numpy as np
from collections import defaultdict
from memory_profiler import profile
import gc
import psutil


class ReadH5Files():
    def __init__(self, robot_infor):
        self.camera_names = robot_infor['camera_names']
        self.camera_sensors = robot_infor['camera_sensors']

        self.arms = robot_infor['arms']
        self.robot_infor = robot_infor['controls']

        # 'joint_velocity_left', 'joint_velocity_right',
        # 'joint_effort_left', 'joint_effort_right',
        pass

    def decoder_image(self, camera_rgb_images, camera_depth_images):
        if type(camera_rgb_images[0]) is np.uint8:
            rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
            depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
            depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            return rgb, depth
        else:
            rgb_images = []
            depth_images = []
            for idx, camera_rgb_image in enumerate(camera_rgb_images):
                rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
                depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                rgb_images.append(rgb)
                depth_images.append(depth)
            rgb_images = np.asarray(rgb_images)
            depth_images = np.asarray(depth_images)
            return rgb_images, depth_images

    def execute(self, file_path, camera_frame=None, control_frame=None):
        image_dict = defaultdict(dict)
        control_dict = defaultdict(dict)
        base_dict = defaultdict(dict)

        with h5py.File(file_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']

            # select camera frame id
            for cam_name in self.camera_names:
                if is_compress:
                    if camera_frame is not None:
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                            camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][camera_frame])
                    else:
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][()],
                            camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][()])
                    image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
                    image_dict[self.camera_sensors[1]][cam_name] = decode_depth

                else:
                    if camera_frame:
                        image_dict[self.camera_sensors[0]][cam_name] = root[
                            'observations'][self.camera_sensors[0]][cam_name][camera_frame]
                        image_dict[self.camera_sensors[1]][cam_name] = root[
                            'observations'][self.camera_sensors[1]][cam_name][camera_frame]
                    else:
                        image_dict[self.camera_sensors[0]][cam_name] = root[
                           'observations'][self.camera_sensors[0]][cam_name][()]
                        image_dict[self.camera_sensors[1]][cam_name] = root[
                           'observations'][self.camera_sensors[1]][cam_name][()]
            # print('image_dict:',image_dict)

            for arm_name in self.arms:
                for control in self.robot_infor:
                    if control_frame:
                        control_dict[arm_name][control] = root[arm_name][control][control_frame]
                    else:
                        control_dict[arm_name][control] = root[arm_name][control][()]
            # print('infor_dict:',infor_dict)
        # gc.collect()
        return image_dict, control_dict, base_dict, is_sim, is_compress


if __name__ == '__main__':
    robot_infor = {'camera_names': ['camera_front', 'camera_left_wrist', 'camera_right_wrist'],
                   'camera_sensors': ['rgb_images', 'depth_images'],
                   'arms': ['master', 'puppet'],
                   'controls': ['joint_position_left', 'joint_position_right',
                              'end_effector_left', 'end_effector_right']}

    read_h5files = ReadH5Files(robot_infor)


    def get_files(dataset_dir):
        # dataset_dir_path = glob.glob("{}".format(dataset_dir))[0]
        files = []
        for trajectory_id in sorted(os.listdir(dataset_dir), key=int):
            trajectory_dir = os.path.join(dataset_dir, trajectory_id)
            file_path = os.path.join(dataset_dir, trajectory_dir, 'success/data/trajectory.hdf5')
            files.append(file_path)
        return files


    start_ts=5
    dataset_dir = '/media/zz/716c2609-b214-4080-80f6-4f272f36aeaf/mobile_aloha/dataset/930/make_bread/h5/kitchen/makebread'
    files = get_files(dataset_dir)
    import torch
    import tracemalloc
    tracemalloc.start()
    condition = 0
    while condition<500:
        for file_path in files:
            # file_path = '/home/zz/Project/Raw_Dataset/raw_to_h5/kitchen/breadcook/1/success/data/trajectory.hdf5'
            print('file_path:',file_path)
            print(f"Memory Usage: {psutil.virtual_memory().percent}%")
            image_dict, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(file_path,camera_frame=start_ts)

            master_joint_position_left = control_dict['master']['joint_position_left'][()]
            master_joint_position_right = control_dict['master']['joint_position_right'][()]
            action = np.concatenate([master_joint_position_left, master_joint_position_right], axis=-1)  # (n, 18)
            if len(base_dict) > 0 and self.use_robot_base:
                base_action = base_dict['base_action'][:-cutoff]
                base_action = preprocess_base_action(base_action)
                action = np.concatenate([action, base_action], axis=-1)
            else:
                dummy_base_action = np.zeros([action.shape[0], 2])  # (n, 2)
                action = np.concatenate([action, dummy_base_action], axis=-1)  # (n, 18)

            original_action_shape = action.shape
            episode_len = original_action_shape[0] - 0

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

            padded_action[:action_len] = action

            is_pad = np.zeros(episode_len)
            is_pad[action_len:] = 1

            padded_action = padded_action[:100]
            is_pad = is_pad[:100]

            # new axis for different cameras
            all_cam_images = []
            # print('image_dict:',image_dict.keys())
            # for cam_name in self.camera_names:
            for cam_name in robot_infor['camera_names']:
                all_cam_images.append(image_dict[robot_infor['camera_sensors'][0]][cam_name])  # rgb
            all_cam_images = np.stack(all_cam_images, axis=0)
            print('all_cam_images:',all_cam_images.shape)
            # construct observations
            # image_data1 = torch.from_numpy(all_cam_images/255).float()
            all_cam_images = (all_cam_images / 255.0).astype(np.float32)
            image_data1 = torch.from_numpy(all_cam_images)
            print('image_data1:',image_data1.shape,image_data1.dtype)
            # print('image_data1:',image_data,image_data.shape)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()
            # # channel last
            image_data1 = torch.einsum('k h w c -> k c h w', image_data1)
            # image_data2 = torch.einsum('k h w c -> k c h w', image_data2)

            #
            # # normalize image and change dtype to float
            print('image_data1:',image_data1.dtype,image_data1.shape)
            # image_data2 = image_data2 / 255.0

            # print('image_data2:',image_data2,image_data2.shape)
            condition +=1

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)


