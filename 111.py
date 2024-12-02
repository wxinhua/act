# import numpy as np

# left_hand_jpos = np.array([1,2,3,4,5,6,7])
# right_hand_jpos = np.array([1,2,3,4,5,6,7])
# left_jpos = np.array([1,2,3,4,5,6,7])
# right_jpos = np.array([1,2,3,4,5,6,7])

# qpos = np.concatenate((left_hand_jpos[3:5], right_hand_jpos[3:5], left_jpos, right_jpos))

# print(f"qpos: {qpos}")


from inrocs.robot_env.tianyi_env import tianyi_env
import h5py
import os
import pickle
from PIL import Image
from pathlib import Path

import sys
import numpy as np

import h5py
import cv2
from collections import defaultdict
import time
from dataset_load.read_franka_h5 import ReadH5Files

def print_hdf5_structure(hdf5_file, indent=0):
    """
    Recursively prints the structure of an HDF5 file.
    
    Args:
    hdf5_file (h5py.Group or h5py.File): The HDF5 group or file object.
    indent (int): The number of spaces to indent for pretty printing.
    """

    for key in hdf5_file.keys():
        item = hdf5_file[key]
        # Print the name and type of the item
        print(' ' * indent + f'{key} ({type(item).__name__})')
        
        if isinstance(item, h5py.Group):
            # If the item is a group, recurse
            print_hdf5_structure(item, indent + 2)
        elif isinstance(item, h5py.Dataset):
            # If the item is a dataset, print its shape and dtype
            print(' ' * (indent + 2) + f'Shape: {item.shape}, Type: {item.dtype}')
        elif isinstance(item, (list, np.ndarray)):
            # If the value is a list or a NumPy array, print its size
            size = len(item)
            print(' ' * (indent + 2) + f'Shape: {item.shape}, Type: {item.dtype}')
            # print(f"Size: {size}")
            # print(f"shape: {item.shape}")    


# qpos = tianyi_env.get_obs_full()['qpos']

# print(qpos)

# qpos[7] = 1.0
# tianyi_env.step_full(qpos)

# tianyi_env.reset_to_home()

# tianyi_env.reset_to_parepre()

# traj_list = ['/home/ps/wk/benchmark_results/tiangong_1122_traj.hdf5',
#             '/home/ps/wk/benchmark_results/tiangong_1122_traj_2.hdf5', 
#             '/home/ps/wk/benchmark_results/tiangong_place_button_traj.hdf5' ]
traj_list = ['/home/ps/wk/benchmark_results/tiangong_place_button_traj.hdf5']
# traj_list = ['/home/ps/wk/benchmark_results/tiangong_1122_traj.hdf5']

robot_infor = {'camera_names': ['camera_top'],
                'camera_sensors': ['rgb_images'],
                'arms': ['master', 'puppet'],
                'controls': ['joint_position', 'end_effector']}

read_h5files = ReadH5Files(robot_infor)

for idx, h5_file_path in enumerate(traj_list):
    with h5py.File(h5_file_path, 'r') as file:
        # Print the structure of the HDF5 file
        print_hdf5_structure(file)

    # ======= idx: 0
    # end_effect: (148, 12)
    # joint_position: (148, 14)
    # left_hand_jpos: [0.99400002 0.99400002 0.99000001 0.99199998 0.99800003 0.        ]
    # right_hand_jpos: [0.98799998 0.99000001 0.98799998 0.98799998 0.986      0.        ]
    # left_arm_jpos: [-0.74680512  0.56827277 -0.58608229 -0.72197765  1.04591412 -0.47618023
    # -0.57963957]
    # right_arm_jpos: [ 4.54250061e-03 -3.83495197e-06 -1.18095428e-01  2.14770733e-01
    # -4.19927241e-04 -4.59235498e-03  2.34750832e-01]
    # is_compress: True
    # cam_name: camera_top
    # ======= idx: 1
    # end_effect: (182, 12)
    # joint_position: (182, 14)
    # left_hand_jpos: [0.995      0.99699998 0.99400002 0.99299997 0.99199998 0.        ]
    # right_hand_jpos: [0.99900001 0.99800003 0.99699998 0.99900001 0.99599999 0.        ]
    # left_arm_jpos: [-1.03158674  0.40831309 -0.38225651 -0.36910262  0.45515703 -0.68055441
    # -0.10716773]
    # right_arm_jpos: [-0.00869192 -0.          0.01991107  0.15646221  0.01381733  0.02146423
    # -0.01239456]
    # is_compress: True
    # cam_name: camera_top
    # ======= idx: 2
    # end_effect: (730, 12)
    # joint_position: (730, 14)
    # left_hand_jpos: [0.69518881 0.73276971 0.69746964 0.7885623  0.78136782 0.30108138]
    # right_hand_jpos: [0.83555851 0.80696826 0.86742733 0.90796847 0.998      0.76228514]
    # left_arm_jpos: [ 0.09055231  0.28894955 -0.45861972 -0.25163102  0.08008693  0.11537293
    # 0.11239017]
    # right_arm_jpos: [ 0.63298313 -0.47553634  0.59014101  1.40712172 -1.9208488  -0.01315794
    # -0.48877337]



    image_dict, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(h5_file_path)
    print(f"======= idx: {idx}, is_compress:{is_compress}")
    end_effect = control_dict['puppet']['end_effector']
    joint_position = control_dict['puppet']['joint_position']
    print(f"end_effect: {end_effect.shape}")
    print(f"joint_position: {joint_position.shape}")
    # left_hand_jpos = end_effect[0, :6]
    # right_hand_jpos = end_effect[0, 6:12]
    # left_arm_jpos = joint_position[0, :7]
    # right_arm_jpos = joint_position[0, 7:14]
    # print(f"left_hand_jpos: {left_hand_jpos}")
    # print(f"right_hand_jpos: {right_hand_jpos}")
    # print(f"left_arm_jpos: {left_arm_jpos}")
    # print(f"right_arm_jpos: {right_arm_jpos}")

    left_hand_jpos = end_effect[:, :6]
    right_hand_jpos = end_effect[:, 6:12]
    left_arm_jpos = joint_position[:, :7]
    right_arm_jpos = joint_position[:, 7:14]
    
    if idx == 0:
        # img_dir = '/home/ps/wk/benchmark_results/'
        # cur_img = image_dict['rgb_images']['camera_top'][0]
        # img = Image.fromarray((cur_img).astype(np.uint8))
        # img.save(os.path.join(img_dir, str(idx)+'_rgb.png'))

        tianyi_env.reset_to_parepre()
        interval = 20 #20
        traj_len = left_arm_jpos.shape[0] // interval
        for i in range(traj_len):
            time.sleep(1)

            # # [:7],               [7:13]          [13:20]        [20:26]
            # # [self.left_jpos, self.left_hand, self.right_jpos, self.right_hand]
            # qpos = tianyi_env.get_obs_full()['qpos']
            # qpos[:7] = left_arm_jpos[i]
            # # qpos[7:13] = left_hand_jpos
            # qpos[13:20] = right_arm_jpos[i]
            # # qpos[20:26] = right_hand_jpos
            # # qpos = np.concatenate((left_hand_jpos, right_hand_jpos, left_arm_jpos, right_arm_jpos))
            # # time.sleep(1)
            # # tianyi_env.step_full(qpos)

            prepare_left = [-0.176873, 0.103522, -1.334014, -0.1800, 1.2640, -0.01137, 0.2419, 1.0]
            prepare_right = [0.55696, -0.043007, 1.26522, 0.94075, -0.58208, -0.10169, 0.11724, 1.0]
            prepare_left[:7] = left_arm_jpos[i*interval]
            prepare_right[:7] = right_arm_jpos[i*interval]
            print(f"======= step:{i*interval}")
            print(f"prepare_left: {prepare_left}")
            print(f"prepare_right: {prepare_right}")
            tianyi_env.move_to_target(prepare_left + prepare_right)
    
        




