import os
import pickle
from PIL import Image
from pathlib import Path

import sys
import numpy as np

import h5py
import cv2
from collections import defaultdict

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
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            return rgb, depth
        else:
            rgb_images = []
            depth_images = []
            # print(f"camera_rgb_images: {camera_rgb_images}")
            for idx, camera_rgb_image in enumerate(camera_rgb_images):
                # print(f"camera_rgb_image: {camera_rgb_image}")
                rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
                if camera_depth_images is not None:
                    depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                    depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                else:
                    depth = np.asarray([])
                rgb_images.append(rgb)
                depth_images.append(depth)
            rgb_images = np.asarray(rgb_images)
            depth_images = np.asarray(depth_images)
            return rgb_images, depth_images

    def execute(self, file_path, camera_frame=None, control_frame=None):
        with h5py.File(file_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']
            print(f"is_compress: {is_compress}")
            # select camera frame id
            is_compress = True
            image_dict = defaultdict(dict)
            for cam_name in self.camera_names:
                print(f"cam_name: {cam_name}")
                # if cam_name == 'camera_left' or cam_name == 'camera_right':
                #     continue
                if is_compress:
                    if camera_frame is not None:
                        # decode_rgb, decode_depth = self.decoder_image(
                        #     camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                        #     camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][camera_frame])
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                            camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][camera_frame])
                    else:
                        # decode_rgb, decode_depth = self.decoder_image(
                        #     camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][:],
                        #     camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][:])
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][:],
                            camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][:])
                        # for i in range(len(decode_rgb)):
                            # cv2.imwrite(os.path.join('/home/zz/Project/Raw_Dataset/test_image/', f'{cam_name}_{i}.jpg'), decode_rgb[i])
                    image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
                    image_dict[self.camera_sensors[1]][cam_name] = decode_depth
                    # image_dict[self.camera_sensors[1]][cam_name] = decode_depth

                else:
                    if camera_frame:
                        image_dict[self.camera_sensors[0]][cam_name] = root[
                            'observations'][self.camera_sensors[0]][cam_name][camera_frame]
                        image_dict[self.camera_sensors[1]][cam_name] = root[
                            'observations'][self.camera_sensors[1]][cam_name][camera_frame]
                        # image_dict[self.camera_sensors[1]][cam_name] = root[
                        #     'observations'][self.camera_sensors[1]][cam_name][camera_frame]
                    else:
                        image_dict[self.camera_sensors[0]][cam_name] = root[
                           'observations'][self.camera_sensors[0]][cam_name][:]
                        image_dict[self.camera_sensors[1]][cam_name] = root[
                           'observations'][self.camera_sensors[1]][cam_name][:]
                        # image_dict[self.camera_sensors[1]][cam_name] = root[
                        #    'observations'][self.camera_sensors[1]][cam_name][:]


            control_dict = defaultdict(dict)
            for arm_name in self.arms:
                for control in self.robot_infor:
                    if control_frame:
                        control_dict[arm_name][control] = root[arm_name][control][control_frame]
                    else:
                        control_dict[arm_name][control] = root[arm_name][control][:]
            # print('infor_dict:',infor_dict)
            base_dict = defaultdict(dict)
        # print('control_dict[puppet]:',control_dict['master']['joint_position_left'][0:1])
        return image_dict, control_dict, base_dict, is_sim, is_compress

def save_rgb_image(rgb_array, output_folder, filename):
    """
    Saves a NumPy array as an RGB image.
    
    Args:
    rgb_array (np.ndarray): A NumPy array representing the RGB image.
    output_folder (str): The path to the output folder.
    filename (str): The name of the output file.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert the NumPy array to an image
    rgb_array = rgb_array[:, :, ::-1].copy()  # Swap BGR to RGB
    # img = Image.fromarray((rgb_array * 255).astype(np.uint8))
    img = Image.fromarray((rgb_array).astype(np.uint8))
    
    # Save the image
    img.save(os.path.join(output_folder, filename))

def save_depth_image(depth_array, output_folder, filename):
    """
    Saves a NumPy array as a depth image.
    
    Args:
    depth_array (np.ndarray): A NumPy array representing the depth image.
    output_folder (str): The path to the output folder.
    filename (str): The name of the output file.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Normalize the depth array to the range [0, 255]
    depth_array_normalized = ((depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array)) * 255).astype(np.uint8)
    
    # Convert the normalized depth array to an image
    depth_img = Image.fromarray(depth_array_normalized)
    
    # Save the depth image
    depth_img.save(os.path.join(output_folder, filename))

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

def load_h5_and_save_images(source_folder, dest_folder):
    # # Franka_3rgb
    # robot_infor = {'camera_names': ['camera_left', 'camera_right', 'camera_top'],
    #                'camera_sensors': ['rgb_images', 'depth_images'],
    #                'arms': ['master', 'puppet'],
    #                'controls': ['joint_position']}

    # # Franka_1rgb
    # robot_infor = {'camera_names': ['camera_top'],
    #                'camera_sensors': ['rgb_images', 'depth_images'],
    #                'arms': ['master', 'puppet'],
    #                'controls': ['joint_position']}

    # UR_1rgb
    # robot_infor = {'camera_names': ['camera_top'],
    #                'camera_sensors': ['rgb_images', 'depth_images'],
    #                'arms': ['master', 'puppet'],
    #                'controls': ['joint_position']}
    
    # Songling_3rgb
    # robot_infor = {'camera_names': ['camera_front', 'camera_left_wrist', 'camera_right_wrist'],
    #                'camera_sensors': ['rgb_images', 'depth_images'],
    #                'arms': ['master', 'puppet'],
    #                'controls': ['joint_position_left', 'joint_position_right']}

    # tiangong_1rgb
    # robot_infor = {'camera_names': ['camera_top'],
    #                'camera_sensors': ['rgb_images', 'depth_images'],
    #                'arms': ['master', 'puppet'],
    #                'controls': ['joint_position', 'end_effector']}

    # simulation_4rgb
    # robot_infor = {'camera_names': ['camera_front_external', 'camera_handeye', 'camera_left_external', 'camera_right_external'],
    #                'camera_sensors': ['rgb_images', 'depth_images'],
    #                'arms': ['franka'],
    #                'controls': ['joint_position']}

    # new tiangong_1rgb
    robot_infor = {'camera_names': ['camera_top'],
                   'camera_sensors': ['rgb_images'],
                   'arms': ['master', 'puppet'],
                   'controls': ['joint_position', 'end_effector']}
    
    # Create the destination folder if it doesn't exist
    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    idx = 0

    # Walk through the source folder and its subdirectories
    for root, dirs, files in os.walk(source_folder):
        # print(f"files: {files}")
        if idx == 1:
            break
        # Get the relative path from the source folder
        rel_path = os.path.relpath(root, source_folder)
        # rel_path: 0923_220502
        # root: /media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/real_franka/success/240923_pick_bread_plate_1/0923_220502
        # print(f"rel_path: {rel_path}")
        # print(f"root: {root}")
        
        # Construct the corresponding destination path
        dest_path = os.path.join(dest_folder, rel_path)
        # dest_path: /media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/real_franka/success/240923_pick_bread_plate_1_image/0923_220502
        # print(f"dest_path: {dest_path}")
        
        # Create the corresponding destination directories
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        
        # Process each .pkl file in the current directory
        for idx, file in enumerate(files):
            print('=======================')
            print(f"idx:{idx} file: {file}")
            print(f"dest_path: {dest_path}")

            if file.endswith('.hdf5'):
                # Construct the full path to the .pkl file
                h5_file_path = os.path.join(root, file)
                print(f"root: {root}")
                print(f"h5_file_path: {h5_file_path}")

                # Open the HDF5 file in read-only mode
                with h5py.File(h5_file_path, 'r') as file:
                    # Print the structure of the HDF5 file
                    print_hdf5_structure(file)
                
                read_h5files = ReadH5Files(robot_infor)
                image_dict, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(h5_file_path)

                for sensor_key in image_dict.keys():
                    cur_sensor_dict = image_dict[sensor_key]
                    for cam_key in cur_sensor_dict.keys():
                        cur_img = cur_sensor_dict[cam_key]
                        print(f"sensor_key: {sensor_key}, cam_key:{cam_key}")
                        print(f"cur_img: {cur_img.shape}")
                        for img_idx in range(cur_img.shape[0]):
                            if sensor_key.find('rgb_images') >= 0:
                                rgb_filename = f'{img_idx}_{cam_key}_rgb.png'
                                rgb_array = cur_img[img_idx]
                                save_rgb_image(rgb_array, dest_path, rgb_filename)
                            if sensor_key.find('depth_images') >= 0:
                                depth_filename = f'{img_idx}_{cam_key}_depth.png'
                                depth_array = cur_img[img_idx]
                                save_depth_image(depth_array, dest_path, depth_filename)

                idx += 1       

if __name__ == "__main__":    
    ######### Franka 3RGB
    # source_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_franka_3rgb/pick_plate_from_plate_rack'
    # dest_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_franka_3rgb_img/pick_plate_from_plate_rack'

    ######### Franka 1RGB
    # source_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_franka_1rgb/bread_on_table' 
    # dest_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_franka_1rgb_img/bread_on_table'

    ######### UR 1RGB
    # source_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_ur_1rgb/pick_up_plastic_bottle' 
    # dest_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_ur_1rgb_img/pick_up_plastic_bottle'

    ######### Songling 3RGB
    # source_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_songling_3rgb/15_steamegg_2' 
    # dest_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_songling_3rgb_img/15_steamegg_2'

    # ######### tiangong 1RGB
    # source_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_tiangong_1rgb/push_break_pick_shelf_insert_machine_press_switch_place_plate' 
    # dest_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_tiangong_1rgb_img/push_break_pick_shelf_insert_machine_press_switch_place_plate'

    # ######### simulation 1RGB
    # source_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_simulation_4rgb/pick_and_place_06' 
    # dest_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_simulation_4rgb_img/pick_and_place_06'

    ######### new tiangong 1122 /media/data/h5_tiangong_1rgb/tiangong_data_1122_test
    source_folder = '/media/data/h5_tiangong_1rgb/tiangong_data_1122_test' 
    dest_folder = '/media/data/h5_tiangong_1rgb_img/tiangong_data_1122_test'

    load_h5_and_save_images(source_folder, dest_folder)



