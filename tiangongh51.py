import os
import h5py
import rosbag
from tqdm import tqdm
import io
import numpy as np
import cv_bridge
import cv2
import shutil
import argparse
from bisect import bisect_left
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import scipy.interpolate as interp

import cv_bridge
import collections

class Raw2TRAJ:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.top = 0
        self.bottom = 80
        self.left = 0
        self.right = 0

        self.set_sim = False
        self.set_compress = True

        pass

    def extract_sorted_timestamps_and_messages(self, bag_file):
        try:
            bag = rosbag.Bag(bag_file)
            timestamps = []
            messages = []
            topics = []
            for topic, msg, t in bag.read_messages():
                timestamps.append(t.to_sec())
                messages.append((t.to_sec(), msg))
                topics.append(topic)
            bag.close()
        except Exception as e:
            print(f"Error reading bag file {bag_file}: {e}")
            return None
        return sorted(zip(timestamps, messages, topics))

    def find_closest_timestamp(self,timestamps, target, tolerance=np.inf):
        closest_timestamp = min(timestamps, key=lambda t: abs(t - target))
        if abs(closest_timestamp - target) <= tolerance:
            return closest_timestamp
        return None

    def align_high_freq_data(self, bag1_times, bag2_times, bag2_values, bag2_topics=None):

        aligned_bag2_values = []
        aligned_bag2_topics = []
        for t in bag1_times:
            # 找到与 bag1 时间戳最近的 bag2 数据
            closest_idx = np.argmin(np.abs(bag2_times - t))
            aligned_bag2_values.append(bag2_values[closest_idx])
            if bag2_topics is not None:
                aligned_bag2_topics.append(bag2_topics[closest_idx])
        return aligned_bag2_values, aligned_bag2_topics

    def align_low_freq_data(self, bag1_times, low_freq_times, low_freq_values, low_topics):
        """ 将低频数据对齐到高频数据 """
        aligned_values = []
        aligned_topics = []
        for t in bag1_times:
            # 找到与bag1时间戳最近的低频数据
            closest_idx = np.argmin(np.abs(low_freq_times - t))
            aligned_values.append(low_freq_values[closest_idx])
            aligned_topics.append(low_topics[closest_idx])
        return aligned_values, aligned_topics

    def interpolate_joint_values(self, low_freq_data, low_freq_times, high_freq_times):
        # low_freq_data: 低频数据，形状为 (N, J)，N 是时间步，J 是关节数量
        # low_freq_times: 低频数据的时间戳，形状为 (N,)
        # high_freq_times: 高频时间戳，形状为 (M,)

        interpolated_data = np.zeros((len(high_freq_times), low_freq_data.shape[1]))

        for joint_idx in range(low_freq_data.shape[1]):
            # 针对每个关节，使用 Cubic Spline 插值
            spline = CubicSpline(low_freq_times, low_freq_data[:, joint_idx])
            interpolated_data[:, joint_idx] = spline(high_freq_times)

        return interpolated_data

    def downsample_by_closest(self, high_freq_data, high_freq_times, low_freq_times):
        downsampled_data = np.zeros((len(low_freq_times), high_freq_data.shape[1]))

        for i, t in enumerate(low_freq_times):
            # 找到与低频时间戳最近的高频时间点
            closest_idx = np.argmin(np.abs(high_freq_times - t))
            downsampled_data[i] = high_freq_data[closest_idx]

        return downsampled_data

    def increase_joint_data(self, joint_values, num_interpolated_points=2):
        # joint_values: 原始的关节矩阵，形状为 (n, 7)，n 是记录的时间点数量
        # num_interpolated_points: 每两个原始点之间插值的点数

        # 时间点对应的索引作为自变量
        original_indices = np.arange(len(joint_values))

        # 构建插值后新的时间点索引
        interpolated_indices = np.linspace(0, len(joint_values) - 1,
                                           len(joint_values) + (len(joint_values) - 1) * num_interpolated_points)

        # 对每个关节（7个）分别进行插值
        interpolated_joint_values = []
        for i in range(joint_values.shape[1]):
            interp_func = interp1d(original_indices, joint_values[:, i], kind='cubic')
            interpolated_joint_values.append(interp_func(interpolated_indices))

        # 将插值后的关节值组合成一个新的矩阵
        interpolated_joint_values = np.stack(interpolated_joint_values, axis=-1)

        return interpolated_joint_values

    def interpolate_data(self, bag1_times, bag5_times, bag5_values):
        """ 使用插值将bag5对齐到bag1 """
        interpolator = interp1d(bag5_times, bag5_values, axis=0, fill_value="extrapolate", kind='linear')
        aligned_bag5_values = interpolator(bag1_times)
        return aligned_bag5_values

    def align_list_data(self, left_joint_timestamps, left_joint_position, left_joint_topics,
                        right_joint_timestamps, right_joint_position, right_joint_topics,
                        right_end_timestamps, right_end_position, right_end_topics,
                        left_end_timestamps, left_end_position, left_end_topics,
                        align_arm):
        if align_arm == 'left':
            aligned_joint_timestamps = left_joint_timestamps
            aligned_left_joint_positions = left_joint_position
            aligned_left_joint_topics = left_joint_topics
            aligned_right_joint_positions, aligned_right_joint_topics = self.align_high_freq_data(aligned_joint_timestamps,
                                                                                             right_joint_timestamps,
                                                                                             right_joint_position,
                                                                                             right_joint_topics)

            ###
            # aligned_end_timestamps = left_end_timestamps
            # aligned_left_end_positions = left_end_position
            # aligned_left_end_topics = left_end_topics
            aligned_right_end_positions, aligned_right_end_topics = self.align_low_freq_data(aligned_joint_timestamps,
                                                                                        right_end_timestamps,
                                                                                        right_end_position,
                                                                                        right_end_topics)

            aligned_left_end_positions, aligned_left_end_topics = self.align_low_freq_data(aligned_joint_timestamps,
                                                                                        left_end_timestamps,
                                                                                        left_end_position,
                                                                                        left_end_topics)

        elif align_arm == 'right':
            aligned_joint_timestamps = right_joint_timestamps
            aligned_right_joint_positions = right_joint_position
            aligned_right_joint_topics = right_joint_topics
            aligned_left_joint_positions, aligned_left_joint_topics = self.align_high_freq_data(aligned_joint_timestamps,
                                                                                           left_joint_timestamps,
                                                                                           left_joint_position,
                                                                                           left_joint_topics)

            ###
            # aligned_end_timestamps = right_end_timestamps
            # aligned_right_end_positions = right_end_position
            # aligned_right_end_topics = right_end_topics
            aligned_left_end_positions, aligned_left_end_topics = self.align_low_freq_data(aligned_joint_timestamps,
                                                                                            left_end_timestamps,
                                                                                            left_end_position,
                                                                                            left_end_topics)

            aligned_right_end_positions, aligned_right_end_topics = self.align_low_freq_data(aligned_joint_timestamps,
                                                                                            right_end_timestamps,
                                                                                            right_end_position,
                                                                                            right_end_topics)

        else:
            pass
        # &&&&&&&&
        aligned_joint_timestamps = np.asarray(aligned_joint_timestamps)

        aligned_left_end_positions = np.asarray(aligned_left_end_positions)
        aligned_right_end_positions = np.asarray(aligned_right_end_positions)

        aligned_left_joint_positions = np.asarray(aligned_left_joint_positions)
        aligned_right_joint_positions = np.asarray(aligned_right_joint_positions)

        # print('aligned_left_end_positions:', aligned_left_end_positions.shape)
        # print('aligned_right_end_positions:', aligned_right_end_positions.shape)
        # print('aligned_left_joint_positions:', aligned_left_joint_positions.shape)
        # print('aligned_right_joint_positions:', aligned_right_joint_positions.shape)
        # print('aligned_joint_timestamps:', aligned_joint_timestamps.shape)

        return (aligned_joint_timestamps, aligned_left_end_positions, aligned_right_end_positions,
                aligned_left_joint_positions, aligned_right_joint_positions)


    def align_hand_to_joint(self, hand_bag_file, arm_bag_file, align_arm='right'):

        execution_result = collections.defaultdict(dict)

        end_data = self.extract_sorted_timestamps_and_messages(hand_bag_file)
        if end_data is None:
            return None
        joint_data = self.extract_sorted_timestamps_and_messages(arm_bag_file)
        if joint_data is None:
            return None

        # print('end_data:',end_data)
        # print('joint_data:',joint_data)

        state_left_end_timestamps = []
        state_left_end_position = []
        state_left_end_topics = []

        state_right_end_timestamps = []
        state_right_end_position = []
        state_right_end_topics = []

        ctrl_left_end_timestamps = []
        ctrl_left_end_position = []
        ctrl_left_end_topics = []

        ctrl_right_end_timestamps = []
        ctrl_right_end_position = []
        ctrl_right_end_topics = []

        for end_messages in end_data:
            end_timestamp = end_messages[0]
            end_message = end_messages[1]
            end_topic = end_messages[2]
            end_position = list(end_message[1].position)
            end_effort = list(end_message[1].effort)

            if 'state/left' in end_topic:
                state_left_end_timestamps.append(end_timestamp)
                state_left_end_position.append(end_position)
                state_left_end_topics.append(end_topic)
            elif 'state/right' in end_topic:
                state_right_end_timestamps.append(end_timestamp)
                state_right_end_position.append(end_position)
                state_right_end_topics.append(end_topic)
            elif 'ctrl/left' in end_topic:
                ctrl_left_end_timestamps.append(end_timestamp)
                ctrl_left_end_position.append(end_position)
                ctrl_left_end_topics.append(end_topic)
            elif 'ctrl/right' in end_topic:
                ctrl_right_end_timestamps.append(end_timestamp)
                ctrl_right_end_position.append(end_position)
                ctrl_right_end_topics.append(end_topic)
            else:
                pass
            # print('end_topics:',end_topics) # '/inspire_hand/state/left_hand'; '/inspire_hand/state/right_hand'

        state_left_joint_timestamps = []
        state_left_joint_position = []
        state_left_joint_topics = []

        state_right_joint_timestamps = []
        state_right_joint_position = []
        state_right_joint_topics = []

        ctrl_left_joint_timestamps = []
        ctrl_left_joint_position = []
        ctrl_left_joint_topics = []

        ctrl_right_joint_timestamps = []
        ctrl_right_joint_position = []
        ctrl_right_joint_topics = []

        for joint_messages in joint_data:
            joint_timestamp = joint_messages[0]
            joint_message = joint_messages[1]
            joint_topic = joint_messages[2]
            joint_position = joint_message[1].position

            if 'state_left' in joint_topic:
                state_left_joint_timestamps.append(joint_timestamp)
                state_left_joint_position.append(joint_position)
                state_left_joint_topics.append(joint_topic)
            elif 'state_right' in joint_topic:
                state_right_joint_timestamps.append(joint_timestamp)
                state_right_joint_position.append(joint_position)
                state_right_joint_topics.append(joint_topic)
            if 'ctrl_left' in joint_topic:
                ctrl_left_joint_timestamps.append(joint_timestamp)
                ctrl_left_joint_position.append(joint_position)
                ctrl_left_joint_topics.append(joint_topic)
            elif 'ctrl_right' in joint_topic:
                ctrl_right_joint_timestamps.append(joint_timestamp)
                ctrl_right_joint_position.append(joint_position)
                ctrl_right_joint_topics.append(joint_topic)
            else:
                pass


        state_left_joint_timestamps = np.asarray(state_left_joint_timestamps)
        state_left_end_timestamps = np.asarray(state_left_end_timestamps)

        state_right_joint_timestamps = np.asarray(state_right_joint_timestamps)
        state_right_end_timestamps = np.asarray(state_right_end_timestamps)

        state_left_joint_position = np.asarray(state_left_joint_position)
        state_right_joint_position = np.asarray(state_right_joint_position)

        state_left_end_position = np.asarray(state_left_end_position)
        state_right_end_position = np.asarray(state_right_end_position)


        ctrl_left_joint_timestamps = np.asarray(ctrl_left_joint_timestamps)
        ctrl_left_end_timestamps = np.asarray(ctrl_left_end_timestamps)

        ctrl_right_joint_timestamps = np.asarray(ctrl_right_joint_timestamps)
        ctrl_right_end_timestamps = np.asarray(ctrl_right_end_timestamps)

        ctrl_left_joint_position = np.asarray(ctrl_left_joint_position)
        ctrl_right_joint_position = np.asarray(ctrl_right_joint_position)

        ctrl_left_end_position = np.asarray(ctrl_left_end_position)
        ctrl_right_end_position = np.asarray(ctrl_right_end_position)


        # print('state_left_joint_position:', state_left_joint_position.shape)
        # print('state_right_joint_position:', state_right_joint_position.shape)
        # print('state_left_end_position:', state_left_end_position.shape)
        # print('state_right_end_position:', state_right_end_position.shape)

        # print('ctrl_left_joint_position:', ctrl_left_joint_position.shape)
        # print('ctrl_right_joint_position:', ctrl_right_joint_position.shape)
        # print('ctrl_left_end_position:', ctrl_left_end_position.shape)
        # print('ctrl_right_end_position:', ctrl_right_end_position.shape)

        assert len(ctrl_left_joint_timestamps) == len(ctrl_left_joint_position)
        assert len(ctrl_right_joint_timestamps) == len(ctrl_right_joint_position)

        assert len(state_left_joint_timestamps) == len(state_left_joint_position)
        assert len(state_right_joint_timestamps) == len(state_right_joint_position)

        org_ctrl_left_joint_dict = {}
        for i in range(len(ctrl_left_joint_timestamps)):
            org_ctrl_left_joint_dict[ctrl_left_joint_timestamps[i]] = ctrl_left_joint_position[i]

        org_ctrl_right_joint_dict = {}
        for i in range(len(ctrl_right_joint_timestamps)):
            org_ctrl_right_joint_dict[ctrl_right_joint_timestamps[i]] = ctrl_right_joint_position[i]

        org_ctrl_left_end_dict = {}
        for i in range(len(ctrl_left_end_timestamps)):
            org_ctrl_left_end_dict[ctrl_left_end_timestamps[i]] = ctrl_left_end_position[i]

        org_ctrl_right_end_dict = {}
        for i in range(len(ctrl_right_end_timestamps)):
            org_ctrl_right_end_dict[ctrl_right_end_timestamps[i]] = ctrl_right_end_position[i]

        #############################
        org_state_left_joint_dict = {}
        for i in range(len(state_left_joint_timestamps)):
            org_state_left_joint_dict[state_left_joint_timestamps[i]] = state_left_joint_position[i]

        org_state_right_joint_dict = {}
        for i in range(len(state_right_joint_timestamps)):
            org_state_right_joint_dict[state_right_joint_timestamps[i]] = state_right_joint_position[i]

        org_state_left_end_dict = {}
        for i in range(len(state_left_end_timestamps)):
            org_state_left_end_dict[state_left_end_timestamps[i]] = state_left_end_position[i]

        org_state_right_end_dict = {}
        for i in range(len(state_right_end_timestamps)):
            org_state_right_end_dict[state_right_end_timestamps[i]] = state_right_end_position[i]
        #############################

        # print('state_left_joint_position:',state_left_joint_position)

        increase_state_left_joint_positions = self.increase_joint_data(state_left_joint_position, num_interpolated_points=2)

        increase_state_right_joint_positions = self.increase_joint_data(state_right_joint_position, num_interpolated_points=2)

        increase_state_left_end_positions = self.increase_joint_data(state_left_end_position, num_interpolated_points=2)

        increase_state_right_end_positions = self.increase_joint_data(state_right_end_position, num_interpolated_points=2)

        ##############################
        increase_ctrl_left_joint_positions = self.increase_joint_data(ctrl_left_joint_position, num_interpolated_points=2)

        increase_ctrl_right_joint_positions = self.increase_joint_data(ctrl_right_joint_position, num_interpolated_points=2)

        increase_ctrl_left_end_positions = self.increase_joint_data(ctrl_left_end_position, num_interpolated_points=2)

        increase_ctrl_right_end_positions = self.increase_joint_data(ctrl_right_end_position, num_interpolated_points=2)


        execution_result['org_left_joint_dict']['ctrl'] = org_ctrl_left_joint_dict
        execution_result['org_right_joint_dict']['ctrl'] = org_ctrl_right_joint_dict

        execution_result['org_left_end_dict']['ctrl'] = org_ctrl_left_end_dict
        execution_result['org_right_end_dict']['ctrl'] = org_ctrl_right_end_dict

        execution_result['increase_org_left_joint_positions']['ctrl'] = np.asarray(increase_ctrl_left_joint_positions)
        execution_result['increase_org_right_joint_positions']['ctrl'] = np.asarray(increase_ctrl_right_joint_positions)

        execution_result['increase_org_left_end_positions']['ctrl'] = np.asarray(increase_ctrl_left_end_positions)
        execution_result['increase_org_right_end_positions']['ctrl'] = np.asarray(increase_ctrl_right_end_positions)

        ################################
        execution_result['org_left_joint_dict']['state'] = org_state_left_joint_dict
        execution_result['org_right_joint_dict']['state'] = org_state_right_joint_dict

        execution_result['org_left_end_dict']['state'] = org_state_left_end_dict
        execution_result['org_right_end_dict']['state'] = org_state_right_end_dict

        execution_result['increase_org_left_joint_positions']['state'] = np.asarray(increase_state_left_joint_positions)
        execution_result['increase_org_right_joint_positions']['state'] = np.asarray(increase_state_right_joint_positions)

        execution_result['increase_org_left_end_positions']['state'] = np.asarray(increase_state_left_end_positions)
        execution_result['increase_org_right_end_positions']['state'] = np.asarray(increase_state_right_end_positions)

        (aligned_ctrl_joint_timestamps, aligned_ctrl_left_end_positions, aligned_ctrl_right_end_positions,
         aligned_ctrl_left_joint_positions, aligned_ctrl_right_joint_positions) = self.align_list_data(
            ctrl_left_joint_timestamps, ctrl_left_joint_position, ctrl_left_joint_topics,
            ctrl_right_joint_timestamps, ctrl_right_joint_position, ctrl_right_joint_topics,
            ctrl_right_end_timestamps, ctrl_right_end_position, ctrl_right_end_topics,
            ctrl_left_end_timestamps, ctrl_left_end_position, ctrl_left_end_topics, align_arm=align_arm)

        (aligned_state_joint_timestamps, aligned_state_left_end_positions, aligned_state_right_end_positions,
         aligned_state_left_joint_positions, aligned_state_right_joint_positions) = self.align_list_data(
            state_left_joint_timestamps, state_left_joint_position, state_left_joint_topics,
            state_right_joint_timestamps, state_right_joint_position, state_right_joint_topics,
            state_right_end_timestamps, state_right_end_position, state_right_end_topics,
            state_left_end_timestamps, state_left_end_position, state_left_end_topics, align_arm=align_arm)


        aligned_state_left_joint_dict = {}
        for i in range(len(aligned_state_joint_timestamps)):
            aligned_state_left_joint_dict[aligned_state_joint_timestamps[i]] = aligned_state_left_joint_positions[i]

        aligned_state_right_joint_dict = {}
        for i in range(len(aligned_state_joint_timestamps)):
            aligned_state_right_joint_dict[aligned_state_joint_timestamps[i]] = aligned_state_right_joint_positions[i]

        aligned_state_left_end_dict = {}
        for i in range(len(aligned_state_joint_timestamps)):
            aligned_state_left_end_dict[aligned_state_joint_timestamps[i]] = aligned_state_left_end_positions[i]

        aligned_state_right_end_dict = {}
        for i in range(len(aligned_state_joint_timestamps)):
            aligned_state_right_end_dict[aligned_state_joint_timestamps[i]] = aligned_state_right_end_positions[i]

        ########################################
        aligned_ctrl_left_joint_dict = {}
        for i in range(len(aligned_ctrl_joint_timestamps)):
            aligned_ctrl_left_joint_dict[aligned_ctrl_joint_timestamps[i]] = aligned_ctrl_left_joint_positions[i]

        aligned_ctrl_right_joint_dict = {}
        for i in range(len(aligned_ctrl_joint_timestamps)):
            aligned_ctrl_right_joint_dict[aligned_ctrl_joint_timestamps[i]] = aligned_ctrl_right_joint_positions[i]

        aligned_ctrl_left_end_dict = {}
        for i in range(len(aligned_ctrl_joint_timestamps)):
            aligned_ctrl_left_end_dict[aligned_ctrl_joint_timestamps[i]] = aligned_ctrl_left_end_positions[i]

        aligned_ctrl_right_end_dict = {}
        for i in range(len(aligned_ctrl_joint_timestamps)):
            aligned_ctrl_right_end_dict[aligned_ctrl_joint_timestamps[i]] = aligned_ctrl_right_end_positions[i]


        execution_result['aligned_left_joint_dict']['ctrl'] = aligned_ctrl_left_joint_dict
        execution_result['aligned_right_joint_dict']['ctrl'] = aligned_ctrl_right_joint_dict

        execution_result['aligned_left_end_dict']['ctrl'] = aligned_ctrl_left_end_dict
        execution_result['aligned_right_end_dict']['ctrl'] = aligned_ctrl_right_end_dict
        execution_result['aligned_joint_timestamps']['ctrl'] = aligned_ctrl_joint_timestamps

        execution_result['aligned_left_joint_dict']['state'] = aligned_state_left_joint_dict
        execution_result['aligned_right_joint_dict']['state'] = aligned_state_right_joint_dict

        execution_result['aligned_left_end_dict']['state'] = aligned_state_left_end_dict
        execution_result['aligned_right_end_dict']['state'] = aligned_state_right_end_dict
        execution_result['aligned_joint_timestamps']['state'] = aligned_state_joint_timestamps

        increase_aligned_ctrl_left_joint_positions = self.increase_joint_data(aligned_ctrl_left_joint_positions,
                                                                         num_interpolated_points=2)

        increase_aligned_ctrl_right_joint_positions = self.increase_joint_data(aligned_ctrl_right_joint_positions,
                                                                         num_interpolated_points=2)

        increase_aligned_ctrl_left_end_positions = self.increase_joint_data(aligned_ctrl_left_end_positions,
                                                                         num_interpolated_points=2)

        increase_aligned_ctrl_right_end_positions = self.increase_joint_data(aligned_ctrl_right_end_positions,
                                                                         num_interpolated_points=2)
        #########
        increase_aligned_state_left_joint_positions = self.increase_joint_data(aligned_state_left_joint_positions,
                                                                         num_interpolated_points=2)

        increase_aligned_state_right_joint_positions = self.increase_joint_data(aligned_state_right_joint_positions,
                                                                         num_interpolated_points=2)

        increase_aligned_state_left_end_positions = self.increase_joint_data(aligned_state_left_end_positions,
                                                                         num_interpolated_points=2)

        increase_aligned_state_right_end_positions = self.increase_joint_data(aligned_state_right_end_positions,
                                                                         num_interpolated_points=2)


        # print('increase_aligned_ctrl_left_joint_positions:',len(increase_aligned_ctrl_left_joint_positions))
        # print('increase_aligned_ctrl_right_joint_positions:',len(increase_aligned_ctrl_right_joint_positions))
        # print('increase_aligned_ctrl_left_end_positions:',len(increase_aligned_ctrl_left_end_positions))
        # print('increase_aligned_ctrl_right_end_positions:',len(increase_aligned_ctrl_right_end_positions))


        execution_result['increase_aligned_left_joint_positions']['ctrl'] = np.asarray(increase_aligned_ctrl_left_joint_positions)
        execution_result['increase_aligned_right_joint_positions']['ctrl'] = np.asarray(increase_aligned_ctrl_right_joint_positions)

        execution_result['increase_aligned_left_end_positions']['ctrl'] = np.asarray(increase_aligned_ctrl_left_end_positions)
        execution_result['increase_aligned_right_end_positions']['ctrl'] = np.asarray(increase_aligned_ctrl_right_end_positions)

        ##############
        execution_result['increase_aligned_left_joint_positions']['state'] = np.asarray(
            increase_aligned_state_left_joint_positions)
        execution_result['increase_aligned_right_joint_positions']['state'] = np.asarray(
            increase_aligned_state_right_joint_positions)

        execution_result['increase_aligned_left_end_positions']['state'] = np.asarray(
            increase_aligned_state_left_end_positions)
        execution_result['increase_aligned_right_end_positions']['state'] = np.asarray(
            increase_aligned_state_right_end_positions)

        # plt.figure(figsize=(10, 6))
        # for i in range(left_joint):
        #     plt.plot(left_joint_timestamps, left_joint_position[:, i], label=f'Joint {i + 1}')
        # plt.legend()
        # plt.title("Joint Value Interpolation (Cubic Spline)")
        # plt.xlabel("Time")
        # plt.ylabel("Joint Value")
        # plt.show()


        return (execution_result, aligned_ctrl_joint_timestamps, aligned_ctrl_left_end_positions,
                aligned_ctrl_right_end_positions, aligned_ctrl_left_joint_positions, aligned_ctrl_right_joint_positions)


    def align_multi_camera(self, camera_bag_file):
        camera_data = self.extract_sorted_timestamps_and_messages(camera_bag_file)
        if camera_data is None:
            return None

        camera_color_images = []
        camera_depth_images = []
        camera_depth_timestamps = []
        camera_color_timestamps = []

        for camera_messages in camera_data:
            camera_timestamp = camera_messages[0]
            camera_message = camera_messages[1]
            camera_topic = camera_messages[2]
            width = camera_message[1].width
            height = camera_message[1].height
            # print('camera_topic:',camera_topic)
            if '/camera/depth/image_raw' == camera_topic:

                # camera_front_depth_image = np.frombuffer(camera_message[1].data, dtype=np.uint16)
                # camera_front_depth_image = np.reshape(camera_front_depth_image, (height, width))
                camera_depth_image = self.bridge.imgmsg_to_cv2(camera_message[1], 'passthrough')
                camera_depth_images.append(camera_depth_image)
                camera_depth_timestamps.append(camera_timestamp)

            elif '/camera/color/image_raw' == camera_topic:
                camera_color_image = self.bridge.imgmsg_to_cv2(camera_message[1], 'passthrough')
                camera_color_images.append(camera_color_image)
                camera_color_timestamps.append(camera_timestamp)

        # print('camera_depth_timestamps:',len(camera_depth_timestamps))
        # print('camera_color_timestamps:',len(camera_color_timestamps))
        camera_color_timestamps = np.asarray(camera_color_timestamps)
        camera_depth_timestamps = np.asarray(camera_depth_timestamps)

        camera_color_images = np.asarray(camera_color_images)
        camera_depth_images = np.asarray(camera_depth_images)

        # bag1_times, bag2_times, bag2_values, bag2_topics
        camera_timestamps = camera_color_timestamps
        camera_depth_images, _ = self.align_high_freq_data(camera_timestamps,
                                                           camera_depth_timestamps, camera_depth_images)
        camera_depth_images = np.asarray(camera_depth_images)
        return camera_timestamps, camera_color_images, camera_depth_images

    def set_h5_data(self, aligned_left_joint_data, aligned_right_joint_data,
                    aligned_left_end_data, aligned_right_end_data,
                    camera_color_images, camera_depth_images, camera_timestamps, h5_file_path):

        camera_rgb_image_all = []
        camera_depth_image_all = []

        print('camera_color_images:',camera_color_images.shape)
        print('camera_depth_images:',camera_depth_images.shape)
        for i in range(len(camera_color_images)):
            camera_rgb_image = camera_color_images[i]
            camera_depth_image = camera_depth_images[i]

            # np.save('camera_front_depth_image_0.npy', camera_front_depth_image)
            # quit()
            if self.set_compress:
                camera_rgb_image = cv2.imencode(".jpg", camera_rgb_image)[1]#.tobytes()
                camera_depth_image = cv2.imencode(".png", camera_depth_image)[1]#.tobytes()

            camera_rgb_image_all.append(camera_rgb_image)
            camera_depth_image_all.append(camera_depth_image)

        if not self.set_compress:
            camera_rgb_image_all = np.asarray(camera_rgb_image_all)
            camera_depth_image_all = np.asarray(camera_depth_image_all)

        # print('aligned_left_joint_data:',aligned_left_joint_data.shape)
        # print('aligned_right_joint_data:',aligned_right_joint_data.shape)

        # print('aligned_left_end_data:',aligned_left_end_data.shape)
        # print('aligned_right_end_data:',aligned_right_end_data.shape)

        joint_position_all = np.c_[aligned_left_joint_data, aligned_right_joint_data]
        end_effector_all = np.c_[aligned_left_end_data, aligned_right_end_data]

        with h5py.File(h5_file_path, 'w') as root:
            root.attrs['sim'] = False
            root.attrs['compress'] = self.set_compress
            obs = root.create_group('observations')
            rgb_images = obs.create_group('rgb_images')
            depth_images = obs.create_group('depth_images')

            if self.set_compress:
                dt = h5py.vlen_dtype(np.dtype('uint8'))

                dset_top_images = rgb_images.create_dataset('camera_top',
                                                            (len(camera_rgb_image_all),), dtype=dt)
                for i, camera_top_rgb_image in enumerate(camera_rgb_image_all):
                    # print(camera_top_rgb_image.shape)
                    dset_top_images[i] = camera_top_rgb_image.flatten()

                ############################
                dt2 = h5py.special_dtype(vlen=np.dtype('uint8'))

                dset_top_depth = depth_images.create_dataset('camera_top',
                                                             (len(camera_depth_image_all),), dtype=dt2)
                for i, camera_top_depth_image in enumerate(camera_depth_image_all):
                    dset_top_depth[i] = np.frombuffer(camera_top_depth_image.tobytes(), dtype=np.uint8)

            else:
                rgb_images.create_dataset('camera_top', data=camera_rgb_image_all)
                depth_images.create_dataset('camera_top', data=camera_depth_image_all)

            master = root.create_group('master')
            master.create_dataset('joint_position', data=joint_position_all)

            puppet = root.create_group('puppet')
            puppet.create_dataset('joint_position', data=joint_position_all)
            puppet.create_dataset('end_effector', data=end_effector_all)

    def execute(self, base_path, save_traj_dir_path):
        for task in tqdm(os.listdir(base_path)):
                traj_task_path = os.path.join(save_traj_dir_path, task)
                if not os.path.exists(traj_task_path):
                    os.makedirs(traj_task_path)
                task_path = os.path.join(base_path, task)
                print('task_path:',task_path)
                for success_or_failure in os.listdir(task_path):
                    if 'success' in success_or_failure:
                        success_path = os.path.join(task_path, success_or_failure)
                        traj_success_path = os.path.join(traj_task_path, success_or_failure)
                        if not os.path.exists(traj_success_path):
                            os.makedirs(traj_success_path)
                        print('success_path:',success_path)
                        for collect_time in sorted(os.listdir(success_path)):
                            try:
                                collect_time_path = os.path.join(success_path, collect_time)
                                traj_collect_time_path = os.path.join(traj_success_path, collect_time)
                                if not os.path.exists(traj_collect_time_path):
                                    os.makedirs(traj_collect_time_path)
                                data_path = os.path.join(collect_time_path, 'data')
                                if len(os.listdir(data_path)) > 0:

                                    traj_data_path = os.path.join(traj_collect_time_path, 'data_traj')
                                    if not os.path.exists(traj_data_path):
                                        os.makedirs(traj_data_path)
                                    h5_data_path = os.path.join(traj_collect_time_path, 'data')
                                    if not os.path.exists(h5_data_path):
                                        os.makedirs(h5_data_path)

                                    h5_file_path = os.path.join(h5_data_path, 'trajectory.hdf5')

                                    # try:
                                    camera_h_color_file = os.path.join(data_path, 'head_camera.bag')

                                    arm_joint_file = os.path.join(data_path, 'arm.bag')
                                    end_hand_file = os.path.join(data_path, 'hand.bag')

                                    (execution_result, aligned_joint_timestamps, aligned_left_end_positions,
                                    aligned_right_end_positions, aligned_left_joint_positions,
                                    aligned_right_joint_positions) = self.align_hand_to_joint(
                                        end_hand_file, arm_joint_file, align_arm='right')
                                    # data_name = 'trajectory.npy'
                                    # np.save(os.path.join(traj_data_path, data_name), execution_result)


                                    camera_timestamps, camera_color_images, camera_depth_images = self.align_multi_camera(
                                        camera_h_color_file)

                                    aligned_left_joint_data = self.downsample_by_closest(
                                        high_freq_data=aligned_left_joint_positions,
                                        high_freq_times=aligned_joint_timestamps,
                                        low_freq_times=camera_timestamps)

                                    aligned_right_joint_data = self.downsample_by_closest(
                                        high_freq_data=aligned_right_joint_positions,
                                        high_freq_times=aligned_joint_timestamps,
                                        low_freq_times=camera_timestamps)

                                    aligned_left_end_data = self.downsample_by_closest(
                                        high_freq_data=aligned_left_end_positions,
                                        high_freq_times=aligned_joint_timestamps,
                                        low_freq_times=camera_timestamps)

                                    aligned_right_end_data = self.downsample_by_closest(
                                        high_freq_data=aligned_right_end_positions,
                                        high_freq_times=aligned_joint_timestamps,
                                        low_freq_times=camera_timestamps)

                                    assert len(aligned_left_joint_data) == len(aligned_right_joint_data)
                                    assert len(aligned_right_joint_data) == len(aligned_left_end_data)
                                    assert len(aligned_left_end_data) == len(aligned_right_end_data)

                                    self.set_h5_data(aligned_left_joint_data, aligned_right_joint_data,
                                                    aligned_left_end_data, aligned_right_end_data,
                                                    camera_color_images, camera_depth_images, camera_timestamps,
                                                    h5_file_path)
                            except Exception as e:
                                print(f"Error processing task {task}: {e}")
                                continue
                            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', action='store', type=str, help='raw_dir_path', required=True)
    parser.add_argument('--save_traj_dir_path', action='store', type=str, help='traj_dir_path', required=True)
    # base_path = './bread_cook_dataset/raw'
    # save_hdf5_dir_path = './bread_cook_dataset/traj'
    args = vars(parser.parse_args())
    raw2traj = Raw2TRAJ()
    raw2traj.execute(args['base_path'], args['save_traj_dir_path'])
