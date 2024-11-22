def execute(self, file_path, camera_frame=None, control_frame=None, use_depth_image=False):
    image_dict = defaultdict(dict)
    control_dict = defaultdict(dict)
    base_dict = defaultdict(dict)
    with h5py.File(file_path, 'r') as root:
        is_sim = root.attrs['sim']
        is_compress = root.attrs['compress']

        lang_embed = None
        control_dict['language_distilbert'] = lang_embed

        # print(f"is_compress: {is_compress}")
        # select camera frame id
        for cam_name in self.camera_names:
            decode_rgb, decode_depth = self.decoder_image(
                camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                camera_depth_images=None)
            image_dict[self.camera_sensors[0]][cam_name] = decode_rgb


        # print('image_dict:',image_dict)
        for arm_name in self.arms:
            for control in self.robot_infor:
                if control_frame:
                    control_dict[arm_name][control] = root[arm_name][control][control_frame]
                else:
                    control_dict[arm_name][control] = root[arm_name][control][()]
        # print('infor_dict:',infor_dict)
    
    gc.collect()
    root.close()
    return image_dict, control_dict, base_dict, is_sim, is_compress