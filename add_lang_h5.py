import h5py
import os
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
import json

# tokenizer = AutoTokenizer.from_pretrained('/nfsroot/DATA/IL_Research/wk/huggingface_model/distilbert-base-uncased')
# model = AutoModel.from_pretrained("/nfsroot/DATA/IL_Research/wk/huggingface_model/distilbert-base-uncased", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained('/media/users/wk/huggingface_model/distilbert-base-uncased')
model = AutoModel.from_pretrained("/media/users/wk/huggingface_model/distilbert-base-uncased", torch_dtype=torch.float16)
model.to('cuda')

# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# model = AutoModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16)
# model.to('cuda')

def add_attributes_to_hdf5(file_path, raw_lang, encoded_lang):
    # Open the HDF5 file in read/write mode
    with h5py.File(file_path, 'a') as hdf5_file:
        # Add attributes to the root of the HDF5 file
       # for key, value in attributes.items():
    #    hdf5_file.create_dataset("language_raw", data=[raw_lang])
        encoded_lang = encoded_lang.cpu().detach().numpy()
        key = 'language_distilbert'
        # print(f"h5 keys: {hdf5_file.keys()}")
        if key not in hdf5_file.keys():
            hdf5_file.create_dataset("language_distilbert", data=[encoded_lang])
        key = 'language_raw'
        # print(f"h5 keys: {hdf5_file.keys()}")
        if key not in hdf5_file.keys():
            hdf5_file.create_dataset("language_raw", data=[raw_lang])
        print(f"==============")
        print(f"h5 keys: {hdf5_file.keys()}")

# Example usage

# ############## Franka_1rgb ################
# file_path = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_franka_1rgb'
# file_path = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_franka_1rgb'

# tasks = {
#     'bread_on_table': 'Pick bread on table.', 
#     'bread_on_table_1': 'Pick bread on table.',
# }

# ############## Franka_3rgb ################
# file_path = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_franka_3rgb'
# file_path = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_franka_3rgb'

# tasks = {
#     'pick_plate_from_plate_rack': 'Pick plate from plate rack.', 
#     'pick_plate_from_plate_rack_1': 'Pick plate from plate rack.', 
# }

# ############## Songling_3rgb ################
# file_path = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_songling_3rgb'
# file_path = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_songling_3rgb'

# tasks = {
#     '15_steamegg_2': 'Steam egg.', 
#     '15_steamegg_2_1': 'Steam egg.',
# }

# ############## Tiangong_1rgb ################
# file_path = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_tiangong_1rgb'
# file_path = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_tiangong_1rgb'

# tasks = {
#     'push_break_pick_shelf_insert_machine_press_switch_place_plate': 'Push break pick shelf insert machine press switch place plate.', 
#     'push_break_pick_shelf_insert_machine_press_switch_place_plate_1': 'Push break pick shelf insert machine press switch place plate.', 
# }

############## UR_1rgb ################
# file_path = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_ur_1rgb'
# file_path = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_ur_1rgb'

# tasks = {
#     'pick_up_plastic_bottle': 'Pick up plastic bottle.', 
#     'pick_up_plastic_bottle_1': 'Pick up plastic bottle.', 
# }

dataset_path = ['/media/data/h5_franka_3rgb', 
                '/media/data/h5_franka_1rgb', 
                '/media/data/h5_simulation', 
                '/media/data/h5_tiangong_1rgb', 
                '/media/data/h5_ur_1rgb',
                '/media/data/h5_songling_3rgb']

json_path = '/root/wk/git-repo/act_benchmark/benchmark_1_0_instr.json'

with open(json_path, 'r') as file:
    data = json.load(file)

print(len(data.keys()))




# for t, lang in tasks.items():
#     print("="*40)
#     raw_lang = lang
#     print(f'raw_lang: {raw_lang}')
#     encoded_input = tokenizer(raw_lang, return_tensors='pt').to('cuda')
#     outputs = model(**encoded_input)
#     encoded_lang = outputs.last_hidden_state.sum(1).squeeze().unsqueeze(0)
#     # [1, 768]
#     print(f'encoded_lang size: {encoded_lang.size()}')
#     # cfg['lang_intrs_distilbert'] = encoded_lang

#     t_p = os.path.join(file_path, t, 'train')
#     episodes = os.listdir(t_p)
#     for ep in tqdm(episodes):
#         ep_p = os.path.join(t_p, ep, 'data', 'trajectory.hdf5')
#         print(f"path: {ep_p}")
#         raw_lang = lang
#         add_attributes_to_hdf5(ep_p, raw_lang, encoded_lang)
    
#     t_p = os.path.join(file_path, t, 'val')
#     episodes = os.listdir(t_p)
#     for ep in tqdm(episodes):
#         ep_p = os.path.join(t_p, ep, 'data', 'trajectory.hdf5')
#         print(f"path: {ep_p}")
#         raw_lang = lang
#         add_attributes_to_hdf5(ep_p, raw_lang, encoded_lang)

