import h5py
import os
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('/nfsroot/DATA/IL_Research/wk/huggingface_model/distilbert-base-uncased')
model = AutoModel.from_pretrained("/nfsroot/DATA/IL_Research/wk/huggingface_model/distilbert-base-uncased", torch_dtype=torch.float16)
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

# file_path = '/data/team/wk/datasets/real_franka/act_datasets'
# file_path = '/media/jz08/HDD/wk/datasets/real_franka/act_datasets'
# file_path = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/real_franka_1/h5_data_example'
# file_path = '/nfsroot/DATA/IL_Research/datasets/real_franka_1/h5_data_example'

# 1. 把倒了的马克杯翻转回正 Turn the overturned mug upright.
# 2. 抓取面包，放到盘子上 Pick up the bread and place it on the plate.
# 3. 从盘子上把面包放回桌面 Move the bread from the plate back to the table.
# 4. 把蓝色方块堆在粉色方块上 Stack the blue block on top of the pink block.
# 5. 把蓝色方块从粉方块上取下 Remove the blue block from the pink block.
# 6. 把草莓放碗里 Place the strawberries in the bowl.
# 7. 把草莓从碗里拿出来，放碗右边 Take the strawberries out of the bowl and place them to the right of the bowl.
# 8. 把蓝色的盘子取出放桌面中心下方 Remove the blue plate and place it below the center of the table.
# 9. 把蓝色的盘子放回盘子架空槽中 Return the blue plate to the empty slot in the plate rack.
# 10. 把抽屉拉开 Pull the drawer open.
# 11. 把抽屉关上 Close the drawer.
# 12. 打开锅盖，放到桌面中心下方 Open the pot lid and place it below the center of the table.
# 13. 盖上锅盖 Close the pot lid.
# 14. 把马克笔放进笔筒 Put the marker into the pen holder.
# 15. 把马克笔从笔筒取出，放到笔筒左侧 Take the marker out of the pen holder and place it to the left of the pen holder.
# 16. 按开垃圾桶 Press the trash can to open it.
# 17. 从后往下压，关闭垃圾桶 Close the trash can by pressing down from the back.
# 18. 从左侧面拉开上层抽屉 Pull open the upper drawer from the left side.
# 19. 从右侧关闭上层抽屉 Close the upper drawer from the right side.
# 20. 按台灯开关，打开台灯 Press the desk lamp switch to turn on the lamp.
# 21. 按台灯开关，关闭台灯 Press the desk lamp switch to turn off the lamp.
# 22. 把梨子放粉色碗里 Place the pear in the pink bowl.
# 23. 把梨子从粉色碗里拿出来，放碗右边 Remove the pear from the pink bowl and place it to the right of the bowl.
# 24. 把梨子放粉色碗里 Place the pear in the pink bowl.
# 25. 把梨子从粉色碗里拿出来，放碗右边 Remove the pear from the pink bowl and place it to the right of the bowl.

# tasks = {
#     '241012_upright_blue_cup_1': 'Turn the overturned mug upright.', 
#     '241015_pick_bread_plate_1': "Pick up the bread and place it on the plate.", 
# }

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
file_path = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_ur_1rgb'

tasks = {
    'pick_up_plastic_bottle': 'Pick up plastic bottle.', 
    'pick_up_plastic_bottle_1': 'Pick up plastic bottle.', 
}


for t, lang in tasks.items():
    print("="*40)
    raw_lang = lang
    print(f'raw_lang: {raw_lang}')
    encoded_input = tokenizer(raw_lang, return_tensors='pt').to('cuda')
    outputs = model(**encoded_input)
    encoded_lang = outputs.last_hidden_state.sum(1).squeeze().unsqueeze(0)
    # [1, 768]
    print(f'encoded_lang size: {encoded_lang.size()}')
    # cfg['lang_intrs_distilbert'] = encoded_lang

    t_p = os.path.join(file_path, t, 'train')
    episodes = os.listdir(t_p)
    for ep in tqdm(episodes):
        ep_p = os.path.join(t_p, ep, 'data', 'trajectory.hdf5')
        print(f"path: {ep_p}")
        raw_lang = lang
        add_attributes_to_hdf5(ep_p, raw_lang, encoded_lang)
    
    t_p = os.path.join(file_path, t, 'val')
    episodes = os.listdir(t_p)
    for ep in tqdm(episodes):
        ep_p = os.path.join(t_p, ep, 'data', 'trajectory.hdf5')
        print(f"path: {ep_p}")
        raw_lang = lang
        add_attributes_to_hdf5(ep_p, raw_lang, encoded_lang)

