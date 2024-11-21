### Task parameters

# DATA_DIR = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/ai_station_data'

# DATA_DIR = '/nfsroot/DATA/IL_Research/datasets/real_franka_1/h5_data_example'
# DATA_DIR = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/real_franka_1/h5_data_example'

####### local
# Franka_1rgb_DATA_DIR = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_franka_1rgb'
# Franka_3rgb_DATA_DIR = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_franka_3rgb'
# Songling_3rgb_DATA_DIR = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_songling_3rgb'
# Tiangong_1rgb_DATA_DIR = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_tiangong_1rgb'
# UR_1rgb_DATA_DIR = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/benchmark_data_1/h5_ur_1rgb'

########## AI station
# Franka_1rgb_DATA_DIR = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_franka_1rgb'
# Franka_3rgb_DATA_DIR = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_franka_3rgb'
# Songling_3rgb_DATA_DIR = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_songling_3rgb'
# Tiangong_1rgb_DATA_DIR = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_tiangong_1rgb'
# UR_1rgb_DATA_DIR = '/nfsroot/DATA/IL_Research/datasets/benchmark_data_1/h5_ur_1rgb'

######## baidu test
Franka_1rgb_DATA_DIR = '/media/users/wk/benchmark_data_1/h5_franka_1rgb'
Franka_3rgb_DATA_DIR = '/media/users/wk/benchmark_data_1/h5_franka_3rgb'
Songling_3rgb_DATA_DIR = '/media/users/wk/benchmark_data_1/h5_songling_3rgb'
Tiangong_1rgb_DATA_DIR = '/media/users/wk/benchmark_data_1/h5_tiangong_1rgb'
UR_1rgb_DATA_DIR = '/media/users/wk/benchmark_data_1/h5_ur_1rgb'

# ######## baidu
# Franka_1rgb_DATA_DIR = '/media/datasets/benchmark_data_1/h5_franka_1rgb'
# Franka_3rgb_DATA_DIR = '/media/datasets/benchmark_data_1/h5_franka_3rgb'
# Songling_3rgb_DATA_DIR = '/media/datasets/benchmark_data_1/h5_songling_3rgb'
# Tiangong_1rgb_DATA_DIR = '/media/datasets/benchmark_data_1/h5_tiangong_1rgb'
# UR_1rgb_DATA_DIR = '/media/datasets/benchmark_data_1/h5_ur_1rgb'


# train_ratio > 1: random train_ratio num of traj
# train_ratio -> [0, 1]: ratio of traj

Franka_1rgb_TASK_CONFIGS = {
    'bread_on_table':{
        'dataset_dir': Franka_1rgb_DATA_DIR + '/bread_on_table',
    },
    'multi_task_2':{
        'dataset_dir': [
            Franka_1rgb_DATA_DIR + '/bread_on_table',
            Franka_1rgb_DATA_DIR + '/bread_on_table_1',
        ],
        'sample_weights': [5, 5],
    },
}

Franka_3rgb_TASK_CONFIGS = {
    'pick_plate_from_plate_rack':{
        'dataset_dir': Franka_3rgb_DATA_DIR + '/pick_plate_from_plate_rack',
        'train_ratio': [100]
    },
    'multi_task_2':{
        'dataset_dir': [
            Franka_3rgb_DATA_DIR + '/pick_plate_from_plate_rack',
            Franka_3rgb_DATA_DIR + '/pick_plate_from_plate_rack_1',
        ],
        'sample_weights': [5, 5],
    },
}

Songling_3rgb_TASK_CONFIGS = {
    '15_steamegg_2':{
        'dataset_dir': Songling_3rgb_DATA_DIR + '/15_steamegg_2',
    },
    'multi_task_2':{
        'dataset_dir': [
            Songling_3rgb_DATA_DIR + '/15_steamegg_2',
            Songling_3rgb_DATA_DIR + '/15_steamegg_2_1',
        ],
        'sample_weights': [5, 5],
    },
}


Tiangong_1rgb_TASK_CONFIGS = {
    '15_steamegg_2':{
        'dataset_dir': Tiangong_1rgb_DATA_DIR + '/push_break_pick_shelf_insert_machine_press_switch_place_plate',
    },
    'multi_task_2':{
        'dataset_dir': [
            Tiangong_1rgb_DATA_DIR + '/push_break_pick_shelf_insert_machine_press_switch_place_plate',
            Tiangong_1rgb_DATA_DIR + '/push_break_pick_shelf_insert_machine_press_switch_place_plate_1',
        ],
        'sample_weights': [5, 5],
    },
}

UR_1rgb_TASK_CONFIGS = {
    '15_steamegg_2':{
        'dataset_dir': UR_1rgb_DATA_DIR + '/pick_up_plastic_bottle',
    },
    'multi_task_2':{
        'dataset_dir': [
            UR_1rgb_DATA_DIR + '/pick_up_plastic_bottle',
            UR_1rgb_DATA_DIR + '/pick_up_plastic_bottle_1',
        ],
        'sample_weights': [5, 5],
    },
}










