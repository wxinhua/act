## The data format collected from current real Franka is pkl

## Convert pkl to h5


`conda activate sl_act`
`cd ~/wk/gitlab/data_script`

Then modify the path in wk_pkl2h5_franka.sh.
Here is an example
    
    python3 pkl2h5_franka.py \
        --pkl_dir_path /media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/real_franka/pkl_data \
        --save_h5_path /media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/real_franka/h5_data \
        --task_name 240924_pick_bread_plate_1 \
        --h5_save_compress

## Visualiza the RGB and depth data from h5

`conda activate sl_act`
`cd ~/wk/gitlab/action_frame`

Then modify the path in check_h5_img.sh. 
It will load all RGB and depth images from h5 files, and save them to .jpg with the same folder structure.
Here is an example.

    source_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/real_franka/success/table_scene/240923_pick_bread_plate_1_h5' #'path/to/folder/A'
    dest_folder = '/media/wk/4852d46a-6164-41f4-bd60-f88410dc2041/wk_dir/datasets/real_franka/success/table_scene/240923_pick_bread_plate_1_image' #'path/to/folder/B'
    load_h5_and_save_images(source_folder, dest_folder)



