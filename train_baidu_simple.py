import sys
import os

# example: python ./train_baidu_simple.py 1～68

cmds = [
    "bash ./train_baidu.sh --task_name place_in_bread_on_plate_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name place_in_bread_on_plate_2 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name pick_plate_from_plate_rack --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name pick_up_strawberry_in_bowl --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name open_cap_lid --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name stick_target_blue_on_the_pink_obejct --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name place_plate_in_plate_rack --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name close_cap_lid --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name slide_close_drawer_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name slide_close_drawer_1_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name slide_open_drawer --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name place_in_bread_on_table_2 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name piled_on_stack_blue_block_on_pink_block --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name pick_up_strawberry_from_bowl --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name blue_cub_on_pink --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name place_in_bread_in_plate --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name place_in_bread_on_table --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name place_in_block_in_plate_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name open_cap_trash_can_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name place_in_pick_up_and_throw_away_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241021_close_trash_bin_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241021_insert_marker_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241021_open_trash_bin_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241021_remove_marker_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241022_lamp_off_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241022_lamp_on_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241022_side_pull_close_drawer_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241022_side_pull_open_drawer_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241023_pick_pear_from_bowl_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241023_pick_pear_from_bowl_2 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241023_place_pear_in_bowl_1 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 241023_place_pear_in_bowl_2 --exp_type franka_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 28_packtable_2 --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 37_putegg --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 22_takepotato --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 38_putcorn --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 36_putpepper --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 30_takepumpkin --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 26_cleanplate --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 27_carrotgreenplate --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 9_appleyellowplate_2 --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 15_steamegg_2 --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 10_packplate_2 --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 8_appleblueplate_2 --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 35_putcarrot --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 13_packbowl --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 31_unpackbowl --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 7_applegreenplate_2 --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 29_steampumpkin --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 21_takeegg --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 11_brushcup_2 --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 16_steampotato_2 --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name 1_potatooven --exp_type songling_3rgb --day 241124",
    "bash ./train_baidu.sh --task_name open_top_white_drawer --exp_type ur_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name close_top_white_drawer --exp_type ur_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name open_trash_can --exp_type ur_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name close_trash_can --exp_type ur_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name pick_up_round_bread --exp_type ur_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name pick_up_long_bread --exp_type ur_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name place_button --exp_type tiangong_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name gear_place --exp_type tiangong_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name nut_place --exp_type tiangong_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name cylinder_pick_box_place_close --exp_type tiangong_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name tool_liftn_box_place --exp_type tiangong_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name plug_pullout_then_press --exp_type tiangong_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name wipe_panel --exp_type tiangong_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name throw_battery_twice --exp_type tiangong_1rgb --day 241124",
    "bash ./train_baidu.sh --task_name throw_battery --exp_type tiangong_1rgb --day 241124",
]

def execute_command(cmd_id):
    id = cmd_id - 1
    command = cmds[id]
    print(f"Executing command: {command}")
    os.system(command)

cmd_id = int(sys.argv[1])
execute_command(cmd_id)