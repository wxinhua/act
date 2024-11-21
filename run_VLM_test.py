import datetime
import os
import threading
import time
from pathlib import Path
from venv import logger

import sys
sys.path.append('/home/ps/Dev/inrocs/')
# import os
# cwd = os.getcwd()
# print(cwd)

import numpy as np
from tqdm.auto import tqdm
import tyro
from pynput import keyboard
import termcolor

from robot_env.franka_env import robot_env


preparing = True
stop = False
button_exit = False

def on_press(key):
    global preparing, stop, button_exit
    try:
        print(f'key {key}')
        if key == keyboard.Key.scroll_lock:
            preparing = False
        if key == keyboard.Key.pause:
            stop = True
        if key == keyboard.Key.esc:
            button_exit = True

    except AttributeError:
        pass


def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def main():
    # going to start position
    print("Going to start position")
    obs = robot_env.get_obs()

    logger.info("\nStart 🚀🚀🚀")
    ###
    listener_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
    listener_thread.start()
    ###

    for i in range(100):
        obs = robot_env.get_obs()
        print(f'obs {obs}')

        # obs = robot_env.step(action)

        if button_exit:
            speaker.speak("关闭程序。", sync=True)
            exit()


if __name__ == "__main__":
    main()