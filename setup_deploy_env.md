# To deploy model on Franka

1. switch to the small server, stop the camera node and server node in the small server

2. change the USB cable of the camera, robot control hub from small server(used for collecting data) to deployment machine

3. start the camera node on the deployment machine
    cd /home/ps/Dev/inrocs
    conda activate wk_act
    python3 node/launch_camera_nodes.py

    it will show:
        wait cameras up: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.00s/it]
        Found 3 cameras: ['231122071307', '231122072580', '233522074731']
        Launching camera 231122071307 on port 5000
        Launching camera 231122072580 on port 5001
        Launching camera 233522074731 on port 5002
        Camera 233522074731 intrinsics saved to /home/ps/Dev/inrocs/calibration/233522074731_intrinsics.json
        Camera Sever Binding to tcp://127.0.0.1:5002, Camera: RealSenseCamera(device_id=233522074731)
        Starting camera server on port 5002
        Camera 231122072580 intrinsics saved to /home/ps/Dev/inrocs/calibration/231122072580_intrinsics.json
        Camera Sever Binding to tcp://127.0.0.1:5001, Camera: RealSenseCamera(device_id=231122072580)
        Starting camera server on port 5001
        Camera 231122071307 intrinsics saved to /home/ps/Dev/inrocs/calibration/231122071307_intrinsics.json
        Camera Sever Binding to tcp://127.0.0.1:5000, Camera: RealSenseCamera(device_id=231122071307)
        Starting camera server on port 5000
        Timeout in Camera Server, Camera: RealSenseCamera(device_id=233522074731)
        Timeout in Camera Server, Camera: RealSenseCamera(device_id=231122072580)
        Timeout in Camera Server, Camera: RealSenseCamera(device_id=231122071307)
        Timeout in Camera Server, Camera: RealSenseCamera(device_id=233522074731)
        Timeout in Camera Server, Camera: RealSenseCamera(device_id=231122072580)
        Timeout in Camera Server, Camera: RealSenseCamera(device_id=231122071307)

4. start the smalle server (used for connecting Franka robot)
    go to the setting in Ubuntu system. 
    In the network section, open the Wired connection 1 with mannual IPv4: 172.16.0.7:255.255.255.0:172.16.0.1
    ssh eai@172.16.0.5, password: eai
    cd /home/eai/Dev/droid
    conda activate polymetis-local
    python3 scripts/server/run_server.py


5. In the deployment machine
    conda activate wk_act
    cd /home/ps/wk/github/act_benchmark
    ./deploy_act_franka_3rgb.sh

    task information
    /media/ps/wk/benchmark_results/act/franka_3rgb_241021_insert_marker_1

    ##############
    conda activate wk_act
    cd /home/ps/Dev/inrocs
    Example:
        cd run
        python run_VLM_test.py

    For deploying your own model:
        git clone gitlab: act_frame in this folder, and updata the code.
        We have folder: /home/ps/Dev/inrocs/act_frame
        create your own deploy python program.
        then run
        cd act_frame
        python run_xxx_test.py 

    ####### Gemo_act
    cd /home/ps/Dev/inrocs/action_frame
    git checkout gemo_act
    ./run_franka_gemo_act_inference.sh
        
6. Stop the camera node in the deployment machine
    Stop the server node in the small machine
    change the USB cable of the camera, robot control hub to the small server(used for collecting data in the next day)





 

