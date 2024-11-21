# Data_Script

## Installation
This code has been tested with python=3.8.12, CUDA12.1, Ubuntu22

### Configure the environment
1、Install torch、torchvision、torchaudio
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
2、Install others
```
pip install -r requirements.txt
```

3. Install robomimic for droid diffusion

```
https://github.com/ARISE-Initiative/robomimic/tree/diffusion-policy-mg

pip uninstall robomimic
cd somewhere && git clone git@github.com:ARISE-Initiative/robomimic.git
cd somewhere/robomimic && git checkout diffusion-policy-mg
pip install -e .
pip install diffusers==0.11.1

```

## Train 
### act
```
sh train.sh
```

