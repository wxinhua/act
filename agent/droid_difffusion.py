import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
from agent.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
#import IPython
#e = IPython.embed

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import warnings

import logging

# Configure logging level for GLFW if it uses logging
logging.getLogger("glfw").setLevel(logging.ERROR)  # Ignores warnings and below

# Suppress specific future warnings from a library
warnings.filterwarnings('ignore', category=FutureWarning)

import random
import torchvision.transforms.functional as TVF
from torchvision.transforms import Lambda, Compose

class ColorRandomizer(nn.Module):
    """
    Randomly sample color jitter at input, and then average across color jtters at output.
    """
    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        num_samples=1,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            brightness (None or float or 2-tuple): How much to jitter brightness. brightness_factor is chosen uniformly
                from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.
            contrast (None or float or 2-tuple): How much to jitter contrast. contrast_factor is chosen uniformly
                from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.
            saturation (None or float or 2-tuple): How much to jitter saturation. saturation_factor is chosen uniformly
                from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.
            hue (None or float or 2-tuple): How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or
                the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. To jitter hue, the pixel
                values of the input image has to be non-negative for conversion to HSV space; thus it does not work
                if you normalize your image to an interval with negative values, or use an interpolation that
                generates negative values before using this function.
            num_samples (int): number of random color jitters to take
        """
        super(ColorRandomizer, self).__init__()

        assert len(input_shape) == 3 # (C, H, W)

        self.input_shape = input_shape
        self.brightness = [max(0, 1 - brightness), 1 + brightness] if type(brightness) in {float, int} else brightness
        self.contrast = [max(0, 1 - contrast), 1 + contrast] if type(contrast) in {float, int} else contrast
        self.saturation = [max(0, 1 - saturation), 1 + saturation] if type(saturation) in {float, int} else saturation
        self.hue = [-hue, hue] if type(hue) in {float, int} else hue
        self.num_samples = num_samples

    @torch.jit.unused
    def get_transform(self):
        """
        Get a randomized transform to be applied on image.

        Implementation taken directly from:

        https://github.com/pytorch/vision/blob/2f40a483d73018ae6e1488a484c5927f2b309969/torchvision/transforms/transforms.py#L1053-L1085

        Returns:
            Transform: Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(Lambda(lambda img: TVF.adjust_brightness(img, brightness_factor)))

        if self.contrast is not None:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(Lambda(lambda img: TVF.adjust_contrast(img, contrast_factor)))

        if self.saturation is not None:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(Lambda(lambda img: TVF.adjust_saturation(img, saturation_factor)))

        if self.hue is not None:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(Lambda(lambda img: TVF.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def get_batch_transform(self, N):
        """
        Generates a batch transform, where each set of sample(s) along the batch (first) dimension will have the same
        @N unique ColorJitter transforms applied.

        Args:
            N (int): Number of ColorJitter transforms to apply per set of sample(s) along the batch (first) dimension

        Returns:
            Lambda: Aggregated transform which will autoamtically apply a different ColorJitter transforms to
                each sub-set of samples along batch dimension, assumed to be the FIRST dimension in the inputted tensor
                Note: This function will MULTIPLY the first dimension by N
        """
        return Lambda(lambda x: torch.stack([self.get_transform()(x_) for x_ in x for _ in range(N)]))

    def output_shape_in(self, input_shape=None):
        # outputs are same shape as inputs
        return list(input_shape)

    def output_shape_out(self, input_shape=None):
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def _forward_in(self, inputs):
        """
        Samples N random color jitters for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions

        # Make sure shape is exactly 4
        if len(inputs.shape) == 3:
            inputs = torch.unsqueeze(inputs, dim=0)

        # TODO: Make more efficient other than implicit for-loop?
        # Create lambda to aggregate all color randomizings at once
        transform = self.get_batch_transform(N=self.num_samples)

        return transform(inputs)

    # def _forward_out(self, inputs):
    #     """
    #     Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
    #     to result in shape [B, ...] to make sure the network output is consistent with
    #     what would have happened if there were no randomization.
    #     """
    #     batch_size = (inputs.shape[0] // self.num_samples)
    #     out = TensorUtils.reshape_dimensions(inputs, begin_axis=0, end_axis=0, target_dims=(batch_size, self.num_samples))
    #     return out.mean(dim=1)

    # def _visualize(self, pre_random_input, randomized_input, num_samples_to_visualize=2):
    #     batch_size = pre_random_input.shape[0]
    #     random_sample_inds = torch.randint(0, batch_size, size=(num_samples_to_visualize,))
    #     pre_random_input_np = TensorUtils.to_numpy(pre_random_input)[random_sample_inds]
    #     randomized_input = TensorUtils.reshape_dimensions(
    #         randomized_input,
    #         begin_axis=0,
    #         end_axis=0,
    #         target_dims=(batch_size, self.num_samples)
    #     )  # [B * N, ...] -> [B, N, ...]
    #     randomized_input_np = TensorUtils.to_numpy(randomized_input[random_sample_inds])

    #     pre_random_input_np = pre_random_input_np.transpose((0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
    #     randomized_input_np = randomized_input_np.transpose((0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]

    #     visualize_image_randomizer(
    #         pre_random_input_np,
    #         randomized_input_np,
    #         randomizer_name='{}'.format(str(self.__class__.__name__))
    #     )

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + f"(input_shape={self.input_shape}, brightness={self.brightness}, contrast={self.contrast}, " \
                       f"saturation={self.saturation}, hue={self.hue}, num_samples={self.num_samples})"
        return msg


class DroidDiffusionPolicy(nn.Module):
    def __init__(self, args_override, rank=None):
        from collections import OrderedDict
        from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
        from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D

        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.training_utils import EMAModel

        super().__init__()

        self.camera_names = args_override['camera_names']
        self.use_depth_image = False #args_override['use_depth_image']

        self.observation_horizon = args_override['observation_horizon'] ### TODO TODO TODO DO THIS
        self.prediction_horizon = args_override['prediction_horizon'] # chunk size
        #self.num_inference_timesteps = args_override['num_inference_timesteps']

        self.num_inference_timesteps = 10 #30 #15 #30 #10 #15 #30 #50 #100 #16 #args_override['num_inference_timesteps']

        self.ema_power = args_override['ema_power'] # 0.75
        self.lr = args_override['lr'] # 1e-4
        self.weight_decay = args_override['weight_decay'] #0.0 # 1e-6

        self.pool_class = args_override['pool_class'] # null
        # self.img_flatten = args_override['img_flatten'] # True
        # for pool spatial softmax num_kp
        self.stsm_num_kp = args_override['stsm_num_kp'] #512

        self.img_fea_dim = args_override['img_fea_dim'] #512
        self.ac_dim = args_override['action_dim'] # 14 + 2
        self.state_dim = args_override['state_dim'] # 14 + 2

        self.num_queries = args_override['num_queries']
        self.num_noise_samples = args_override['num_noise_samples'] # 8
        self.use_color_rand = args_override['use_color_rand']
        if self.use_color_rand:
            self.color_randomizer = ColorRandomizer(input_shape=[3, 480, 640])
        else:
            self.color_randomizer = None

        #### init backbone ####
        from agent.detr.models.backbone import build_backbone
        import argparse
        from agent.detr.main import get_args_parser
        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        # diff_args = parser.parse_args()
        diff_args, unknown = parser.parse_known_args()
        for k, v in args_override.items():
            setattr(diff_args, k, v)

        self.diff_args = diff_args
        if "clip" in args_override['backbone']:
            self.norm = False
        else:
            self.norm = True
        self.use_film = "film" in args_override['backbone']
        self.use_lang = args_override['use_lang']
        self.backbone_name = args_override['backbone']
        
        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbone = build_backbone(diff_args)
            backbones.append(backbone)

            if self.pool_class == 'SpatialSoftmax':
                if self.backbone_name == 'resnet50':
                    # [2048, 9, 15] for 270 480 droid, resnet 50
                    # [512, 15, 20] for 480 640 droid, resnet 50
                    input_shape = [2048, 15, 20]
                if self.backbone_name == 'resnet34':
                    # [512, 9, 15] for 270 480 droid, resnet 34
                    # [512, 15, 20] for 480 640 droid, resnet 34
                    input_shape = [512, 15, 20]
                if self.backbone_name == 'resnet18':
                    # [512, 15, 20] for 480 640 droid, resnet 34
                    input_shape = [512, 15, 20]
                if self.backbone_name == 'efficientnet_b0film':
                    # [1280, 7, 7] for 480 640 droid, efficientnet_b0film
                    input_shape = [1280, 7, 7]
                if self.backbone_name == 'efficientnet_b3film':
                    # [1536, 10, 10] for 480 640 droid, efficientnet_b3film
                    input_shape = [1536, 10, 10]
                if self.backbone_name == 'efficientnet_b5film':
                    # [2048, 15, 15] for 480 640 droid, efficientnet_b5film
                    input_shape = [2048, 15, 15]
                pools.append(
                    nn.Sequential(
                        SpatialSoftmax(**{'input_shape': input_shape, 'num_kp': self.stsm_num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}),
                        nn.Flatten(start_dim=1, end_dim=-1)
                        )
                )
                linears.append(
                    nn.Sequential(
                        nn.Linear(int(np.prod([self.stsm_num_kp, 2])), self.stsm_num_kp),
                        nn.ReLU(),
                        nn.Linear(self.stsm_num_kp, self.img_fea_dim)
                    )
                )
            elif self.pool_class == 'null':
                pools.append(
                    nn.Sequential(
                        nn.Conv2d(backbones[0].num_channels, 8, kernel_size=1),
                        nn.Flatten(start_dim=1, end_dim=-1)
                    )
                )
                if self.backbone_name == 'resnet50':
                    image_flatten_dim = 8*9*15
                    # image_flatten_dim = 8*6*10
                elif self.backbone_name == 'resnet18':
                    image_flatten_dim = 8*9*15
                linears.append(torch.nn.Linear(image_flatten_dim, self.img_fea_dim))  

        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)

        self.obs_input_dim = self.img_fea_dim * len(self.camera_names) + self.state_dim

        if self.use_lang:
            self.obs_input_dim += 768 

        self.cond_obs_dim = args_override['cond_obs_dim'] # 512
        self.combine = nn.Sequential(
                nn.Linear(self.obs_input_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, self.cond_obs_dim)
            )

        if self.use_film:
            backbones = replace_bn_with_gn(backbones, features_per_group=8) # TODO
        else:
            backbones = replace_bn_with_gn(backbones) # TODO

        print(f'Build noise_pred_net!')
        print(f"noise_pred_net cond obs dim: {self.cond_obs_dim}")

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.cond_obs_dim
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'combine': self.combine,
                'noise_pred_net': noise_pred_net
            })
        })
    
        #nets = nets.float().cuda()
        # ENABLE_EMA = False
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets.cuda(), power=self.ema_power)#.to(device=rank)
            #ema  = ema.to(rank)
            #ema  = DDP(ema , device_ids=[rank])
        else:
            ema = None
        self.nets = nets
        self.ema = ema
        #self.ema = ema

        # self.noise_scheduler = DDPMScheduler(
        #     num_train_timesteps=100,
        #     beta_schedule='squaredcos_cap_v2',
        #     clip_sample=True,
        #     prediction_type='epsilon'
        # )
        
        # "ddim": {
        #     "enabled": true,
        #     "num_train_timesteps": 100,
        #     "num_inference_timesteps": 10,
        #     "beta_schedule": "squaredcos_cap_v2",
        #     "clip_sample": true,
        #     "set_alpha_to_one": true,
        #     "steps_offset": 0,
        #     "prediction_type": "epsilon"
        # },
        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
           num_train_timesteps=100,
           beta_schedule='squaredcos_cap_v2',
           clip_sample=True,
           set_alpha_to_one=True,
           steps_offset=0,
           prediction_type='epsilon'
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        print("droid_diffusion number of parameters: %.2fM" % (n_parameters/1e6,))


    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        param_dicts = [
        {"params": [p for n, p in self.nets.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in self.nets.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": self.diff_args.lr_backbone,
        },
    ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                    weight_decay=self.weight_decay)

        return optimizer


    def __call__(self, qpos, image, depth_image, actions=None, is_pad=None, language_distilbert=None):
        lang_embed = language_distilbert
        B = qpos.shape[0]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # language_distilbert: [1, 768]
        if self.norm:
            image = normalize(image)
        if actions is not None: # training time
            nets = self.nets

            actions = actions[:, :self.num_queries]
            is_pad = is_pad[:, :self.num_queries]
            # actions size: torch.Size([64, 50, 10])
            # print(f"actions size: {actions.size()}")

            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                if self.use_film and lang_embed is not None:
                    cur_img = image[:, cam_id]
                    # print(f"1. cur_img: {cur_img.size()}")
                    if self.color_randomizer is not None:
                        cur_img = self.color_randomizer._forward_in(cur_img)
                        # print(f"2. cur_img: {cur_img.size()}")
                    features, pos = self.nets['policy']['backbones'][cam_id](cur_img, lang_embed)
                else:
                    cur_img = image[:, cam_id]
                    # cur_img: torch.Size([64, 3, 270, 480])
                    # print(f"3. cur_img: {cur_img.size()}")
                    if self.color_randomizer is not None:
                        cur_img = self.color_randomizer._forward_in(cur_img)
                        # print(f"4. cur_img: {cur_img.size()}")
                    features, pos = self.nets['policy']['backbones'][cam_id](cur_img)
                
                features = features[0] # take the last layer feature
                pos = pos[0]
                # resnet18: cam_id: 0, features size: torch.Size([64, 512, 9, 15])
                # resnet18: cam_id: 0, pos size: torch.Size([1, 256, 9, 15])
                # print(f"cam_id: {cam_id}, features size: {features.size()}")
                # print(f"cam_id: {cam_id}, pos size: {pos.size()}")
                # cam_features = self.input_proj(features)
                cam_features = features
                # cam_features size: torch.Size([24, 512, 15, 20])
                # print(f"cam_features size: {cam_features.size()}")

                pool_features = nets['policy']['pools'][cam_id](cam_features)
                # resnet18: 1. pool_features: torch.Size([64, 512, 2])
                # print(f"1. pool_features: {pool_features.size()}")
                out_features = nets['policy']['linears'][cam_id](pool_features)
                # print(f"out_features: {out_features.size()}")

                all_cam_features.append(out_features)
                all_cam_pos.append(pos)
            
            # qpos size: torch.Size([64, 7])
            # print(f"qpos size: {qpos.size()}")
            # 1. obs_cond size: torch.Size([64, 1031])
            # 2. obs_cond size: torch.Size([64, 512])
            if self.use_lang and lang_embed is not None:
                obs_cond = torch.cat(all_cam_features + [qpos] + [lang_embed], dim=1)
            else:
                 obs_cond = torch.cat(all_cam_features + [qpos], dim=1)
            # print(f"1. obs_cond size: {obs_cond.size()}")
            obs_cond = self.combine(obs_cond)
            # print(f"2. obs_cond size: {obs_cond.size()}")

            # sample noise to add to actions
            # noise = torch.randn(actions.shape, device=obs_cond.device)
            noise = torch.randn([self.num_noise_samples] + list(actions.shape), device=obs_cond.device)
            # print(f"noise size: {noise.size()}")
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=obs_cond.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            #print("before timesteps: {}, noise: {}, obs_cond: {}".format(
                #timesteps.device, noise.device, obs_cond.device))
            timesteps, noise = timesteps.to(obs_cond.device), noise.to(obs_cond.device)
            #print("before timesteps: {}, noise: {}, obs_cond: {}".format(
                #timesteps.device, noise.device, obs_cond.device))
            # noisy_actions = self.noise_scheduler.add_noise(
            #     # actions, noise, timesteps)
            noisy_actions = torch.cat([self.noise_scheduler.add_noise(
                            actions, noise[i], timesteps)
                            for i in range(len(noise))], dim=0)
            
            obs_cond = obs_cond.repeat(self.num_noise_samples, 1)
            timesteps = timesteps.repeat(self.num_noise_samples)
            is_pad = is_pad.repeat(self.num_noise_samples, 1)

            # noisy_actions size: torch.Size([512, 50, 10])
            # timesteps size: torch.Size([512])
            # obs_cond size: torch.Size([512, 1031])
            # print(f"noisy_actions size: {noisy_actions.size()}")
            # print(f"timesteps size: {timesteps.size()}")
            # print(f"obs_cond size: {obs_cond.size()}")
            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
            # noise_pred size: torch.Size([512, 60, 10])
            # noise size: torch.Size([512, 60, 10])
            # print(f"noise_pred size: {noise_pred.size()}")
            # print(f"noise size: {noise.size()}")
            
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            # is_pad size: torch.Size([64, 60])
            # all_l2 size: torch.Size([512, 60, 10])
            # print(f"is_pad size: {is_pad.size()}")
            # print(f"all_l2 size: {all_l2.size()}")
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss

            if self.training and self.ema is not None:
                self.ema.step(nets)
                #torch.distributed.barrier()
            return loss_dict
        else: # inference time
            To = self.observation_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            
            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model
            
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                if self.use_film and lang_embed is not None:
                    features, pos = self.nets['policy']['backbones'][cam_id](image[:, cam_id], lang_embed)
                else:
                    features, pos = self.nets['policy']['backbones'][cam_id](image[:, cam_id])
                
                features = features[0] # take the last layer feature
                pos = pos[0]
                # cam_features = self.input_proj(features)
                cam_features = features
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                # pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)

                all_cam_features.append(out_features)
                all_cam_pos.append(pos)

            if self.use_lang and lang_embed is not None:
                obs_cond = torch.cat(all_cam_features + [qpos] + [lang_embed], dim=1)
            else:
                obs_cond = torch.cat(all_cam_features + [qpos], dim=1)
            obs_cond = self.combine(obs_cond)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action
            
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def get_latent_out_fea(self, qpos, image, actions=None, is_pad=None, language_distilbert=None):
        lang_embed = language_distilbert
        B = qpos.shape[0]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # language_distilbert: [1, 768]
        if self.norm:
            image = normalize(image)

        To = self.observation_horizon
        Tp = self.prediction_horizon
        action_dim = self.ac_dim
        
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        all_cam_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            if self.use_film and lang_embed is not None:
                features, pos = self.nets['policy']['backbones'][cam_id](image[:, cam_id], lang_embed)
            else:
                features, pos = self.nets['policy']['backbones'][cam_id](image[:, cam_id])
            
            features = features[0] # take the last layer feature
            pos = pos[0]
            # cam_features = self.input_proj(features)
            cam_features = features
            pool_features = nets['policy']['pools'][cam_id](cam_features)
            # pool_features = torch.flatten(pool_features, start_dim=1)
            out_features = nets['policy']['linears'][cam_id](pool_features)

            all_cam_features.append(out_features)
            all_cam_pos.append(pos)

        if self.use_lang and lang_embed is not None:
            obs_cond = torch.cat(all_cam_features + [qpos] + [lang_embed], dim=1)
        else:
            obs_cond = torch.cat(all_cam_features + [qpos], dim=1)
        obs_cond = self.combine(obs_cond)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=obs_cond.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            latent_fea, noise_pred = nets['policy']['noise_pred_net'](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond,
                output_latent=True
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        
        return latent_fea, obs_cond, noise_pred, naction

        

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if self.ema!=None:
            if model_dict.get("ema", None) is not None:
                print('Loaded EMA')
                status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
                status = [status, status_ema]
        return status

