import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
from agent.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer

# from extractor.maskclip import MaskCLIPFeaturizer
class ACTPolicy(nn.Module):
    def __init__(self, args_override, rank=None):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.use_vq = args_override['use_vq']
        print(f'KL Weight {self.kl_weight}')
        self.use_lang = args_override['use_lang']
        if "clip" in args_override['backbone']:
            self.norm_rgb = False
        else:
            self.norm_rgb = True

    def __call__(self, qpos, image, depth_image, actions=None, is_pad=None, vq_sample=None, language_distilbert=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        depth_normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        # language_distilbert: [1, 768]
        if self.norm_rgb:
            image = normalize(image)
        if depth_image is not None:
            depth_image = depth_normalize(depth_image)

        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            # a_hat, (mu, logvar) = self.model(image, depth_image, qpos, actions, is_pad)
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, depth_image, env_state, actions, is_pad, vq_sample, lang_embed=language_distilbert)

            if self.use_vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            if self.use_vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')
            # a_hat: torch.Size([b, 50, 8])
            # print(f"a_hat: {a_hat.size()}")
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:  # inference time
            # a_hat, (_, _) = self.model(image, depth_image, qpos)  # no action, sample from prior
            a_hat, _, (_, _), _, _ = self.model(qpos, image, depth_image, env_state, vq_sample=vq_sample, lang_embed=language_distilbert) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)

        return binaries

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
