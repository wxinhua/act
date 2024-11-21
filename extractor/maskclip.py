import numpy as np
import torch
import torch.nn as nn
import maskclip_onnx

class MaskCLIPFeaturizer(nn.Module):
    def __init__(self, clip_model_name):
        super().__init__()
        self.model, self.preprocess = maskclip_onnx.clip.load(clip_model_name)
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

        self.input_size = 224

        self.transform = T.Compose([
            T.Resize(self.input_size),
            # T.CenterCrop((input_size, input_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def forward(self, image_rgb):
        image_tensor = self.transform (image_rgb).unsqueeze(0).cuda()
        b, _, input_size_h, input_size_w = image_tensor.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(image_tensor).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)