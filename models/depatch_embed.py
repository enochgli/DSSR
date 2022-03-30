import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

from timm.models import create_model
from timm.models.vision_transformer import _cfg, Block
from .ms_deform_attn_func import MSDeformAttnFunction
from .box_coder import *

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[32, 128], patch_size=4, patch_count=14, in_chans=3, embed_dim=768, with_norm=False):
        super().__init__()
        patch_stride = [img_size[0] // patch_count[0], img_size[1] // patch_count[1]]
        patch_pad_h = (patch_stride[0] * (patch_count[0] - 1) + patch_size - img_size[0]) // 2
        patch_pad_w = (patch_stride[1] * (patch_count[1] - 1) + patch_size - img_size[1]) // 2
        patch_pad = [patch_pad_h, patch_pad_w]
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = patch_count[0] * patch_count[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride[0], padding=patch_pad[0])
        if with_norm:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if hasattr(self, "norm"):
            x = self.norm(x)
        assert x.shape[1] == self.num_patches
        return x


class Simple_Patch(nn.Module):
    def __init__(self, offset_embed, img_size=224, patch_size=16, patch_pixel=16, patch_count=14,
                 in_chans=3, embed_dim=192, another_linear=False, use_GE=False, local_feature=False, with_norm=False):
        super().__init__()
        self.num_patches = patch_count[0] * patch_count[1]
        self.another_linear = another_linear
        if self.another_linear:
            self.patch_embed = PatchEmbed(img_size, 1 if local_feature else patch_size, patch_count, in_chans, embed_dim, with_norm=with_norm)
            self.act = nn.GELU() if use_GE else nn.Identity()
            self.offset_predictor = nn.Linear(embed_dim, offset_embed, bias=False)
        else:
            self.patch_embed = PatchEmbed(img_size, 1 if local_feature else patch_size, patch_count, in_chans, offset_embed)

        self.img_size, self.patch_size, self.patch_pixel, self.patch_count = img_size, patch_size, patch_pixel, patch_count
        self.in_chans, self.embed_dim = in_chans, embed_dim

    def reset_offset(self):
        if self.another_linear:
            nn.init.constant_(self.offset_predictor.weight, 0)
            if hasattr(self.offset_predictor, "bias") and self.offset_predictor.bias is not None:
                nn.init.constant_(self.offset_predictor.bias, 0)
        else:
            nn.init.constant_(self.patch_embed.proj.weight, 0)
            if hasattr(self.patch_embed.proj, "bias") and self.patch_embed.proj.bias is not None:
                nn.init.constant_(self.patch_embed.proj.bias, 0)
        print("Parameter for offsets reseted.")

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, model_offset=None):
        if x.dim() == 3:
            B, H, W = x.shape[0], self.img_size, self.img_size
            assert x.shape[1] == H * W
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        img = x
        x = self.patch_embed(x)   # x [N, patch_w * patch_h, C]
        if self.another_linear:
            pred_offset = self.offset_predictor(self.act(x))
        else:
            pred_offset = x.contiguous()
        return self.get_output(img, pred_offset, model_offset), (self.patch_count[0], self.patch_count[1])

class Simple_DePatch(Simple_Patch):
    def __init__(self, box_coder, show_dim=4, **kwargs):
        super().__init__(show_dim, **kwargs)
        self.box_coder = box_coder
        self.register_buffer("value_spatial_shapes", torch.as_tensor([[self.img_size[0], self.img_size[1]]], dtype=torch.long))
        self.register_buffer("value_level_start_index", torch.as_tensor([0], dtype=torch.long))
        self.output_proj = nn.Linear(self.in_chans * self.patch_pixel * self.patch_pixel, self.embed_dim)
        self.num_sample_points = self.patch_pixel * self.patch_pixel * self.patch_count[0] * self.patch_count[1]
        if kwargs["with_norm"]:
            self.with_norm=True
            self.norm = nn.LayerNorm(self.embed_dim)
        else:
            self.with_norm=False

    def get_output(self, img, pred_offset, model_offset=None):
        #copyed
        B = img.shape[0]
        sample_location = self.box_coder(pred_offset, model_offset)
        sampling_locations = sample_location.view(B, self.num_sample_points,1,1,1,2).to(torch.float)
        attention_weights = torch.ones((B, self.num_sample_points, 1, 1, 1), device=img.device)
        x = img.view(B, self.in_chans, 1, -1).transpose(1, 3).contiguous()
        output = MSDeformAttnFunction.apply(x, self.value_spatial_shapes, self.value_level_start_index, sampling_locations, attention_weights, 1)
        # output_proj
        output = output.view(B, self.num_patches, self.in_chans*self.patch_pixel*self.patch_pixel)
        output = self.output_proj(output)
        if self.with_norm:
            output = self.norm(output)
        return output

def build_patch_embeds(embed_dims, img_size, Depatch):   # img_size: [h,w]
    patch_embeds=[]
    # for i in range(1):
    inchans = embed_dims
    in_size = img_size
    patch_size = 4
    if Depatch:
        box_coder = pointwhCoder(input_size=in_size, patch_count=[in_size[0]//patch_size, in_size[1]//patch_size], weights=(1.,1.,1.,1.), pts=[3,3], tanh=True, wh_bias=torch.tensor(5./3.).sqrt().log())
        patch_embeds.append(
            Simple_DePatch(box_coder, img_size=in_size, patch_size=patch_size, patch_pixel=3, patch_count=[in_size[0]//patch_size, in_size[1]//patch_size],
             in_chans=inchans, embed_dim=embed_dims, another_linear=True, use_GE=True, with_norm=True))
    else:
        patch_embeds.append(
            PatchEmbed(img_size=in_size, patch_size=patch_size, in_chans=inchans,
                       embed_dim=embed_dims))
    return patch_embeds