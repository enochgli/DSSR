import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional
from torch import Tensor
import math

from utils import get_clones
from models.depatch_embed import build_patch_embeds

class Transformer(nn.Module):
    """
    Transformer computes self (intra image) and cross (inter image) attention
    """

    def __init__(self, img_size: list = [32, 128], hidden_dim: int = 128, nhead: int = 8, num_attn_layers: int = 6):
        super().__init__()

        self_attn_layer = TransformerSelfAttnLayer(hidden_dim, nhead)
        self.self_attn_layers = get_clones(self_attn_layer, num_attn_layers)

        cross_attn_layer = TransformerCrossAttnLayer(hidden_dim, nhead)
        self.cross_attn_layers = get_clones(cross_attn_layer, num_attn_layers)

        self.norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers

        patch_embeds_left = build_patch_embeds(embed_dims=hidden_dim, img_size=img_size, Depatch=True)
        patch_embeds_right = build_patch_embeds(embed_dims=hidden_dim, img_size=img_size, Depatch=True)
        self.patch_embeds_left = nn.Sequential(*patch_embeds_left)
        self.patch_embeds_right = nn.Sequential(*patch_embeds_right)

    def _alternating_attn(self, feat: torch.Tensor, pos_enc: torch.Tensor, pos_indexes: Tensor, hn: int):
        """
        Alternate self and cross attention with gradient checkpointing to save memory
        :param feat: image feature concatenated from left and right, [W,2HN,C]
        :param pos_enc: positional encoding, [W,HN,C]
        :param pos_indexes: indexes to slice positional encoding, [W,HN,C]
        :param hn: size of HN
        :return: attention weight [N,H,W,W]
        """

        global layer_idx
        # alternating
        for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
            layer_idx = idx

            # checkpoint self attn
            def create_custom_self_attn(module):
                def custom_self_attn(*inputs):
                    return module(*inputs)

                return custom_self_attn

            feat = checkpoint(create_custom_self_attn(self_attn), feat, pos_enc, pos_indexes)

            # add a flag for last layer of cross attention
            if idx == self.num_attn_layers - 1:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, True)

                    return custom_cross_attn
            else:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, False)

                    return custom_cross_attn

            feat, attn_weight = checkpoint(create_custom_cross_attn(cross_attn), feat[:, :hn], feat[:, hn:], pos_enc,
                                           pos_indexes)

        layer_idx = 0
        return attn_weight

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor, pos_enc: Optional[Tensor] = None):
        """
        :param feat_left: feature descriptor of left image, [N,C,H,W]
        :param feat_right: feature descriptor of right image, [N,C,H,W]
        :param pos_enc: relative positional encoding, [N,C,H,2W-1]
        :return: cross attention values [N,H,W,W], dim=2 is left image, dim=3 is right image
        """

        # flatten NxCxHxW to WxHNxC
        bs, c, hn, w = feat_left.shape

        # feat_left, _ = self.patch_embeds_left(feat_left)
        # feat_right, _ = self.patch_embeds_left(feat_right)

        feat_left = feat_left.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)  # CxWxHxN -> CxWxHN -> WxHNxC
        feat_right = feat_right.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)

        if pos_enc is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding
                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(feat_left.device)
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(feat_left.device)
                pos_indexes = (indexes_r + indexes_c).view(-1).long()  # WxW' -> WW'
        else:
            pos_indexes = None

        # concatenate left and right features
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC

        # compute attention
        attn_weight = self._alternating_attn(feat, pos_enc, pos_indexes, hn)
        attn_weight = attn_weight.view(hn, bs, w, w).permute(1, 0, 2, 3)  # NxHxWxW, dim=2 left image, dim=3 right image

        return attn_weight

class TransformerSelfAttnLayer(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.self_attn = MultiheadAttentionRelative(hidden_dim, nhead)

        self.norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, feat: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None):
        """
        :param feat: image feature [W,2HN,C]
        :param pos: pos encoding [2W-1,HN,C]
        :param pos_indexes: indexes to slice pos encoding [W,W]
        :return: updated image feature
        """
        feat2 = self.norm1(feat)

        # torch.save(feat2, 'feat_self_attn_input_' + str(layer_idx) + '.dat')

        feat2, attn_weight, _ = self.self_attn(query=feat2, key=feat2, value=feat2, pos_enc=pos,
                                               pos_indexes=pos_indexes)

        # torch.save(attn_weight, 'self_attn_' + str(layer_idx) + '.dat')

        feat = feat + feat2

        return feat

class TransformerCrossAttnLayer(nn.Module):
    """
    Cross attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.cross_attn = MultiheadAttentionRelative(hidden_dim, nhead)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, feat_left: Tensor, feat_right: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None,
                last_layer: Optional[bool] = False):
        """
        :param feat_left: left image feature, [W,HN,C]
        :param feat_right: right image feature, [W,HN,C]
        :param pos: pos encoding, [2W-1,HN,C]
        :param pos_indexes: indexes to slicer pos encoding [W,W]
        :param last_layer: Boolean indicating if the current layer is the last layer
        :return: update image feature and attention weight
        """
        feat_left_2 = self.norm1(feat_left)
        feat_right_2 = self.norm1(feat_right)

        # torch.save(torch.cat([feat_left_2, feat_right_2], dim=1), 'feat_cross_attn_input_' + str(layer_idx) + '.dat')

        # update right features
        if pos is not None:
            pos_flipped = torch.flip(pos, [0])
        else:
            pos_flipped = pos
        feat_right_2 = self.cross_attn(query=feat_right_2, key=feat_left_2, value=feat_left_2, pos_enc=pos_flipped,
                                       pos_indexes=pos_indexes)[0]

        feat_right = feat_right + feat_right_2

        # update left features
        # use attn mask for last layer
        if last_layer:
            w = feat_left_2.size(0)
            attn_mask = self._generate_square_subsequent_mask(w).to(feat_left.device)  # generate attn mask
        else:
            attn_mask = None

        # normalize again the updated right features
        feat_right_2 = self.norm2(feat_right)
        feat_left_2, attn_weight, raw_attn = self.cross_attn(query=feat_left_2, key=feat_right_2, value=feat_right_2,
                                                             attn_mask=attn_mask, pos_enc=pos,
                                                             pos_indexes=pos_indexes)

        # torch.save(attn_weight, 'cross_attn_' + str(layer_idx) + '.dat')

        feat_left = feat_left + feat_left_2

        # concat features
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC

        return feat, raw_attn

    @torch.no_grad()
    def _generate_square_subsequent_mask(self, sz: int):
        """
        Generate a mask which is upper triangular
        :param sz: square matrix size
        :return: diagonal binary mask [sz,sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask[mask == 1] = float('-inf')
        return mask

class MultiheadAttentionRelative(nn.MultiheadAttention):
    """
    Multihead attention with relative positional encoding
    """

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionRelative, self).__init__(embed_dim, num_heads, dropout=0.0, bias=True,
                                                         add_bias_kv=False, add_zero_attn=False,
                                                         kdim=None, vdim=None)

    def forward(self, query, key, value, attn_mask=None, pos_enc=None, pos_indexes=None):
        """
        Multihead attention
        :param query: [W,HN,C]
        :param key: [W,HN,C]
        :param value: [W,HN,C]
        :param attn_mask: mask to invalidate attention, -inf is used for invalid attention, [W,W]
        :param pos_enc: [2W-1,C]
        :param pos_indexes: index to select relative encodings, flattened in transformer WW
        :return: output value vector, attention with softmax (for debugging) and raw attention (used for last layer)
        """

        w, bsz, embed_dim = query.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # project to get qkv
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # cross-attention
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = self.in_proj_bias
                _start = embed_dim
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        # project to find q_r, k_r
        if pos_enc is not None:
            # reshape pos_enc
            pos_enc = torch.index_select(pos_enc, 0, pos_indexes).view(w, w,   # HxWxC
                                                                       -1)  # 2W-1xC -> WW'xC -> WxW'xC
            # compute k_r, q_r
            _start = 0
            _end = 2 * embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            _b = self.in_proj_bias[_start:_end]
            q_r, k_r = F.linear(pos_enc, _w, _b).chunk(2, dim=-1)  # WxW'xC
        else:
            q_r = None
            k_r = None

        # scale query
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        if q_r is not None:
            q_r = q_r * scaling

        # reshape
        q = q.contiguous().view(w, bsz, self.num_heads, head_dim)  # WxNxExC
        if k is not None:
            k = k.contiguous().view(-1, bsz, self.num_heads, head_dim)
        if v is not None:
            v = v.contiguous().view(-1, bsz, self.num_heads, head_dim)

        if q_r is not None:
            q_r = q_r.contiguous().view(w, w, self.num_heads, head_dim)  # WxW'xExC
        if k_r is not None:
            k_r = k_r.contiguous().view(w, w, self.num_heads, head_dim)

        # compute attn weight
        attn_feat = torch.einsum('wnec,vnec->newv', q, k)  # NxExWxW'

        # add positional terms
        if pos_enc is not None:
            # 0.3 s
            attn_feat_pos = torch.einsum('wnec,wvec->newv', q, k_r)  # NxExWxW'
            attn_pos_feat = torch.einsum('vnec,wvec->newv', k, q_r)  # NxExWxW'

            # 0.1 s
            attn = attn_feat + attn_feat_pos + attn_pos_feat
        else:
            attn = attn_feat

        assert list(attn.size()) == [bsz, self.num_heads, w, w]

        # apply attn mask
        if attn_mask is not None:
            attn_mask = attn_mask[None, None, ...]
            attn += attn_mask

        # raw attn
        raw_attn = attn

        # softmax
        attn = F.softmax(attn, dim=-1)

        # compute v, equivalent to einsum('',attn,v),
        # need to do this because apex does not support einsum when precision is mixed
        v_o = torch.bmm(attn.view(bsz * self.num_heads, w, w),
                        v.permute(1, 2, 0, 3).view(bsz * self.num_heads, w, head_dim))  # NxExWxW', W'xNxExC -> NExWxC
        assert list(v_o.size()) == [bsz * self.num_heads, w, head_dim]
        v_o = v_o.reshape(bsz, self.num_heads, w, head_dim).permute(2, 0, 1, 3).reshape(w, bsz, embed_dim)
        v_o = F.linear(v_o, self.out_proj.weight, self.out_proj.bias)

        # average attention weights over heads
        attn = attn.sum(dim=1) / self.num_heads

        # raw attn
        raw_attn = raw_attn.sum(dim=1)

        return v_o, attn, raw_attn

def build_position_encoding(position_encoding='sine1d_rel', channel_dim=64):
    mode = position_encoding
    channel_dim = channel_dim
    if mode == 'sine1d_rel':
        n_steps = channel_dim
        position_encoding = PositionEncodingSine1DRelative(n_steps, normalize=False)
    elif mode == 'none':
        position_encoding = no_pos_encoding
    else:
        raise ValueError(f"not supported {mode}")

    return position_encoding

def no_pos_encoding(x):
    return None

class PositionEncodingSine1DRelative(nn.Module):
    """
    relative sine encoding 1D, partially inspired by DETR (https://github.com/facebookresearch/detr)
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    @torch.no_grad()
    def forward(self, inputs):
        """
        :param inputs: NestedTensor
        :return: pos encoding [N,C,H,2W-1]
        """
        x = inputs['left']

        # update h and w if downsampling
        bs, _, h, w = x.size()
        # if inputs.sampled_cols is not None:
        #     bs, w = inputs.sampled_cols.size()
        # if inputs.sampled_rows is not None:
        #     _, h = inputs.sampled_rows.size()

        # populate all possible relative distances
        x_embed = torch.linspace(w - 1, -w + 1, 2 * w - 1, dtype=torch.float32, device=x.device)

        # scale distance if there is down sample
        # if inputs.sampled_cols is not None:
        #     scale = x.size(-1) / float(inputs.sampled_cols.size(-1))
        #     x_embed = x_embed * scale

        if self.normalize:
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t  # 2W-1xC
        # interleave cos and sin instead of concatenate
        pos = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC

        return pos