import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from torchvision import transforms

from models.model_util import DCNv2Pack
from models.Stereo_Transformer_new import build_position_encoding, Transformer

class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.
    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            padding=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.
        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x    

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x
    
class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.deep_feature = RDG(G0=64, C=4, G=24, n_RDB=4)
        
        # align
        self.init_align = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        self.pcd_align = PCDAlignment(num_feat=64, deformable_groups=8)
        self.conv_l2_1 = nn.Conv2d(64, 64, 3, 2, 1)  # pyramid
        self.conv_l2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(64, 64, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.align_fusion_left = nn.Conv2d(6*64, 4*64, 3, 1, 1)
        self.align_fusion_right = nn.Conv2d(6*64, 4*64, 3, 1, 1)
        
        self.pam = PAM(64)
        self.fusion = nn.Sequential(
            RDB(G0=128, C=4, G=32),
            CALayer(128),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True))
        
        self.get_g_nopadding = Get_gradient_nopadding()
        self.init_grad = nn.Conv2d(3, 64, 3, 1, 1, bias=True)
        
        self.reconstruct = RDG_with_Grad(G0=64, C=4, G=24, n_RDB=4) # RDG(G0=64, C=4, G=24, n_RDB=4)
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=True))

    def forward(self, x_left, x_right, is_training):
        x_left_upscale = F.interpolate(x_left, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x_right_upscale = F.interpolate(x_right, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        buffer_left, catfea_left = self.deep_feature(buffer_left)
        buffer_right, catfea_right = self.deep_feature(buffer_right)
        
        # Align
        align_left = self.init_align(x_left)
        align_right = self.init_align(x_right)
        feat_l1_l = align_left.clone()  # pyramid l1 (original)
        feat_l2_l = self.lrelu(self.conv_l2_1(align_left))  # pyramid l2(1/2 spatial)
        feat_l2_l = self.lrelu(self.conv_l2_2(feat_l2_l))
        feat_l3_l = self.lrelu(self.conv_l3_1(feat_l2_l))  # pyramid l3(1/4 spatial)
        feat_l3_l = self.lrelu(self.conv_l3_2(feat_l3_l))
        feat_l1_r = align_right.clone()   # pyramid l1 (original)
        feat_l2_r = self.lrelu(self.conv_l2_1(align_right))  # pyramid l2(1/2 spatial)
        feat_l2_r = self.lrelu(self.conv_l2_2(feat_l2_r))
        feat_l3_r = self.lrelu(self.conv_l3_1(feat_l2_r))  # pyramid l3(1/4 spatial)
        feat_l3_r = self.lrelu(self.conv_l3_2(feat_l3_r))
        left_pyramid = [
            feat_l1_l,
            feat_l2_l,
            feat_l3_l
        ]
        right_pyramid = [
            feat_l1_r,
            feat_l2_r,
            feat_l3_r
        ]
        aligned_feat_l_2_r = self.pcd_align(left_pyramid, right_pyramid)  # right as reference
        aligned_feat_r_2_l = self.pcd_align(right_pyramid, left_pyramid)
        catfea_left = self.align_fusion_left(torch.cat((align_left, aligned_feat_l_2_r, catfea_left), dim=1))
        catfea_right = self.align_fusion_left(torch.cat((align_right, aligned_feat_r_2_l, catfea_right), dim=1))
        
        # gradients
        x_left_grad = self.get_g_nopadding(x_left)
        x_right_grad = self.get_g_nopadding(x_right)
        x_left_grad = self.init_grad(x_left_grad)
        x_right_grad = self.init_grad(x_right_grad)
        

        if is_training == 1:
            buffer_leftT, buffer_rightT, (M_right_to_left, M_left_to_right), (V_left, V_right)\
                = self.pam(buffer_left, buffer_right, catfea_left, catfea_right, is_training)
        if is_training == 0:
            buffer_leftT, buffer_rightT \
                = self.pam(buffer_left, buffer_right, catfea_left, catfea_right, is_training)

        buffer_leftF = self.fusion(torch.cat([buffer_left, buffer_leftT], dim=1))
        buffer_rightF = self.fusion(torch.cat([buffer_right, buffer_rightT], dim=1))
        buffer_leftF, _ = self.reconstruct(buffer_leftF, x_left_grad)
        buffer_rightF, _ = self.reconstruct(buffer_rightF, x_right_grad)
        out_left = self.upscale(buffer_leftF) + x_left_upscale
        out_right = self.upscale(buffer_rightF) + x_right_upscale

        if is_training == 1:
            return out_left, out_right, (M_right_to_left, M_left_to_right), (V_left, V_right)
        if is_training == 0:
            return out_left, out_right


class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x

class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block with gradient fusion
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=24, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        # self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type,norm_type, act_type, mode)
        # self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        # self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)

        self.RDB1 = RDB(nc, 4, gc)
        self.RDB2 = RDB(nc, 4, gc)
        self.RDB3 = RDB(nc, 4, gc)

        self.fusion = nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = out.mul(0.2) + x
        out = self.relu(self.fusion(out))
        return out
    
class RRDB_grad(nn.Module):
    '''
    Residual in Residual Dense Block with gradient fusion
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=24, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB_grad, self).__init__()
        # self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type,norm_type, act_type, mode)
        # self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        # self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)

        self.RDB1 = RDB(nc, 4, gc)
        self.RDB2 = RDB(nc, 4, gc)
        self.RDB3 = RDB(nc, 4, gc)

        self.fusion = nn.Conv2d(nc, int(0.5 * nc), kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = out.mul(0.2) + x
        out = self.relu(self.fusion(out))
        return out

class RDG_with_Grad(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG_with_Grad, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        RRDBs = []  # for gradients
        for i in range(n_RDB):
            RDBs.append(RRDB(G0, C, G))
            RRDBs.append(RRDB_grad(G0*2, C, G))
            #RRDBs.append(one_fusion(G0*2, G0))
        self.RDB = nn.Sequential(*RDBs)
        self.RRDB = nn.Sequential(*RRDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)
        self.grad_fusion = nn.Conv2d(G0*(n_RDB+1), G0*(n_RDB), kernel_size=1, stride=1, padding=0, bias=True)
        print(len(RRDBs))

    def forward(self, x, x_grad):
        buffer = x
        temp = []
        x_cat = x_grad
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            x_cat = torch.cat([buffer, x_cat], dim=1)
            # print(x_cat.shape)
            x_cat = self.RRDB[i](x_cat)
            temp.append(buffer)
        grad_cat = x_cat + x_grad
        temp.append(grad_cat)
        buffer_cat = torch.cat(temp, dim=1)
        buffer_cat = self.grad_fusion(buffer_cat)
        out = self.conv(buffer_cat)
        return out, buffer_cat

class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RRDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out, buffer_cat


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//16, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel//16, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x


class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.bq = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.bs = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(4 * channels)
        self.bn = nn.BatchNorm2d(4 * channels)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, is_training):
        b, c0, h0, w0 = x_left.shape
        Q = self.bq(self.rb(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c0, 1, 1)

        if is_training == 1:
            return out_left, out_right, \
                   (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)),\
                   (V_left_tanh, V_right_tanh)
        if is_training == 0:
            return out_left, out_right


def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M = pad(M[:, :-1-i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M = pad(M[:, i+1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed


if __name__ == "__main__":
    net = Net(upscale_factor=4)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
