import torch
import torch.nn as nn

from collections import OrderedDict

from . import sisr_architecture as arch

class SPSRModel(nn.Module):
    def __init__(self, cfg):
        super(SPSRModel, self).__init__()
        self.cfg = cfg
        # define networks and load pretrained models
        self.netG = arch.SPSRNet(in_nc=cfg.in_nc, out_nc=cfg.out_nc, nf=cfg.nf,
                            nb=cfg.nb, gc=cfg.gc, upscale=cfg.scale,
                            act_type='leakyrelu', mode=cfg.mode, upsample_mode='upconv')

        self.load()  # load G and D if needed

    def load(self):
        load_path_G = self.cfg.model_G_path
        if load_path_G is not None:
            print('Load model G')

            if isinstance(self.netG, nn.DataParallel):
                self.netG = self.netG.module
            pretrained_dict = torch.load(load_path_G)
            #model_dict = self.netG.state_dict()
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            #model_dict.update(pretrained_dict)
            self.netG.load_state_dict(pretrained_dict)

    def test(self, LR_img):
        self.netG.eval()
        with torch.no_grad():
            SR_img = self.netG(LR_img)
        return SR_img