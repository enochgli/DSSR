from torch.autograd import Variable
import torch.nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import argparse
import os, math, cv2, copy
import itertools
from model_ori_align_square_3_rrdb_grad import *
from test_patches import test_patch, imgFusion
from utils import *
from test_SISR import test_SISR
from models import SISR_model

#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='./test_LR')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='./log_SSR/iPASSR_4xSR_epoch236_49300.pth.tar')
    parser.add_argument('--self_ensemble', type=bool, default=True)
    parser.add_argument('--test_patch', type=bool, default=True)
    parser.add_argument('--test_SISR', type=bool, default=True)
    parser.add_argument('--patch_h', type=list, default=[80, 96, 112])
    parser.add_argument('--patch_w', type=list, default=[80, 96, 112])
    parser.add_argument('--overlap_h', type=list, default=[40, 48, 56])
    parser.add_argument('--overlap_w', type=list, default=[40, 48, 56])
    parser.add_argument('--model', type=str, default='spsr')  # SISR option
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--model_G_path', type=str, default='./log_SISR/259200_G.pth')
    parser.add_argument('--mode', type=str, default='CNA')
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=23)
    parser.add_argument('--in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--gc', type=int, default=32)
    parser.add_argument('--group', type=int, default=1)
    return parser.parse_args()

def load_GPUs_pth(model_path, mapLoc='cuda'):
    model = torch.load(model_path, map_location=mapLoc)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    new_model = OrderedDict()
    new_model['epoch'] = model['epoch']
    new_model['state_dict'] = new_state_dict
    for k, v in model['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_model

def compute_diff(inimg1, inimg2):
    img1 = np.float64(cv2.cvtColor(inimg1, cv2.COLOR_BGR2GRAY))
    img2 = np.float64(cv2.cvtColor(inimg2, cv2.COLOR_BGR2GRAY))
    diff = np.sum(np.abs(img1 - img2))
    return diff

def get_pair_diff(img_list):
    diff_list = []
    for i in range(len(img_list)):
        row_diff = 0
        for j in range(len(img_list)):
            if j == i:
                continue
            row_diff += compute_diff(img_list[j], img_list[i])
        diff_list.append(row_diff)
    return diff_list

def get_ensemble_idx(diff, numberofimage=4):
    img_id = []
    for k in range(numberofimage):
        img_id.append(diff.index(max(diff)))
        diff[diff.index(max(diff))] = -1
    return img_id
    
def computer_average(img_list, img_id, numberofimage=4):
    sum_val = np.zeros(img_list[0].shape)
    for c in range(3):
        for cnt in range(numberofimage):
            if img_id[cnt] == 0:
                sum_val[:, :, c] = sum_val[:, :, c] + np.float64(img_list[0][:, :, c])
            elif img_id[cnt] == 1:
                sum_val[:, :, c] = sum_val[:, :, c] + np.float64(img_list[1][:, :, c])
            elif img_id[cnt] == 2:
                sum_val[:, :, c] = sum_val[:, :, c] + np.float64(img_list[2][:, :, c])
            elif img_id[cnt] == 3:
                sum_val[:, :, c] = sum_val[:, :, c] + np.float64(img_list[3][:, :, c])
            elif img_id[cnt] == 4:
                sum_val[:, :, c] = sum_val[:, :, c] + np.float64(img_list[4][:, :, c])
            elif img_id[cnt] == 5:
                sum_val[:, :, c] = sum_val[:, :, c] + np.float64(img_list[5][:, :, c])
            elif img_id[cnt] == 6:
                sum_val[:, :, c] = sum_val[:, :, c] + np.float64(img_list[6][:, :, c])
            elif img_id[cnt] == 7:
                sum_val[:, :, c] = sum_val[:, :, c] + np.float64(img_list[7][:, :, c])
    sum_val = ((sum_val / numberofimage) * 255).astype('uint8')
    return sum_val

def get_grad_ensemble(left_img_list, right_img_list, numberofimage=4):
    #get pair diff
    left_diff = get_pair_diff(left_img_list)
    right_diff = get_pair_diff(right_img_list)

    left_ensemble_id = get_ensemble_idx(left_diff, numberofimage)
    right_ensemble_id = get_ensemble_idx(right_diff, numberofimage)

    sr_left = computer_average(left_img_list, left_ensemble_id, numberofimage)
    sr_right = computer_average(right_img_list, right_ensemble_id, numberofimage)

    return sr_left, sr_right

def get_adaptive_grad_ensemble(left_img_list, right_img_list, number_image_list=[4,5,6,7]):
    # get pair diff
    left_diff = get_pair_diff(left_img_list)
    right_diff = get_pair_diff(right_img_list)
    
    left_diff_copy = copy.deepcopy(left_diff)
    right_diff_copy = copy.deepcopy(right_diff)
    
    left_ensemble_id_cmp, last_l_diff_cmp = get_ensemble_idx(left_diff_copy, 4)
    right_ensemble_id_cmp, last_r_diff_cmp = get_ensemble_idx(right_diff_copy, 4)
    
    for i in number_image_list[1:]:
        left_ensemble_id, last_l_diff = get_ensemble_idx(left_diff, i)
        right_ensemble_id, last_r_diff = get_ensemble_idx(right_diff, i)
        
        if last_l_diff > last_l_diff_cmp * 0.97:
            left_ensemble_id_cmp = left_ensemble_id
        if last_r_diff > last_r_diff_cmp * 0.97:
            right_ensemble_id_cmp = right_ensemble_id
    
    sr_left = computer_average(left_img_list, left_ensemble_id_cmp, len(left_ensemble_id_cmp))
    sr_right = computer_average(right_img_list, right_ensemble_id_cmp, len(right_ensemble_id_cmp))

    return sr_left, sr_right, len(left_ensemble_id_cmp), len(right_ensemble_id_cmp)


def test(cfg):
    net = Net(cfg.scale_factor).to(cfg.device)
    model_path = cfg.model_path
    model = torch.load(model_path)
    net.load_state_dict(model['state_dict'])

    psnr_val_4 = 0
    psnr_val_5 = 0
    psnr_val_6 = 0
    psnr_val_7 = 0
    psnr_val_adaptive = 0

    for idx in range(1, 101):
        LR_left = Image.open(cfg.testset_dir + '/' + str(idx).zfill(4) + '_L.png')
        LR_right = Image.open(cfg.testset_dir + '/' + str(idx).zfill(4) + '_R.png')
        
        print('Time: ' + get_now_time() + ' Running Scene ' + str(idx) + '......')

        if (cfg.self_ensemble):
            sr_left_images = []
            sr_right_images = []
            for flip_index in range(2):  # for flipping

                LR_left = np.transpose(LR_left, axes=(1, 0, 2))
                LR_right = np.transpose(LR_right, axes=(1, 0, 2))

                for rotate_index in range(4):  # for rotating

                    LR_left = np.rot90(LR_left, k=1, axes=(0, 1))
                    LR_right = np.rot90(LR_right, k=1, axes=(0, 1))

                    # if rotate_index == 0 or rotate_index == 2:
                    #     continue

                    LR_left = np.ascontiguousarray(LR_left)
                    LR_right = np.ascontiguousarray(LR_right)

                    if cfg.test_patch:
                        if len(cfg.patch_h) != 1:
                            SR_left_ms = []
                            SR_right_ms = []
                            for ii in range(len(cfg.patch_h)):
                                overlap_h = cfg.overlap_h[ii]
                                overlap_w = cfg.overlap_w[ii]
                                patch_h = cfg.patch_h[ii]
                                patch_w = cfg.patch_w[ii]
                                h, w, _ = LR_left.shape
                                rows = math.ceil((h - patch_h) / overlap_h) + 1  # math.ceil(h/overlap_h)
                                cols = math.ceil((w - patch_w) / overlap_w) + 1

                                SR_left, SR_right = test_patch(net, cfg, LR_left, LR_right, rows, cols, overlap_h, overlap_w, patch_h,
                                                                patch_w)
                                SR_left = np.clip(SR_left, 0, 1)
                                SR_right = np.clip(SR_right, 0, 1)
                                SR_left_ms.append(SR_left)
                                SR_right_ms.append(SR_right)
                            SR_left = np.mean(SR_left_ms, axis=0)
                            SR_right = np.mean(SR_right_ms, axis=0)
                        else:
                            overlap_h = cfg.overlap_h[0]
                            overlap_w = cfg.overlap_w[0]
                            patch_h = cfg.patch_h[0]
                            patch_w = cfg.patch_w[0]
                            h, w, _ = LR_left.shape
                            rows = math.ceil((h - patch_h) / overlap_h) + 1  # math.ceil(h/overlap_h)
                            cols = math.ceil((w - patch_w) / overlap_w) + 1

                            SR_left, SR_right = test_patch(net, cfg, LR_left, LR_right, rows, cols, overlap_h, overlap_w, patch_h,
                                                            patch_w)
                            SR_left = np.clip(SR_left, 0, 1)
                            SR_right = np.clip(SR_right, 0, 1)

                    SR_left = np.rot90(SR_left, k=(3 - rotate_index), axes=(0, 1))
                    SR_right = np.rot90(SR_right, k=(3 - rotate_index), axes=(0, 1))
                    if (flip_index == 0):
                        SR_left = np.transpose(SR_left, axes=(1, 0, 2))
                        SR_right = np.transpose(SR_right, axes=(1, 0, 2))

                    sr_left_images.append(SR_left)
                    sr_right_images.append(SR_right)

            # single image super resolution
            if cfg.test_SISR:
                print('test SISR')
                # Create model
                model = SISR_model.SPSRModel(cfg)
                model.to(cfg.device)
                SISR_left_images = []
                SISR_right_images = []
                for id in range(len(sr_left_images)):
                    SISR_left_images.append(test_SISR(cfg, model, sr_left_images[id]))
                    SISR_right_images.append(test_SISR(cfg, model, sr_right_images[id]))

            #### ensemble fusion
            sr_left, sr_right = get_grad_ensemble(SISR_left_images, SISR_right_images, numberofimage=4)  # return uint8

        else:
            LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
            LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)

            LR_left, LR_right = Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

            with torch.no_grad():
                SR_left, SR_right = net(LR_left, LR_right, is_training=0)
                SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)
        
        save_path = './results/'
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        
        SR_left_img = transforms.ToPILImage()(sr_left)
        SR_left_img.save(save_path + '/' + str(idx).zfill(4) + '_L.png')
        SR_right_img = transforms.ToPILImage()(sr_right)
        SR_right_img.save(save_path + '/' + str(idx).zfill(4) + '_R.png')
                


if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = ['Flickr1024'] #'KITTI2012', 'KITTI2015', 'Middlebury']
    for i in range(len(dataset_list)):
        print('SSR_path--%s' % cfg.model_path)
        print('SISR_path--%s' % cfg.model_G_path)
        cfg.dataset = dataset_list[i]
        test(cfg)
    print('Finished!')
