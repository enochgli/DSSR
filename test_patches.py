from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import os
import math
import cv2
from model_ori_align_square_3_rrdb_grad import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='./data/Validation')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='iPASSR_4xSR_epoch126')
    return parser.parse_args()


def imgFusion(img_1_l, img_1_r, img_2_l, img_2_r, overlap, left_right=True):
    '''
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    '''
    # 考虑平行向融合
    # w = calWeight(overlap, 0.06)

    if left_right:  # 左右融合
        row_1, col_1, channel = img_1_l.shape
        row_2, col_2, channel = img_2_l.shape

        img_new_l = np.zeros((row_1, col_1 + col_2 - overlap, channel), np.float32)
        img_new_r = np.zeros((row_1, col_1 + col_2 - overlap, channel), np.float32)

        img_new_l[:, :col_1-8, :] = img_1_l[:, :col_1-8, :]
        img_new_r[:, :col_1-8, :] = img_1_r[:, :col_1-8, :]

        # w_expand = np.tile(w, (row_1, 1))  # 权重扩增
        img_new_l[:, col_1 - overlap+8:col_1-8, :] = 0.5 * img_1_l[:, col_1 - overlap+8:col_1-8, :] + 0.5 * img_2_l[:, 8:overlap-8, :]
        img_new_r[:, col_1 - overlap+8:col_1-8, :] = 0.5 * img_1_r[:, col_1 - overlap+8:col_1-8, :] + 0.5 * img_2_r[:, 8:overlap-8, :]

        img_new_l[:, col_1-8:, :] = img_2_l[:, overlap-8:, :]
        img_new_r[:, col_1-8:, :] = img_2_r[:, overlap-8:, :]

    else:  # 上下融合
        row_1, col_1, channel = img_1_l.shape
        row_2, col_2, channel = img_2_l.shape

        img_new_l = np.zeros((row_1 + row_2 - overlap, col_1, channel), np.float32)
        img_new_r = np.zeros((row_1 + row_2 - overlap, col_1, channel), np.float32)

        img_new_l[:row_1-8, :, :] = img_1_l[:row_1-8, :, :]
        img_new_r[:row_1-8, :, :] = img_1_r[:row_1-8, :, :]

        img_new_l[row_1 - overlap+8:row_1-8, :, :] = 0.5 * img_1_l[row_1 - overlap+8:row_1-8, :, :] + \
                                          0.5 * img_2_l[8:overlap-8, :, :]
        img_new_r[row_1 - overlap+8:row_1-8, :, :] = 0.5 * img_1_r[row_1 - overlap+8:row_1-8, :, :] + \
                                          0.5 * img_2_r[8:overlap-8, :, :]

        img_new_l[row_1-8:, :, :] = img_2_l[overlap-8:, :, :]
        img_new_r[row_1-8:, :, :] = img_2_r[overlap-8:, :, :]

    return img_new_l, img_new_r

def test_patch(net, cfg, LR_left, LR_right, rows, cols, overlap_h, overlap_w, patch_h, patch_w):
    h, w, _ = LR_left.shape
    out_row_l = []
    out_row_r = []

    for y in range(rows):
        img_l_1 = None
        img_r_1 = None
        img_l_2 = None
        img_r_2 = None

        for x in range(cols):
            left = x * overlap_w
            right = left + patch_w
            top = y * overlap_h
            bottom = top + patch_h

            if x != (cols - 1) and y != (rows - 1):
                real_left = left
                real_right = right
                real_top = top
                real_bottom = bottom

            elif x == (cols - 1) and y != (rows - 1):
                if right >= w:
                    real_right = w
                    real_left = real_right - patch_w
                    real_top = top
                    real_bottom = bottom
            elif x != (cols - 1) and y == (rows - 1):
                if bottom >= h:
                    real_bottom = h
                    real_top = real_bottom - patch_h
                    real_left = left
                    real_right = right
            else:
                # print('the latest patch')
                real_left = left
                real_right = right
                real_top = top
                real_bottom = bottom
                if right >= w:
                    real_right = w
                    real_left = real_right - patch_w
                if bottom >= h:
                    real_bottom = h
                    real_top = real_bottom - patch_h

            # last_bottom_idx.append(real_bottom)
            subimg_l = LR_left[real_top:real_bottom, real_left:real_right, :]
            subimg_r = LR_right[real_top:real_bottom, real_left:real_right, :]

            #print(str(real_top)+'|'+str(real_bottom)+'|'+str(real_left)+'|'+str(real_right))

            subimg_l, subimg_r = ToTensor()(subimg_l), ToTensor()(subimg_r)
            subimg_l, subimg_r = subimg_l.unsqueeze(0), subimg_r.unsqueeze(0)
            subimg_l, subimg_r = Variable(subimg_l).to(cfg.device), Variable(subimg_r).to(cfg.device)

            with torch.no_grad():
                sub_sr_l, sub_sr_r = net(subimg_l, subimg_r, is_training=0)
            sub_sr_l = torch.squeeze(sub_sr_l.data.cpu(), 0).detach().numpy()
            sub_sr_r = torch.squeeze(sub_sr_r.data.cpu(), 0).detach().numpy()

            sub_sr_l_ = np.transpose(sub_sr_l, axes=(1, 2, 0))
            sub_sr_r_ = np.transpose(sub_sr_r, axes=(1, 2, 0))
            # LR_left = np.ascontiguousarray(sub_sr_l)
            # LR_right = np.ascontiguousarray(LR_right)

            if x == 0:
                img_l_1 = sub_sr_l_
                img_r_1 = sub_sr_r_
                last_right_idx = real_right
                continue

            img_l_2 = sub_sr_l_
            img_r_2 = sub_sr_r_

            overlap = last_right_idx * 4 - real_left * 4
            new_l, new_r = imgFusion(img_l_1, img_r_1, img_l_2, img_r_2, overlap, True)
            img_l_1 = new_l[:]
            img_r_1 = new_r[:]
            last_right_idx = real_right

        out_row_l.append(new_l)
        out_row_r.append(new_r)

    # last_bottom_idx = list(set(last_bottom_idx))
    img_1_l = out_row_l[0]
    img_1_r = out_row_r[0]
    for i in range(1, rows):
        img_2_l = out_row_l[i]
        img_2_r = out_row_r[i]

        h_1, _, _ = img_1_l.shape
        h_2, _, _ = img_2_l.shape

        # overlap = h_2 + - last_bottom_idx[i-1]
        if (h_1 + h_2 - overlap_h * 4) > h * 4:
            overlap = patch_h * 4 - (h * 4 - h_1)
        else:
            overlap = overlap_h * 4

        new_img_l, new_img_r = imgFusion(img_1_l, img_1_r, img_2_l, img_2_r, overlap, left_right=False)
        img_1_l = new_img_l[:]
        img_1_r = new_img_r[:]

    return new_img_l, new_img_r

def test(cfg):
    net = Net(cfg.scale_factor).to(cfg.device)
    model = torch.load('./log/' + cfg.model_name + '.pth.tar')
    net.load_state_dict(model['state_dict'])
    overlap_h = 48
    overlap_w = 48
    patch_h = 96
    patch_w = 96

    file_list = os.listdir(cfg.testset_dir + '/' + cfg.dataset + '/LR_x' + str(cfg.scale_factor))#'/patches_x' + str(cfg.scale_factor))
    file_list = sorted(file_list)

    for i in range(0, len(file_list), 2): #range(1, 113):
        idx = file_list[i][:4]
        # if idx in ['0066', '0077', '0081']:
        #     continue
        LR_left = Image.open(cfg.testset_dir + '/' + cfg.dataset + '/LR_x' + str(cfg.scale_factor) + '/' + idx + '_L.png')
        LR_right = Image.open(cfg.testset_dir + '/' + cfg.dataset + '/LR_x' + str(cfg.scale_factor) + '/' + idx + '_R.png')

        LR_left = np.array(LR_left)
        LR_right = np.array(LR_right)

        print('Running Scene ' + idx + ' of ' + cfg.dataset + ' Dataset......')
        h, w, _ = LR_left.shape

        rows = math.ceil((h - patch_h)/overlap_h) + 1 # math.ceil(h/overlap_h)
        cols = math.ceil((w - patch_w)/overlap_w) + 1

        out_row_l = []  # np.zeros([128, w * 4, 3], np.float32)   # sr row img
        out_row_r = []
        last_bottom_idx = []

        out_sr_l, out_sr_r = test_patch(net, LR_left, LR_right, rows, cols, overlap_h, overlap_w, patch_h, patch_w)

        SR_left, SR_right = (255*np.clip(out_sr_l, 0, 1)).astype('uint8'), (255*np.clip(out_sr_r, 0, 1)).astype('uint8')

        save_path = './results/' + cfg.model_name + '/' + cfg.dataset #+ '/' + idx
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        SR_left_img = transforms.ToPILImage()(SR_left)
        SR_left_img.save(save_path + '/' + idx + '_L.png')
        SR_right_img = transforms.ToPILImage()(SR_right)
        SR_right_img.save(save_path+ '/' + idx + '_R.png')


if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = ['Flickr1024']
    for i in range(len(dataset_list)):
        cfg.dataset = dataset_list[i]
        test(cfg)
    print('Finished!')
