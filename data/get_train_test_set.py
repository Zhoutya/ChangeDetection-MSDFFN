import torch
import data.data_preprocess as data_preprocess
from data.get_dataset import get_dataset as getdata


def get_train_test_set(cfg):
    # load
    current_dataset = cfg['current_dataset']
    train_set_num = cfg['train_set_num']
    patch_size = cfg['patch_size']

    img1, img2, gt = getdata(current_dataset)

    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)
    gt = torch.from_numpy(gt)

    img1 = img1.permute(2, 0, 1)   # channel放到第一维CxHxW
    img2 = img2.permute(2, 0, 1)
    img1 = data_preprocess.std_norm(img1)
    img2 = data_preprocess.std_norm(img2)
    # label transform
    img_gt = gt

    # construct samples
    img1_pad, img2_pad, patch_coordinates = data_preprocess.construct_sample(img1, img2, patch_size)

    # Divide samples
    data_sample = data_preprocess.select_sample(img_gt, train_set_num)

    data_sample['img1_pad'] = img1_pad
    data_sample['img2_pad'] = img2_pad

    data_sample['patch_coordinates'] = patch_coordinates
    data_sample['img_gt'] = img_gt  #
    data_sample['ori_gt'] = gt

    return data_sample
