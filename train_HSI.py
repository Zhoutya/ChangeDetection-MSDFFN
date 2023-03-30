# -*- coding: utf-8 -*-

import os
import torch.nn as nn

import scipy.io as io

import configs.configs as cfg
import torch.optim as optim

from data.HSICD_data import HSICD_data
from data.get_train_test_set import get_train_test_set as get_set
from tools.train import train as fun_train
from tools.test import test as fun_test

# tools
from tools.show import *
from tools.assessment import *
# models
from model.MSDFFN import MSDFFN as fun_model

def main():

    current_dataset = cfg.current_dataset
    current_model = cfg.current_model
    model_name = current_dataset + current_model
    print('model {}'.format(model_name))
    cfg_data = cfg.data
    cfg_model = cfg.model
    cfg_train = cfg.train['train_model']
    cfg_optim = cfg.train['optimizer']
    cfg_test = cfg.test

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    #Data import and data set partition
    data_sets = get_set(cfg_data)
    img_gt = data_sets['img_gt']
    train_data = HSICD_data(data_sets, cfg_data['train_data'])
    test_data = HSICD_data(data_sets, cfg_data['test_data'])

    # Load model
    model = fun_model(cfg_model['in_fea_num']).to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg_optim['lr'], momentum=cfg_optim['momentum'], weight_decay=cfg_optim['weight_decay'])
    # train
    fun_train(train_data, model, loss_fun, optimizer, device, cfg_train)
    # test
    pred_train_label, pred_train_acc = fun_test(train_data, data_sets['img_gt'], model, device, cfg_test)
    pred_test_label, pred_test_acc = fun_test(test_data, data_sets['img_gt'], model, device, cfg_test)

    # Post processing
    predict_label = torch.cat([pred_train_label, pred_test_label], dim=0)
    print('pred_train_acc {:.2f}%, pred_test_acc {:.2f}%'.format(pred_train_acc, pred_test_acc))
    predict_img = Predict_Label2Img(predict_label, img_gt)

    conf_mat, oa, kappa_co, P, R, F1, acc = accuracy_assessment(img_gt, predict_img)
    assessment_result = [round(oa, 4) * 100, round(kappa_co, 4), round(F1, 4) * 100, round(P, 4) * 100,
                           round(R, 4) * 100, model_name]
    print('assessment_result', assessment_result)

    # Store
    save_folder = cfg_test['save_folder']
    save_name = cfg_test['save_name']

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    io.savemat(save_folder + '/' + save_name + ".mat",
               {"predict_img": np.array(predict_img.cpu()), "oa": assessment_result})
    imageio.imwrite(save_folder + '/' + save_name + '+predict_img.png', predict_img)
    print('save predict_img successful!')


if __name__ == '__main__':

    main()

