import torch
import time
import datetime
import math
import os

from torch.utils.data import DataLoader


def adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, step_index):
    # Adjust the learning rate in stages
    if epoch < 1:
        lr = 0.0001 * lr_init
    elif epoch <= step_index[0]:
        lr = lr_init
    elif epoch <= step_index[1]:
        lr = lr_init * lr_gamma
    elif epoch > step_index[1]:
        lr = lr_init * lr_gamma ** 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train(train_data, model, loss_fun, optimizer, device, cfg):
    torch.autograd.set_detect_anomaly(True)
    num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']

    save_folder = cfg['save_folder']
    save_name = cfg['save_name']

    lr_init = cfg['lr']
    lr_gamma = cfg['lr_gamma']
    lr_step = cfg['lr_step']
    lr_adjust = cfg['lr_adjust']

    epoch_size = cfg['epoch']
    batch_size = cfg['batch_size']

    # gpu_num
    if gpu_num > 1 and cfg['gpu_train']:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    '''# Load the model and start training'''
    model.train()

    if cfg['reuse_model']:
        print('load model...')
        checkpoint = torch.load(cfg['reuse_file'], map_location=device)
        start_epoch = checkpoint['epoch']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        start_epoch = 0

    batch_num = math.ceil(len(train_data) / batch_size)
    train_loss_save = []
    train_acc_save = []

    print('start training...')

    for epoch in range(start_epoch + 1, epoch_size + 1):

        epoch_time0 = time.time()
        epoch_loss = 0
        predict_correct = 0
        label_num = 0

        batch_data = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        if lr_adjust:
            lr = adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, lr_step)
        else:
            lr = lr_init

        for batch_idx, batch_sample in enumerate(batch_data):

            iteration = (epoch - 1) * batch_num + batch_idx + 1
            batch_time0 = time.time()

            img1, img2, target, indices = batch_sample
            img1 = img1.to(device)
            img2 = img2.to(device)
            target = target.to(device)

            prediction = model(img1, img2)
            loss = loss_fun(prediction, target.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time1 = time.time()
            batch_time = batch_time1 - batch_time0
            batch_eta = batch_time * (batch_num - batch_idx)
            epoch_eta = int(batch_time * (epoch_size - epoch) * batch_num + batch_eta)

            epoch_loss += loss.item()
            predict_label = prediction.detach().argmax(dim=1, keepdim=True)

            predict_correct += predict_label.eq(target.view_as(predict_label)).sum().item()
            label_num += len(target)

        train_acc = 100 * predict_correct/label_num
        epoch_time1 = time.time()
        epoch_time = epoch_time1 - epoch_time0
        epoch_eta = int(epoch_time * (epoch_size - epoch))

        print('Epoch: {}/{} || lr: {} || loss: {} || Train acc: {:.2f}% || '
              'Epoch time: {:.4f}s || Epoch ETA: {}'
              .format(epoch, epoch_size, lr, epoch_loss/batch_num, train_acc,
                      epoch_time, str(datetime.timedelta(seconds=epoch_eta))
                      )
              )

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        train_loss_save.append(epoch_loss / batch_num)
        train_acc_save.append(train_acc)

    # Store the final model
    save_model = dict(
        model=model.state_dict(),
        epoch=epoch_size
    )
    torch.save(save_model, os.path.join(save_folder, save_name + '_Final.pth'))


