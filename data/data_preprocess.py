import torch
import torch.nn as nn
from torchvision import transforms


def std_norm(image):  # input tensor image size with CxHxW
    image = image.permute(1, 2, 0).numpy()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.tensor(image).mean(dim=[0, 1]), torch.tensor(image).std(dim=[0, 1]))
    ])   # (x - mean(x))/std(x) normalize to mean: 0, std: 1

    return trans(image)


def one_zero_norm(image):  # input tensor image size with CxHxW
    channel, height, width = image.shape
    data = image.reshape(channel, height * width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = (data - data_min.unsqueeze(1))/(data_max.unsqueeze(1) - data_min.unsqueeze(1))
    # (x - min(x))/(max(x) - min(x))  normalize to (0, 1) for each channel

    return data.view(channel, height, width)


def pos_neg_norm(image):  # input tensor image size with CxHxW
    channel, height, width = image.shape
    data = image.reshape(channel, height*width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = -1 + 2 * (data - data_min.unsqueeze(1))/(data_max.unsqueeze(1) - data_min.unsqueeze(1))
    # -1 + 2 * (x - min(x))/(max(x) - min(x))  normalize to (-1, 1) for each channel

    return data.view(channel, height, width)


def construct_sample(img1, img2, window_size=5):
    _, height, width = img1.shape  # input float tensor image size with CxHxW
    half_window = int(window_size//2)

    # padding
    pad = nn.ReplicationPad2d(half_window)
    pad_img1 = pad(img1.unsqueeze(0)).squeeze(0)
    pad_img2 = pad(img2.unsqueeze(0)).squeeze(0)

    # get coordinates
    patch_coordinates = torch.zeros((height*width, 4), dtype=torch.long)
    t = 0
    for h in range(height):
        for w in range(width):
            patch_coordinates[t, :] = torch.tensor([h, h + window_size, w, w + window_size])
            t += 1

    return pad_img1, pad_img2, patch_coordinates


def select_sample(gt, ntr):  # input tensor data with NxCxHxW, tensor gt with HxW
    gt_vector = gt.reshape(-1, 1).squeeze(1)
    label = torch.unique(gt)

    first_time = True

    for each in range(len(label)):
        indices_vector = torch.where(gt_vector == label[each])
        indices = torch.where(gt == label[each])

        indices_vector = indices_vector[0]
        indices_row = indices[0]
        indices_column = indices[1]

        class_num = torch.tensor(len(indices_vector))
        # print('class_num', class_num)  # Farmland 0->44723, 1->18277

        # get select_num
        if ntr < 1:
            ntr0 = int(ntr*class_num)

        else:
            ntr0 = ntr

        if ntr0 < 10:
            select_num = 10

        elif ntr0 > class_num//2:
            select_num = class_num//2

        else:
            select_num = ntr0

        select_num = torch.tensor(select_num)

        # disorganize
        rand_indices0 = torch.randperm(class_num)
        rand_indices = indices_vector[rand_indices0]

        # Divide train and test
        tr_ind0 = rand_indices0[0:select_num]
        te_ind0 = rand_indices0[select_num:]
        tr_ind = rand_indices[0:select_num]
        te_ind = rand_indices[select_num:]
        # index+Sample center coordinate
        select_tr_ind = torch.cat([tr_ind.unsqueeze(1),
                                indices_row[tr_ind0].unsqueeze(1),
                                indices_column[tr_ind0].unsqueeze(1)],
                               dim=1
                               )
        select_te_ind = torch.cat([te_ind.unsqueeze(1),
                                indices_row[te_ind0].unsqueeze(1),
                                indices_column[te_ind0].unsqueeze(1)],
                               dim=1
                               )

        if first_time:
            first_time = False

            train_sample_center = select_tr_ind
            train_sample_num = select_num.unsqueeze(0)

            test_sample_center = select_te_ind
            test_sample_num = (class_num - select_num).unsqueeze(0)

        else:
            train_sample_center = torch.cat([train_sample_center, select_tr_ind], dim=0)
            train_sample_num = torch.cat([train_sample_num, select_num.unsqueeze(0)])

            test_sample_center = torch.cat([test_sample_center, select_te_ind], dim=0)
            test_sample_num = torch.cat([test_sample_num, (class_num - select_num).unsqueeze(0)])


    rand_tr_ind = torch.randperm(train_sample_num.sum())
    train_sample_center = train_sample_center[rand_tr_ind, ]   # torch.Size([22316, 3])   22316 = 不变20377+ 变化1939
    rand_te_ind = torch.randperm(test_sample_num.sum())
    test_sample_center = test_sample_center[rand_te_ind, ]   # torch.Size([89267, 3])

    data_sample = {'train_sample_center': train_sample_center, 'train_sample_num': train_sample_num,
                   'test_sample_center': test_sample_center, 'test_sample_num': test_sample_num,
                   }

    return data_sample