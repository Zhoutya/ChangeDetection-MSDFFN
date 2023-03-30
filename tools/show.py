import torch

def Predict_Label2Img(predict_label, img_gt):
    predict_img = torch.zeros_like(img_gt)
    num = predict_label.shape[0]   # 111583

    for i in range(num):
        x = int(predict_label[i][1])
        y = int(predict_label[i][2])
        l = predict_label[i][3]
        predict_img[x][y] = l

    return predict_img



