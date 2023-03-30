'''
 Dataset Source:
    Farmland: http://crabwq.github.io/
    River: https://share.weiyun.com/5ugrczK
    Hermiston: https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset
'''

from scipy.io import loadmat

def get_Farmland_dataset():
    data_set_before = loadmat(r'../../datasets/Yancheng/farm06.mat')['imgh']
    data_set_after = loadmat(r'../../datasets/Yancheng/farm07.mat')['imghl']
    ground_truth = loadmat(r'../../datasets/Yancheng/label.mat')['label']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt


def get_dataset(current_dataset):
    if current_dataset == 'Farmland':
        return get_Farmland_dataset()   # Farmland(450, 140, 155), gt[0. 1.]

