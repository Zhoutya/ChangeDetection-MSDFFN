
# MSDFFN (Multi-Scale Diff-changed Feature Fusion Network)


Pytorch implementation for [MSDFFN](https://ieeexplore.ieee.org/document/10032617) 《Multi-Scale Diff-changed Feature Fusion Network for Hyperspectral Image Change Detection》
```c  
F. Luo, T. Zhou, J. Liu, T. Guo, X. Gong and J. Ren, "Multiscale Diff-Changed Feature Fusion Network for Hyperspectral Image Change Detection," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-13, 2023, Art no. 5502713, doi: 10.1109/TGRS.2023.3241097.
```
### Introduction
This repository includes MSDFFN implementations in PyTorch version and partial datasets in the paper.

### Model Structure
The proposed MSDFFN for HSI CD task is composed of a temporal feature encoder-decoder (TFED) sub-network, a bidirectional diff-changed feature representation (BDFR) module and a multi-scale attention fusion (MSAF) module.

The bi-temporal HSIs are passed through the temporal feature extraction sub-network, which combines the reduced inception (RI) module and the skip layer attention (SLA) module to obtain multi-scale features. And then, the diff-changed features with fine representation power are acquired and learned by the BDFR module from these multi-scale features. After that, the MSAF module fuses the multi-scale diff-changed features adaptively with residual attention.

![flowchart](flowchart.jpg)


### Requirements
Python 3.9  <br />
PyTorch 1.10.2  <br />

### Citation
Please cite our paper if you use this code in your research.
```c  
F. Luo, T. Zhou, J. Liu, T. Guo, X. Gong and J. Ren, "Multiscale Diff-Changed Feature Fusion Network for Hyperspectral Image Change Detection," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-13, 2023, Art no. 5502713, doi: 10.1109/TGRS.2023.3241097.
```
```c
@ARTICLE{10032617,
  author={Luo, Fulin and Zhou, Tianyuan and Liu, Jiamin and Guo, Tan and Gong, Xiuwen and Ren, Jinchang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multiscale Diff-Changed Feature Fusion Network for Hyperspectral Image Change Detection}, 
  year={2023},
  volume={61},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2023.3241097}}
```

### License
Code and datasets are released for non-commercial and research purposes only. For commercial purposes, please contact the authors.
