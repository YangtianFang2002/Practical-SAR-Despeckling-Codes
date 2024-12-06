# Deep Learning Codes for SAR Despeckling
Recent Practical Code Implementations for SAR Intensity Image Despeckling Algorithms

## 📰 News
- [2024-11-15] Our works "[Contrastive Learning for SAR Despeckling](https://www.sciencedirect.com/science/article/pii/S0924271624004118)" has been accepted by **ISPRS Journal of Photogrammetry and Remote Sensing**, [Code](https://github.com/YangtianFang2002/CL-SAR-Despeckling)

## 🎈 Code for Practical SAR Intensity Image Despeckling
- We integrate open-sourced codes that are compatible with single SAR intensity image despeckling. All deep-learning based codes have corresponding training scripts for reproductibility.
- Some deep-learning methods are re-implemented by PyTorch carefully according to their original papers, marked as `PyTorch Re`. The training configs are stored in the manner of [BasicSR](https://github.com/XPixelGroup/BasicSR) in the [options](options) folder, all of whose dependencies (`archs`, `dataset`, etc.) are implemented in the [basicsr](basicsr) folder and can be run easily. The dataset for the re-implementation is available at [CL-SAR-Despeckling](https://github.com/YangtianFang2002/CL-SAR-Despeckling).
  - e.g. `python basicsr/train.py -opt options/SARUSE-t1.yml`

| Method             | Code                                                         | Paper                                                        | Conference / Journal                                         | Paradigm                  | Dataset                                                      | Year |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------- | ------------------------------------------------------------ | ---- |
| CL-SAR-Despeckling | [PyTorch](https://github.com/YangtianFang2002/CL-SAR-Despeckling) | [Link](https://www.sciencedirect.com/science/article/pii/S0924271624004118) | ISPRS                                                        | Unsupervised + Supervised | [Real SAR Images](https://github.com/guan-jianjun/SAR-despeckle)(Public) + [Synthetic](https://ieeexplore.ieee.org/abstract/document/7907303) | 2024 |
| SAR-USE            | [PyTorch Re](options/SARUSE-t1.yml)                          | [Link](https://ieeexplore.ieee.org/abstract/document/10005116) | TGRS                                                         | Unsupervised + Supervised | Synthetic (Waterloo, UC Merced land-use, BSD300)             | 2023 |
| Speckle2Void       | [TensorFlow](https://github.com/diegovalsesia/speckle2void), [PyTorch Re](options/Speckle2Void-t1.yml) | [Link](https://ieeexplore.ieee.org/abstract/document/9383788) | TGRS                                                         | Unsupervised              | Real SAR Images                                              | 2021 |
| MRDDANet           | [PyTorch Re](options/MRDDANet-t2.yml)                        | [Link](https://ieeexplore.ieee.org/abstract/document/9526864) | TGRS                                                         | Supervised                | Synthetic                                                    | 2021 |
| SAR-CAM            | [PyTorch](https://github.com/JK-the-Ko/SAR-CAM), [PyTorch Re](options/SARCAM-t1.yml) | [Link](https://ieeexplore.ieee.org/abstract/document/9633208) | JSTARS                                                       | Supervised                | Synthetic                                                    | 2021 |
| MONet              | [PyTorch Re](options/MONet-t2.yml)                           | [Link](https://ieeexplore.ieee.org/abstract/document/9261137) | TGRS                                                         | Supervised                | Synthetic                                                    | 2020 |
| NR-SAR-DL          | [TensorFlow](https://github.com/ZX-HUB561/NR-SAR-DL/tree/master) | [Link](https://ieeexplore.ieee.org/abstract/document/9091002) | TGRS                                                         | Supervised                | Private Multitemporal  Sentinel-1 Real SAR Images            | 2020 |
| SAR-CNN            | [PyTorch](https://github.com/grip-unina/SAR-CNN), [PyTorch Re](options/SARCNN-t3.yml) | [Link](https://ieeexplore.ieee.org/abstract/document/8053792) | [IEEE Signal Processing Letters](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=97) | Supervised                | Synthetic                                                    | 2017 |
| SAR-BM3D           | [Matlab](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/) | [Link](https://ieeexplore.ieee.org/abstract/document/5989862) | TGRS                                                         | Numerical                 | /                                                            | 2011 |
| PPB                | [Matlab](https://www.charles-deledalle.fr/pages/ppb.php#download) | [Link](https://ieeexplore.ieee.org/abstract/document/5196737) | TIP                                                          | Numerical                 | /                                                            | 2009 |

## 🧱 Dataset for SAR Despeckling
- [Awesome-Remote-Sensing-Dataset](https://github.com/IenLong/Awesome-Remote-Sensing-Dataset): [SEN1-2](https://mediatum.ub.tum.de/1436631)
- [Training Dataset for PolSAR Despeckling with an Hybrid Approach](https://github.com/impress-parthenope/PolSAR-despeckling-with-an-hybrid-approach/releases/tag/data)
- [GCOANet: A Gradient Consistency Constraints Semi-Supervised Network for Optical Image-Assisted SAR despeckling](https://github.com/yangyang12318/MBSD-CR)
- [SAR Image Despeckling Using a Convolutional Neural](https://www.kaggle.com/code/javidtheimmortal/sar-image-despeckling-using-a-convolutional-neural/data)
- [Supervised Deep Learning Training Dataset and Network for SAR despeckling](https://github.com/guan-jianjun/SAR-despeckle)

## ✨ Citation
If you find our work helpful in your research, please kindly consider citing our works. We appreciate your support! 😊

```
@article{FANG2024376,
title = {Contrastive learning for real SAR image despeckling},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {218},
pages = {376-391},
year = {2024},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2024.11.003},
url = {https://www.sciencedirect.com/science/article/pii/S0924271624004118},
author = {Yangtian Fang and Rui Liu and Yini Peng and Jianjun Guan and Duidui Li and Xin Tian},
keywords = {Real SAR despeckling, Self-supervised learning, Contrastive learning, Multi-scale despeckling network, Excitation aggregation pooling},
}

@article{10368288,
  author={Guan, Jianjun and Liu, Rui and Tian, Xin and Tang, Xinming and Li, Song},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Robust SAR Image Despeckling by Deep Learning From Near-Real Datasets}, 
  year={2024},
  volume={17},
  number={},
  pages={2963-2979},
  keywords={Speckle;Radar polarimetry;Microwave filters;Filtering algorithms;Distortion;Noise reduction;Deep learning;Despecked;Generalized likelihood ratio test (GLRT);near-real synthetic aperture radar (SAR) datasets;phase-guided deep despeckling Network (PGD2Net)},
  doi={10.1109/JSTARS.2023.3345538}}
```
