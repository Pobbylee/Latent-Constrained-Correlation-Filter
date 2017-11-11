Latent-Constrained-Correlation-Filter
========
Introduction
-------
This is the research code for the paper: Latent Constrained Correlation Filter，which has been accepted by IEEE Transcation on Image Processing recently. In this paper we propose a new method, named latent constrained correlation filters (LCCF), by mapping the correlation filters to a given latent subspace, and develop a new learning framework in the latent subspace that embeds distribution-related constraints into the original problem. To solve the optimization problem, we introduce a subspace based alternating direction method of multipliers (SADMM), which is proven to converge at the saddle point. Our approach is successfully applied to three different tasks, including eye localization, car detection and object tracking. Extensive experiments demonstrate that LCCF achieve state-of-the-art.

![idea](https://raw.githubusercontent.com/bczhangbczhang/Latent-Constrained-Correlation-Filter/master/idea.jpg)

Run
-------
**1.** To run this code，you should download OTB-50 dataset first:

(1)You can just download it on http://pan.baidu.com/s/1pKTljWR without additional operations.

(2)Or you can download OTB-50 by runing 'download_videos.m', but remember to make a copy of the 'jogging' folder into double, and rename it as 'Jogging-1' and 'Jogging-2' respectively. The label file 'groundtruth_rect.1.txt' and 'groundtruth_rect.2.txt' should be renamed as 'groundtruth_rect.txt'

**2.** We provide the test code for LCCF on gary feature, hog feature and a deep feature, which have been compressed in two documents respectively. According to our experiments，LCCF obtained accuracy improvement on all of these features.

Feature | Gray    | Hog       | VGG-19|
--------|:-------:|:---------:|:---------:
KCF     | 56.1%   | 74.0%     | 89.1%
LC_KCF  | 56.9%   | 79.4%     | 89.6%

**3.** To use it, just start with 'run_tracker.m'

Citation
-------
If you find these code useful, please consider to cite our paper：
```bibtex
@article{Zhang2017Latent,
  title={Latent Constrained Correlation Filter},
  author={Baochang Zhang, Shangzhen Luan, Chen Chen, Jungong Han, Wei Wang, Alessandro Perina and Ling Shao},
  journal={IEEE Transactions on Image Processing},
  year={2017},
}
```
Acknowledgements
-------
[Chao Ma](https://sites.google.com/site/jbhuang0604/publications/cf2) "Hierarchical Convolutional Features for Visual Tracking", ICCV 2015

[João F. Henriques](http://www.isr.uc.pt/~henriques/circulant/)，“High-Speed Tracking with Kernelized Correlation Filters“, IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015

Contact
-------
Baochang Zhang

bczhang@buaa.edu.cn
