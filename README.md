# Bag of Instances Aggregation Boosts Self-supervised Learning

Official implementation of the paper [Bag of Instances Aggregation Boosts Self-supervised Learning](https://arxiv.org/abs/2107.01691).

Recent advances in self-supervised learning have experienced remarkable progress, especially for contrastive learning based methods, which regard each image as well as its augmentations as an individual class and try to distinguish them from all other images. However, due to the large quantity of exemplars, this kind of pretext task intrinsically suffers from slow convergence and is hard for optimization. This is especially true for small scale models, which we find the performance drops dramatically comparing with its supervised counterpart. 

In this paper, we propose a simple but effective distillation strategy for unsupervised learning. The highlight is that the relationship among similar samples counts and can be seamlessly transferred to the student to boost the performance. Our method, termed as BINGO, which is short for **B**ag of **I**nsta**N**ces a**G**gregati**O**n, targets at transferring the relationship learned by the teacher to the student. Here bag of instances indicates a set of similar samples constructed by the teacher and are grouped within a bag, and the goal of distillation is to aggregate compact representations over the student with respect to instances in a bag. Notably, BINGO achieves new state-of-the-art performance on small scale models, *i.e.*, 65.5% and 68.9% top-1 accuracies with linear evaluation on ImageNet, using ResNet-18 and ResNet-34 as backbone, respectively, surpassing baselines (52.5% and 57.4% top-1 accuracies) by a significant margin.

![framework](./imgs/framework.png)

## Noting
* Code will be released in the future.

## Performance
**Linear evaluation accuracy on ImageNet**
![imagenet](./imgs/imagenet.png)

**Semi-supervised learning on ImageNet with ResNet-18**
![semi](./imgs/semi.png)

## Citation
If you find this repository/work helpful in your research, welcome to cite the paper.
```
@article{xu2021bingo,
    title={Bag of Instances Aggregation Boosts Self-supervised Learning}, 
    author={Haohang Xu and Jiemin Fang and Xiaopeng Zhang and Lingxi Xie and Xinggang Wang and Wenrui Dai and Hongkai Xiong and Qi Tian},
    journal={arXiv:2107.01691},
    year={2021}
}