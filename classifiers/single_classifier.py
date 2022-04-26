import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1) 

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        nchannels = opt['nchannels']
        num_class = opt['num_class']
        out_feat_size = opt['out_feat_size']
        pool_type = opt['pool_type']
        nchannels_after_pool = nchannels*out_feat_size*out_feat_size

        self.classifier = nn.Sequential()
        if pool_type == 'max':
            self.classifier.add_module('MaxPool', nn.AdaptiveMaxPool2d((out_feat_size, out_feat_size)))
        if pool_type == 'avg':
            self.classifier.add_module('AvgPool', nn.AdaptiveAvgPool2d((out_feat_size, out_feat_size)))

        # self.classifier.add_module('BatchNorm', nn.BatchNorm2d(nchannels, affine=True))
        self.classifier.add_module('Flatten', Flatten())
        self.classifier.add_module('LinearClassifier', nn.Linear(nchannels_after_pool, num_class))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # keep same with ori moco
                m.weight.data.normal_(mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                #nn.init.constant_(m.weight, 1)
                #nn.init.constant_(m.bias, 0)
                m.weight.data.uniform_()
                m.bias.data.zero_()


    def forward(self, feat):
        return self.classifier(feat)
