import os
import pdb
import torch
import torch.nn as nn

from .single_classifier import Classifier as LinearClassifier


class MulClassifier(nn.Module):
    def __init__(self, opts):
        super(MulClassifier, self).__init__()
        self.classifiers = nn.ModuleList([LinearClassifier(opt) for opt in opts])
        self.num_classifiers = len(opts)

    def forward(self, feats):
        assert(len(feats) == self.num_classifiers)
        return [self.classifiers[i](feat) for i, feat in enumerate(feats)]
