#!/usr/bin/env python
import torch
import torch.nn as nn
from models.mlp import MLP_for_GIN
import time


class GINConvolution(nn.Module):
    def __init__(self, input_dim, \
                 output_dim, \
                 support, \
                 hidden_dim = 64,\
                 act_func=None, \
                 featureless=False, \
                 dropout_rate=0., \
                 bias=False, num_mlp_layers=2, eps=0,train_eps=False):
        super(GINConvolution, self).__init__()
        self.support = support
        self.featureless = featureless

        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

        self.nn = MLP_for_GIN(num_mlp_layers, input_dim, hidden_dim, output_dim)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))

    def forward(self, x):
        start = time.time()
        x = self.dropout(x)

        # for i in range(len(self.support)):
        #     if self.featureless:
        #         pre_sup = getattr(self, 'W{}'.format(i))
        #     else:
        #         pre_sup = x.mm(getattr(self, 'W{}'.format(i)))
        #
        #     if i == 0:
        #         out = self.support[i].mm(pre_sup)
        #     else:
        #         out += self.support[i].mm(pre_sup)
        #


        for i in range(len(self.support)):
            if self.featureless:
                if i ==0:
                    AX = self.support[i]
                else:
                    AX+= self.support[i]
            else:
                if i == 0:
                    AX = self.support[i].mm(x)
                else:
                    AX += self.support[i].mm(x)
        print('Time comsuming for AX: %.2f s' % (start - time.time()))
        batch_size = 128
        if self.featureless:
            out = self.nn(AX)
        else:
            out = self.nn(AX+(1+self.eps)*x)
        # out=AX.mm(getattr(self, 'W{}'.format(0)))

        if self.act_func is not None:
            out = self.act_func(out)
        self.embedding = out
        print('Time comsuming: %.2f s'%(start-time.time()))
        return out


class GIN(nn.Module):
    def __init__(self, input_dim, \
                 support, \
                 dropout_rate=0., \
                 num_classes=10):
        super(GIN, self).__init__()

        # GraphConvolution
        self.layer1 = GINConvolution(input_dim, 64, support, act_func=nn.ReLU(), featureless=True,
                                       dropout_rate=dropout_rate,train_eps=True)
        self.layer2 = GINConvolution(64, num_classes, support, dropout_rate=dropout_rate, train_eps=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
