#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP_for_GIN
import time


class GINConvolution(nn.Module):
    def __init__(self, input_dim, \
                 output_dim, \
                 support, \
                 hidden_dim = 64,\
                 act_func=None, \
                 featureless=False, \
                 dropout_rate=0.5, \
                 bias=False, num_mlp_layers=2, eps=-1,train_eps=False):
        super(GINConvolution, self).__init__()
        self.support = support
        self.featureless = featureless
        self.num_mlp_layers = num_mlp_layers
        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))
        self.dropout = nn.Dropout(dropout_rate)
        print('dropout_rate ', dropout_rate)

        self.nn = MLP_for_GIN(num_mlp_layers, input_dim, hidden_dim, output_dim)

        if train_eps:
            eps = 0.0001
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.eps = eps

    def forward(self, x):
        for i in range(len(self.support)):
            if self.featureless:
                if i ==0:
                    AX = self.support[i]
                else:
                    AX += self.support[i]
            else:
                x = self.dropout(x)
                if i == 0:
                    AX = self.support[i].mm(x)
                else:
                    AX += self.support[i].mm(x)

        # out = AX.mm(self.w)
        if self.featureless:
            out = self.nn(AX)
            # out = AX.mm(self.w)+0*(1+self.eps)*self.w
        else:
            out = self.nn(AX)
            # out = (AX + 0*(1+self.eps) * x).mm(self.w)
        # else:
        #     out = self.nn(AX * x)
        #     # out = self.nn(AX+0*(1+self.eps)*x)
        self.embedding = out
        return out


class GIN(nn.Module):
    def __init__(self, input_dim, \
                 support, \
                 dropout_rate=0.5, \
                 num_classes=10):
        super(GIN, self).__init__()
        self.num_layers = 4
        self.embed_dim = 64
        hidden_dim_MLP = 64
        num_mlp_layers = 2
        self.dropout_rate = 0
        train_eps = True
        # GraphConvolution
        self.layers = nn.ModuleList()

        for i in range(self.num_layers-1):
            if i == 0:
                self.layers.append(GINConvolution(input_dim, self.embed_dim, support,hidden_dim=hidden_dim_MLP, featureless=True, train_eps=train_eps,num_mlp_layers=1))
            else:
                self.layers.append(
                    GINConvolution(self.embed_dim, self.embed_dim, support, hidden_dim=hidden_dim_MLP, train_eps=train_eps, num_mlp_layers=num_mlp_layers))
        self.layers.append(GINConvolution(self.embed_dim, num_classes, support, hidden_dim=hidden_dim_MLP, train_eps=train_eps, num_mlp_layers=2))


        self.classifier = nn.ModuleList()
        for i in range(self.num_layers):
            self.classifier.append(nn.Linear(self.embed_dim,num_classes))

    def forward(self, x):
        hidden_rep = []
        h = x
        for i in range(0, self.num_layers-1):
            h = F.relu(self.layers[i](h))
            hidden_rep.append(h)
        self.layer1 = self.layers[0]
        out = self.layers[-1](h)
        return out
