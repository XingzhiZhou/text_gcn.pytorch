#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP_for_GIN


class GINConvolution(nn.Module):
    def __init__(self, input_dim, \
                 output_dim, \
                 support, \
                 hidden_dim = 64,\
                 act_func=None, \
                 featureless=False, \
                 dropout_rate=0.5, \
                 num_mlp_layers=2, eps = 0,train_eps=False):
        super(GINConvolution, self).__init__()
        self.support = support
        self.featureless = featureless
        self.num_mlp_layers = num_mlp_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.train_eps = train_eps
        self.nn = MLP_for_GIN(num_mlp_layers, input_dim, hidden_dim, output_dim)

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))

    def forward(self, x):
        for i in range(len(self.support)):
            if self.featureless:
                if i ==0:
                    AX = self.support[i]
                else:
                    AX += self.support[i]
            else:
                # x = self.dropout(x)
                if i == 0:
                    AX = self.support[i].mm(x)
                else:
                    AX += self.support[i].mm(x)

        if self.featureless:
            out = self.nn(AX)
        else:
            if self.train_eps:
                out = self.nn(AX+(1+self.eps)*x)
            else:
                out = self.nn(AX)
        self.embedding = out
        return out


class GIN(nn.Module):
    def __init__(self, input_dim, \
                 support, \
                 dropout_rate = 0.5, \
                 num_classes = 10,
                 num_layers = 2,
                 num_mlp_layers = 1,
                 embed_dim = 200,
                 hidden_dim_mlp = 200,
                 train_eps = False):
        super(GIN, self).__init__()
        self.num_layers = num_layers

        # GraphConvolution
        self.layers = nn.ModuleList()

        for i in range(num_layers-1):
            if i == 0:
                self.layers.append(
                    GINConvolution(input_dim,                                                         embed_dim,
                                   support,
                                   hidden_dim=hidden_dim_mlp,
                                   featureless=True,
                                   train_eps=train_eps,                                               num_mlp_layers=num_mlp_layers))
            else:
                self.layers.append(
                    GINConvolution(embed_dim,
                                   embed_dim,
                                   support,
                                   hidden_dim=hidden_dim_mlp,
                                   train_eps=train_eps,
                                   num_mlp_layers=num_mlp_layers))


        self.layers.append(
            GINConvolution(embed_dim,
                           num_classes,
                           support,
                           hidden_dim=hidden_dim_mlp,
                           train_eps=train_eps,
                           num_mlp_layers=num_mlp_layers))

        self.layer1 = self.layers[0]  ## ignore this. This is to incorporate the original code
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = x
        for i in range(0, self.num_layers-1):
            h = F.relu(self.layers[i](h))
        out = self.layers[-1](h)
        return out
