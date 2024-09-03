


import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv ,CFConv ,GATv2Conv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.Sequential(
            SAGEConv(in_size, hid_size, aggregator_type='gcn', feat_drop=0.32, activation=F.elu),
            SAGEConv(hid_size, hid_size, aggregator_type='gcn', feat_drop=0.32, activation=F.elu)

        )
        self.drop = nn.Dropout(0.32)
    
    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.drop(h)
            h = layer(g, h)
        return h

class CFC(nn.Module):
    def __init__(self, node_in_feats , edge_in_feats , hidden_feats , out_feats):
        super().__init__()
        self.layers = nn.Sequential(
            CFConv(node_in_feats, edge_in_feats, hidden_feats, out_feats),
            CFConv(node_in_feats, edge_in_feats, hidden_feats, out_feats)

        )
        self.drop = nn.Dropout(0.32)
    
    def forward(self ,g, node_feats, edge_feats):
        h = node_feats
        e = edge_feats
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.drop(h)
                g = self.drop(e)
                h , e = layer(g ,h, e)
        return h , e





"""-----------------------------------------------------------------------"""
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size,num_head):
        super().__init__()
        self.layers = nn.Sequential(
            GATv2Conv(in_size, hid_size,num_head),
            GATv2Conv(hid_size, out_size,num_head)

        )
        self.drop = nn.Dropout(0.32)
    
    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.drop(h)
            h = layer(g, h)
        return h



"""-------------------------------------------------------"""









