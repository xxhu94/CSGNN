import torch
import time
from dgl.nn import GATConv
import dgl.function as fn
from utils import MixedDropout, MixedLinear
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from numpy import random
import dgl, os
import math
from utils import _set_cost_matrix,cost_table_calc,_validate_cost_matrix



class GAT_COBO(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 dropout,
                 dropout_adj,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT_COBO, self).__init__()
        # MixedLinear
        fcs = [MixedLinear(in_dim, num_hidden, bias=False)]
        fcs.append(nn.Linear(num_hidden, num_classes, bias=False))

        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())
        if dropout is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout)
        if dropout_adj is 0:
            self.dropout_adj = lambda x: x
        else:
            self.dropout_adj = MixedDropout(dropout_adj)
        self.act_fn = nn.ReLU()

        # GAT-based weak-classifier
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=False))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=False))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def transform_features(self, x):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(x)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.act_fn(self.fcs[-1](self.dropout_adj(layer_inner)))
        return res

    def forward(self, inputs):
        logits_inter_GAT = self.transform_features(inputs)
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        logits_inner_GAT, attention = self.gat_layers[-1](self.g, h, True)
        return logits_inter_GAT, logits_inner_GAT, attention


class GraSen(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes, edges, activation=None, step_size=0.02):
        super(GraSen, self).__init__()
        # 未来可以考虑替换成mixed linear
        # MixedLinear
        # fcs = [MixedLinear(in_dim, num_hidden, bias=False)]
        # fcs.append(nn.Linear(num_hidden, num_classes, bias=False))
        #
        # self.fcs = nn.ModuleList(fcs)
        # self.reg_params = list(self.fcs[0].parameters())
        # if dropout is 0:
        #     self.dropout = lambda x: x
        # else:
        #     self.dropout = MixedDropout(dropout)
        # if dropout_adj is 0:
        #     self.dropout_adj = lambda x: x
        # else:
        #     self.dropout_adj = MixedDropout(dropout_adj)
        # self.act_fn = nn.ReLU()

        self.activation = activation
        self.step_size = step_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.edges = edges
        self.dist = {}

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.MLP = nn.Linear(self.in_dim, self.num_classes)

        self.p = {}
        self.last_avg_dist = {}
        self.f = {}
        self.cvg = {}
        for etype in edges:
            self.p[etype] = 0.5
            self.last_avg_dist[etype] = 0
            self.f[etype] = []
            # cvg用于停止强化学习的标识
            self.cvg[etype] = False

    def _calc_distance(self, edges):
        # formula 2
        d = torch.norm(torch.tanh(self.MLP(edges.src['h'])) - torch.tanh(self.MLP(edges.dst['h'])), 1, 1)
        return {'d': d}

    def _top_p_sampling(self, g, p):
        # this implementation is low efficient
        # optimization requires dgl.sampling.select_top_p requested in issue #3100
        dist = g.edata['d']
        neigh_list = []
        for node in g.nodes():
            edges = g.in_edges(node, form='eid').long()
            num_neigh = torch.ceil(g.in_degrees(node) * p).int().item()
            neigh_dist = dist[edges]
            if neigh_dist.shape[0] > num_neigh:
                neigh_index = np.argpartition(neigh_dist.cpu().detach(), num_neigh)[:num_neigh]
            else:
                neigh_index = np.arange(num_neigh)
            neigh_list.append(edges[neigh_index])
        return torch.cat(neigh_list)

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata['h'] = feat

            hr = {}
            for i, etype in enumerate(graph.canonical_etypes):
                graph.apply_edges(self._calc_distance, etype=etype)
                self.dist[etype] = graph.edges[etype].data['d']
                sampled_edges = self._top_p_sampling(graph[etype], self.p[etype]).int()

                # formula 8
                graph.send_and_recv(sampled_edges, fn.copy_u('h', 'm'), fn.mean('m', 'h_%s' % etype[1]), etype=etype)
                hr[etype] = graph.ndata['h_%s' % etype[1]]
                if self.activation is not None:
                    hr[etype] = self.activation(hr[etype])

            # formula 9 using mean as inter-relation aggregator
            p_tensor = torch.Tensor(list(self.p.values())).view(-1, 1, 1).to(graph.device)
            h_homo = torch.sum(torch.stack(list(hr.values())) * p_tensor, dim=0)
            h_homo += feat
            if self.activation is not None:
                h_homo = self.activation(h_homo)

            return self.linear(h_homo)


class SENGNN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_classes,
                 hid_dim=64,
                 edges=None,
                 num_layers=2,
                 activation=None,
                 step_size=0.02):
        super(SENGNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.edges = edges
        self.activation = activation
        self.step_size = step_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        if self.num_layers == 1:
            # Single layer
            self.layers.append(GraSen(self.in_dim,
                                        self.num_classes,
                                        self.num_classes,
                                        self.edges,
                                        activation=self.activation,
                                        step_size=self.step_size))

        else:
            # Input layer
            self.layers.append(GraSen(self.in_dim,
                                        self.hid_dim,
                                        self.num_classes,
                                        self.edges,
                                        activation=self.activation,
                                        step_size=self.step_size))

            # Hidden layers with n - 2 layers
            for i in range(self.num_layers - 2):
                self.layers.append(GraSen(self.hid_dim,
                                            self.hid_dim,
                                            self.num_classes,
                                            self.edges,
                                            activation=self.activation,
                                            step_size=self.step_size))

            # Output layer
            self.layers.append(GraSen(self.hid_dim,
                                        self.num_classes,
                                        self.num_classes,
                                        self.edges,
                                        activation=self.activation,
                                        step_size=self.step_size))

    def forward(self, graph, feat):
        # For full graph training, directly use the graph
        # formula 4
        sim = torch.tanh(self.layers[0].MLP(feat))

        # Forward of n layers of CARE-GNN
        for layer in self.layers:
            feat = layer(graph, feat)

        return feat, sim

    # 后续去掉多边类型
    def RLModule(self, graph, epoch, idx):
        for layer in self.layers:
            for etype in self.edges:
                if not layer.cvg[etype]:
                    # formula 5
                    eid = graph.in_edges(idx, form='eid', etype=etype)
                    avg_dist = torch.mean(layer.dist[etype][eid.long()])

                    # formula 6
                    if layer.last_avg_dist[etype] < avg_dist:
                        if layer.p[etype] - self.step_size > 0:
                            layer.p[etype] -= self.step_size
                        layer.f[etype].append(-1)
                    else:
                        if layer.p[etype] + self.step_size <= 1:
                            layer.p[etype] += self.step_size
                        layer.f[etype].append(+1)
                    layer.last_avg_dist[etype] = avg_dist

                    # formula 7
                    if epoch >= 9 and abs(sum(layer.f[etype][-10:])) <= 2:
                        layer.cvg[etype] = True


class CostMatrix(nn.Module):
    def __init__(self, num_classes,labels_train,n_classes):
        super(CostMatrix, self).__init__()

        # 随机初始化值
        # self.cost_matrix = nn.Parameter(torch.rand(num_classes, num_classes, dtype=torch.long))

        # 全1初始化
        # self.cost_matrix = nn.Parameter(torch.ones(num_classes, num_classes, dtype=torch.long).float())

        # log1p初始化
        how_dic = {0: 'uniform', 1: 'inverse', 2: 'log1p-inverse'}
        pmatrix = _set_cost_matrix(labels_train, how=how_dic[2])
        cost_matrix = _validate_cost_matrix(pmatrix, n_classes)
        self.cost_matrix = nn.Parameter(torch.Tensor(cost_matrix))
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
    #     nn.init.kaiming_uniform_(self.cost_matrix, mode='fan_out', a=math.sqrt(5))
    #     # if self.bias is not None:
    #     #     _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.cost_matrix)
    #     #     bound = 1 / math.sqrt(fan_out)
    #     #     nn.init.uniform_(self.bias, -bound, bound)

    def cost_table_calc(self):
        """Creates a table of values from the cost matrix.
        Write the matrix form cost matrix in the form of coordinates + cost value

        Parameters
        ----------
        cost_matrix : array-like of shape = [n_classes, n_classes]

        Returns
        -------
        df : dataframe of shape = [n_classes * n_classes, 3]

        """
        table = np.empty((0, 3))

        for (x, y), value in np.ndenumerate(self.cost_matrix.detach().numpy()):
            # table = np.vstack((table, np.array([x + 1, y + 1, value])))
            table = np.vstack((table, np.array([x, y, value])))

        return pd.DataFrame(table, columns=['row', 'column', 'cost'])

    # 输出经过代价矩阵乘积+softmax的嵌入
    def cost_logits(self, feat: torch.tensor, labels: torch.tensor, cost_table: pd.DataFrame):
        # calculate cost*exp(o_n),return normalized Tensor
        dic = {}
        cost = {}
        df = pd.DataFrame()
        for i in range(0, feat.shape[1]):
            dic[i] = pd.DataFrame({'row': labels.cpu(), 'column': i * np.ones(len(labels))})
            cost[i] = dic[i].merge(cost_table, how='left', on=['row', 'column'])
            df.insert(i, i, cost[i]['cost'])
        logits=torch.mul(torch.from_numpy(df.values),torch.exp(feat.cpu()))
        # normalize by row sum
        logits=logits.div(logits.sum(dim=1).repeat(feat.shape[1], 1).T)
        return logits
