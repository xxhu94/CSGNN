import torch
import time
import dgl.function as fn
from utils import MixedDropout, MixedLinear
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from numpy import random
import dgl, os
import math
from utils import _set_cost_matrix,_validate_cost_matrix


class GraSen(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes, edges, activation=None, step_size=0.02):
        super(GraSen, self).__init__()

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
        # cvg is the flag for stopping reinforcement learning
        self.cvg = {}
        for etype in edges:
            self.p[etype] = 0.5
            self.last_avg_dist[etype] = 0
            self.f[etype] = []
            self.cvg[etype] = False

    def _calc_distance(self, edges):
        # Node feature transformation and distance measure
        # Eq.(5)
        d = torch.norm(torch.tanh(self.MLP(edges.src['h'])) - torch.tanh(self.MLP(edges.dst['h'])), 1, 1)
        return {'d': d}

    def _top_p_sampling(self, g, p):
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

                graph.send_and_recv(sampled_edges, fn.copy_u('h', 'm'), fn.mean('m', 'h_%s' % etype[1]), etype=etype)
                hr[etype] = graph.ndata['h_%s' % etype[1]]
                if self.activation is not None:
                    hr[etype] = self.activation(hr[etype])

            # Eq.(12)
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
        # Eq.(4)
        sim = torch.tanh(self.layers[0].MLP(feat))

        # Forward of n layers of CSGNN
        for layer in self.layers:
            feat = layer(graph, feat)

        return feat, sim

    def RLModule(self, graph, epoch, idx):
        for layer in self.layers:
            for etype in self.edges:
                if not layer.cvg[etype]:
                    # Eq.(8)
                    eid = graph.in_edges(idx, form='eid', etype=etype)
                    avg_dist = torch.mean(layer.dist[etype][eid.long()])

                    # Eq.(10)
                    if layer.last_avg_dist[etype] < avg_dist:
                        if layer.p[etype] - self.step_size > 0:
                            layer.p[etype] -= self.step_size
                        layer.f[etype].append(-1)
                    else:
                        if layer.p[etype] + self.step_size <= 1:
                            layer.p[etype] += self.step_size
                        layer.f[etype].append(+1)
                    layer.last_avg_dist[etype] = avg_dist

                    # Eq.(11)
                    if epoch >= 9 and abs(sum(layer.f[etype][-10:])) <= 2:
                        layer.cvg[etype] = True


class CostMatrix(nn.Module):
    def __init__(self, num_classes,labels_train,n_classes):
        super(CostMatrix, self).__init__()
        # log1p initialization, Eq.(21)
        how_dic = {0: 'uniform', 1: 'inverse', 2: 'log1p-inverse'}
        pmatrix = _set_cost_matrix(labels_train, how=how_dic[2])
        cost_matrix = _validate_cost_matrix(pmatrix, n_classes)
        self.cost_matrix = nn.Parameter(torch.Tensor(cost_matrix))

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
            table = np.vstack((table, np.array([x, y, value])))

        return pd.DataFrame(table, columns=['row', 'column', 'cost'])

    # Calculate cost-sensitive embedding,return normalized Tensor , Eq.(13)
    def cost_logits(self, feat: torch.tensor, labels: torch.tensor, cost_table: pd.DataFrame):
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
