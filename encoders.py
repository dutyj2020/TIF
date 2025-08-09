import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from set2set import Set2Set
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import torch_geometric.nn as pyg_nn

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None
    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y
class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1
        self.bias = True
        if args is not None:
            self.bias = args.bias
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim
        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                label_dim, num_aggs=self.num_aggs)
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)
    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias)
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last
    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model
    def construct_mask(self, max_nodes, batch_num_nodes):
        
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()
    def apply_bn(self, x):
        
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)
    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
        
        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor
    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x,adj)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred
    def loss(self, pred, label, type='softmax'):
        if type == 'softmax':
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1,1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)
class GcnSet2SetEncoder(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnSet2SetEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims, concat, bn, dropout, args=args)
        self.s2s = Set2Set(self.pred_input_dim, self.pred_input_dim * 2)
    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None
        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out = self.s2s(embedding_tensor)
        ypred = self.pred_model(out)
        return ypred
def hook_fn(module, input, output):
    input_shapes = [inp.shape for inp in input]
    output_shapes = [output.shape] if isinstance(output, torch.Tensor) else [out.shape for out in output]
    print(f"Layer: {module.__class__.__name__} | Input shapes: {input_shapes} | Output shapes: {output_shapes}")
class ModifiedRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_output=4):
        super(ModifiedRouter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, num_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x
class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
            assign_input_dim=-1, args=None):
        
        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                    add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim
        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        self.perturb_conv_first_modules = nn.ModuleList()
        self.perturb_conv_block_modules = nn.ModuleList()
        self.perturb_conv_last_modules = nn.ModuleList()
        self.perturb_conv_first_modules_2 = nn.ModuleList()
        self.perturb_conv_block_modules_2 = nn.ModuleList()
        self.perturb_conv_last_modules_2 = nn.ModuleList()
        self.perturb_conv_first_modules_round2_1 = nn.ModuleList()
        self.perturb_conv_block_modules_round2_1 = nn.ModuleList()
        self.perturb_conv_last_modules_round2_1 = nn.ModuleList()
        self.perturb_conv_first_modules_2_round2_1 = nn.ModuleList()
        self.perturb_conv_block_modules_2_round2_1 = nn.ModuleList()
        self.perturb_conv_last_modules_2_round2_1 = nn.ModuleList()
        self.perturb_conv_first_modules_round2_2 = nn.ModuleList()
        self.perturb_conv_block_modules_round2_2 = nn.ModuleList()
        self.perturb_conv_last_modules_round2_2 = nn.ModuleList()
        self.perturb_conv_first_modules_2_round2_2 = nn.ModuleList()
        self.perturb_conv_block_modules_2_round2_2 = nn.ModuleList()
        self.perturb_conv_last_modules_2_round2_2 = nn.ModuleList()
        self.perturb_conv_first_modules_round2_3 = nn.ModuleList()
        self.perturb_conv_block_modules_round2_3 = nn.ModuleList()
        self.perturb_conv_last_modules_round2_3 = nn.ModuleList()
        self.perturb_conv_first_modules_2_round2_3 = nn.ModuleList()
        self.perturb_conv_block_modules_2_round2_3 = nn.ModuleList()
        self.perturb_conv_last_modules_2_round2_3 = nn.ModuleList()
        self.perturb_conv_first_modules_round2_4 = nn.ModuleList()
        self.perturb_conv_block_modules_round2_4 = nn.ModuleList()
        self.perturb_conv_last_modules_round2_4 = nn.ModuleList()
        self.perturb_conv_first_modules_2_round2_4 = nn.ModuleList()
        self.perturb_conv_block_modules_2_round2_4 = nn.ModuleList()
        self.perturb_conv_last_modules_2_round2_4 = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)
            perturb_conv_first, perturb_conv_block, perturb_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
            perturb_conv_first_2, perturb_conv_block_2, perturb_conv_last_2 = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
            perturb_conv_first_1, perturb_conv_block_1, perturb_conv_last_1 = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self, normalize=True)
            perturb_conv_first_2_1, perturb_conv_block_2_1, perturb_conv_last_2_1 = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self, normalize=True)
            perturb_conv_first_2, perturb_conv_block_2, perturb_conv_last_2 = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self, normalize=True)
            perturb_conv_first_2_2, perturb_conv_block_2_2, perturb_conv_last_2_2 = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self, normalize=True)
            perturb_conv_first_3, perturb_conv_block_3, perturb_conv_last_3 = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self, normalize=True)
            perturb_conv_first_2_3, perturb_conv_block_2_3, perturb_conv_last_2_3 = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self, normalize=True)
            perturb_conv_first_4, perturb_conv_block_4, perturb_conv_last_4 = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self, normalize=True)
            perturb_conv_first_2_4, perturb_conv_block_2_4, perturb_conv_last_2_4 = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self, normalize=True)
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)
            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)
            self.perturb_conv_first_modules.append(perturb_conv_first)
            self.perturb_conv_block_modules.append(perturb_conv_block)
            self.perturb_conv_last_modules.append(perturb_conv_last)
            self.perturb_conv_first_modules_2.append(perturb_conv_first_2)
            self.perturb_conv_block_modules_2.append(perturb_conv_block_2)
            self.perturb_conv_last_modules_2.append(perturb_conv_last_2)
            self.perturb_conv_first_modules_round2_1.append(perturb_conv_first_1)
            self.perturb_conv_block_modules_round2_1.append(perturb_conv_block_1)
            self.perturb_conv_last_modules_round2_1.append(perturb_conv_last_1)
            self.perturb_conv_first_modules_2_round2_1.append(perturb_conv_first_2_1)
            self.perturb_conv_block_modules_2_round2_1.append(perturb_conv_block_2_1)
            self.perturb_conv_last_modules_2_round2_1.append(perturb_conv_last_2_1)
            self.perturb_conv_first_modules_round2_2.append(perturb_conv_first_2)
            self.perturb_conv_block_modules_round2_2.append(perturb_conv_block_2)
            self.perturb_conv_last_modules_round2_2.append(perturb_conv_last_2)
            self.perturb_conv_first_modules_2_round2_2.append(perturb_conv_first_2_2)
            self.perturb_conv_block_modules_2_round2_2.append(perturb_conv_block_2_2)
            self.perturb_conv_last_modules_2_round2_2.append(perturb_conv_last_2_2)
            self.perturb_conv_first_modules_round2_3.append(perturb_conv_first_3)
            self.perturb_conv_block_modules_round2_3.append(perturb_conv_block_3)
            self.perturb_conv_last_modules_round2_3.append(perturb_conv_last_3)
            self.perturb_conv_first_modules_2_round2_3.append(perturb_conv_first_2_3)
            self.perturb_conv_block_modules_2_round2_3.append(perturb_conv_block_2_3)
            self.perturb_conv_last_modules_2_round2_3.append(perturb_conv_last_2_3)
            self.perturb_conv_first_modules_round2_4.append(perturb_conv_first_4)
            self.perturb_conv_block_modules_round2_4.append(perturb_conv_block_4)
            self.perturb_conv_last_modules_round2_4.append(perturb_conv_last_4)
            self.perturb_conv_first_modules_2_round2_4.append(perturb_conv_first_2_4)
            self.perturb_conv_block_modules_2_round2_4.append(perturb_conv_block_2_4)
            self.perturb_conv_last_modules_2_round2_4.append(perturb_conv_last_2_4)
        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling+1), pred_hidden_dims,
                label_dim, num_aggs=self.num_aggs)
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)
        self.router = ModifiedRouter(
            input_dim=int(1000 * assign_ratio * embedding_dim * 4 * 3),
            hidden_dim=hidden_dim,
            num_output=4
        )
        self.router2 = ModifiedRouter(
            input_dim=int(1000 * assign_ratio * assign_ratio * embedding_dim * 4 * 3),
            hidden_dim=hidden_dim,
            num_output=4
        )
        self.router3 = ModifiedRouter(
            input_dim=int(1000 * assign_ratio * assign_ratio * embedding_dim * 4 * 3),
            hidden_dim=hidden_dim,
            num_output=4
        )
        self.router4 = ModifiedRouter(
            input_dim=int(1000 * assign_ratio * assign_ratio * embedding_dim * 4 * 3),
            hidden_dim=hidden_dim,
            num_output=4
        )
        self.router5 = ModifiedRouter(
            input_dim=int(1000 * assign_ratio * assign_ratio * embedding_dim * 4 * 3),
            hidden_dim=hidden_dim,
            num_output=4
        )
        self.initial_threshold = 0.0
        self.max_threshold = 0.1
        self.hardening_rate = 0.01
    def adjust_threshold(self, current_epoch, total_epochs):
        if current_epoch > 0.7 * total_epochs:
            return min(self.max_threshold,
                       self.initial_threshold + ((current_epoch - 0.7 * total_epochs) / (0.3 * total_epochs)) * (
                                   self.max_threshold - self.initial_threshold))
        else:
            return 0.0
    def harden_clusters(self, S, threshold):
        
        S_hardened = torch.zeros_like(S)
        max_probs, max_indices = torch.max(S, dim=-1)
        S_hardened[max_probs >= threshold] = torch.eye(S.size(-1)).to(S.device)[max_indices[max_probs >= threshold]]
        S_hardened[max_probs < threshold] = S[max_probs < threshold]
        return S_hardened
    def perturb_loss(self, X, X_list, lambda_list, mu):
        loss_perturb = 0.0
        for i in range(len(X_list)):
            loss_perturb += lambda_list[i] * torch.sum((X - X_list[i]) ** 2)
        for i in range(len(X_list)):
            for j in range(i + 1, len(X_list)):
                loss_perturb += mu * torch.sum((X_list[i] - X_list[j]) ** 2)
        return loss_perturb
    def entropy_loss(self, route_probs):
        entropy = -torch.sum(route_probs * torch.log(route_probs + 1e-9), dim=1)
        return torch.mean(entropy)
    def forward(self, x, adj, batch_num_nodes, current_epoch=0, total_epochs=100, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None
        out_all = []
        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)
        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None
            self.assign_tensor = self.gcn_forward(x_a, adj,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            threshold = self.adjust_threshold(current_epoch, total_epochs)
            self.assign_tensor = self.harden_clusters(self.assign_tensor, threshold)
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask
            if i == 0:
                perturb_tensor = self.gcn_forward(x_a, adj,
                                                  self.perturb_conv_first_modules[i],
                                                  self.perturb_conv_block_modules[i],
                                                  self.perturb_conv_last_modules[i], embedding_mask)
                perturb_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](perturb_tensor))
                if embedding_mask is not None:
                    perturb_tensor = perturb_tensor * embedding_mask
                perturb_tensor_2 = self.gcn_forward(x_a, adj,
                                                    self.perturb_conv_first_modules_2[i],
                                                    self.perturb_conv_block_modules_2[i],
                                                    self.perturb_conv_last_modules_2[i], embedding_mask)
                perturb_tensor_2 = nn.Softmax(dim=-1)(self.assign_pred_modules[i](perturb_tensor_2))
                if embedding_mask is not None:
                    perturb_tensor_2 = perturb_tensor_2 * embedding_mask
            else:
                route_index = self.first_round_chosen_route[0].item()
                if route_index == 0:
                    perturb_tensor = self.gcn_forward(x_a, adj,
                                                      self.perturb_conv_first_modules_round2_1[i],
                                                      self.perturb_conv_block_modules_round2_1[i],
                                                      self.perturb_conv_last_modules_round2_1[i],
                                                      embedding_mask)
                    perturb_tensor_2 = self.gcn_forward(x_a, adj,
                                                        self.perturb_conv_first_modules_2_round2_1[i],
                                                        self.perturb_conv_block_modules_2_round2_1[i],
                                                        self.perturb_conv_last_modules_2_round2_1[i],
                                                        embedding_mask)
                elif route_index == 1:
                    perturb_tensor = self.gcn_forward(x_a, adj,
                                                      self.perturb_conv_first_modules_round2_2[i],
                                                      self.perturb_conv_block_modules_round2_2[i],
                                                      self.perturb_conv_last_modules_round2_2[i],
                                                      embedding_mask)
                    perturb_tensor_2 = self.gcn_forward(x_a, adj,
                                                        self.perturb_conv_first_modules_2_round2_2[i],
                                                        self.perturb_conv_block_modules_2_round2_2[i],
                                                        self.perturb_conv_last_modules_2_round2_2[i],
                                                        embedding_mask)
                elif route_index == 2:
                    perturb_tensor = self.gcn_forward(x_a, adj,
                                                      self.perturb_conv_first_modules_round2_3[i],
                                                      self.perturb_conv_block_modules_round2_3[i],
                                                      self.perturb_conv_last_modules_round2_3[i],
                                                      embedding_mask)
                    perturb_tensor_2 = self.gcn_forward(x_a, adj,
                                                        self.perturb_conv_first_modules_2_round2_3[i],
                                                        self.perturb_conv_block_modules_2_round2_3[i],
                                                        self.perturb_conv_last_modules_2_round2_3[i],
                                                        embedding_mask)
                else:
                    perturb_tensor = self.gcn_forward(x_a, adj,
                                                      self.perturb_conv_first_modules_round2_4[i],
                                                      self.perturb_conv_block_modules_round2_4[i],
                                                      self.perturb_conv_last_modules_round2_4[i],
                                                      embedding_mask)
                    perturb_tensor_2 = self.gcn_forward(x_a, adj,
                                                        self.perturb_conv_first_modules_2_round2_4[i],
                                                        self.perturb_conv_block_modules_2_round2_4[i],
                                                        self.perturb_conv_last_modules_2_round2_4[i],
                                                        embedding_mask)
                perturb_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](perturb_tensor))
                perturb_tensor_2 = nn.Softmax(dim=-1)(self.assign_pred_modules[i](perturb_tensor_2))
                if embedding_mask is not None:
                    perturb_tensor = perturb_tensor * embedding_mask
                    perturb_tensor_2 = perturb_tensor_2 * embedding_mask
            S_1 = self.assign_tensor
            S_2 = self.assign_tensor + perturb_tensor
            S_3 = self.assign_tensor + perturb_tensor_2
            S_4 = self.assign_tensor + perturb_tensor + perturb_tensor_2
            x_pooled_1, adj_pooled_1 = self.diffpool(embedding_tensor, adj, S_1)
            x_pooled_2, adj_pooled_2 = self.diffpool(embedding_tensor, adj, S_2)
            x_pooled_3, adj_pooled_3 = self.diffpool(embedding_tensor, adj, S_3)
            x_pooled_4, adj_pooled_4 = self.diffpool(embedding_tensor, adj, S_4)
            pooled_embeddings = torch.cat(
                [x_pooled_1.flatten(1), x_pooled_2.flatten(1), x_pooled_3.flatten(1), x_pooled_4.flatten(1)], dim=1)
            if i == 0:
                route_logits = self.router(pooled_embeddings)
                route_probs = F.softmax(route_logits, dim=1)
                chosen_route = torch.multinomial(route_probs, num_samples=1).squeeze(1)
                self.first_round_chosen_route = chosen_route
            else:
                if self.first_round_chosen_route[0] == 0:
                    current_router = self.router2
                elif self.first_round_chosen_route[0] == 1:
                    current_router = self.router3
                elif self.first_round_chosen_route[0] == 2:
                    current_router = self.router4
                else:
                    current_router = self.router5
                route_logits = current_router(pooled_embeddings)
            route_probs = F.softmax(route_logits, dim=1)
            chosen_route = torch.multinomial(route_probs, num_samples=1).squeeze(1)
            x_pooled = torch.stack([x_pooled_1, x_pooled_2, x_pooled_3, x_pooled_4], dim=1)
            adj_pooled = torch.stack([adj_pooled_1, adj_pooled_2, adj_pooled_3, adj_pooled_4], dim=1)
            batch_indices = torch.arange(x_pooled.size(0))
            x = x_pooled[batch_indices, chosen_route, :, :]
            x_a = x
            adj = adj_pooled[batch_indices, chosen_route, :, :]
            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])
            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred, [x_pooled_1, x_pooled_2, x_pooled_3, x_pooled_4], route_probs
    def diffpool(self, x, adj, S):
        out_x = torch.matmul(S.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(S.transpose(1, 2), adj), S)
        return out_x, out_adj
    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1, perturb_reg=None, route_probs=None, alpha=0.1):
        
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if perturb_reg is not None:
            lambda_list = [1.0, 1.0, 1.0, 1.0]
            mu = 1.0
            perturb_loss = self.perturb_loss(perturb_reg[0], perturb_reg, lambda_list, mu)
            loss += perturb_loss
        if route_probs is not None:
            entropy_loss = self.entropy_loss(route_probs)
            loss += alpha * entropy_loss
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1-adj_mask).bool()] = 0.0
            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            return loss + self.link_loss
        return loss