# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import sys
import time
import math
import random
import subprocess
import logging
import argparse

from collections import defaultdict

import numpy as np
import scipy
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


from common import DATASET_NUM_DIC
#
from fea_extra import FeaExtra
from logistic_function import logistic_embedding



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--devices', type=str, default='cpu', help='Devices')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dataset', default='bitcoin_alpha', help='Dataset')
parser.add_argument('--dim', type=int, default=64, help='Embedding dimension')
parser.add_argument('--fea_dim', type=int, default=64, help='Feature embedding dimension')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout k')
parser.add_argument('--k', default=1, help='Folder k')
parser.add_argument('--agg', default='attantion', choices=['mean', 'attantion'], help='Aggregator choose')

args = parser.parse_args()

OUTPUT_DIR = f''
# if not os.path.exists('embeddings-att'):
#     os.mkdir('embeddings-att')

if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

NEG_LOSS_RATIO = 1
INTERVAL_PRINT = 2

NUM_NODE = DATASET_NUM_DIC[args.dataset]
WEIGHT_DECAY = args.weight_decay
NODE_FEAT_SIZE = args.fea_dim
EMBEDDING_SIZE1 = args.dim
DEVICES = torch.device(args.devices)
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
DROUPOUT = args.dropout
K = args.k



class Encoder(nn.Module):
    """
    Encode features to 2_特征处理
    """

    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggs):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggs = aggs

        self.embed_dim = embed_dim
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.to(DEVICES)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        self.nonlinear_layer = nn.Sequential(
                nn.Linear((len(adj_lists) + 1) * feature_dim, feature_dim),  # motifs+本身
                nn.Tanh(),
                nn.Linear(feature_dim, embed_dim)
        )

        self.nonlinear_layer.apply(init_weights)  # 初始化


    def forward(self, nodes):
        """
        Generates 2_特征处理 for nodes.
        """

        if not isinstance(nodes, list) and nodes.is_cuda:
            nodes = nodes.data.cpu().numpy().tolist()

        neigh_feats = [agg(nodes, adj, ind) for adj, agg, ind in zip(self.adj_lists, self.aggs, range(len(self.adj_lists)))]
        self_feats = self.features(torch.LongTensor(nodes).to(DEVICES))  # 节点本身motifs的特征
        combined = torch.cat([self_feats] + neigh_feats, 1)  # 邻居+节点本身
        combined = self.nonlinear_layer(combined)
        return combined

        k = self.k(self_feats)


class AttentionAggregator(nn.Module):
    def __init__(self, features, in_dim, out_dim, node_num,  dropout_rate=DROUPOUT, slope_ratio=0.1):
        super(AttentionAggregator, self).__init__()

        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.slope_ratio = slope_ratio
        self.a = nn.Parameter(torch.FloatTensor(out_dim * 2, 1))  # Whi||Whj
        nn.init.kaiming_normal_(self.a.data)

        self.out_linear_layer = nn.Linear(self.in_dim, self.out_dim)  # W*H
        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)


    def forward(self, nodes, adj, ind):
        """
        nodes --- list of nodes in a batch
        adj --- sp.csr_matrix
        """
        node_pku = np.array(nodes)
        edges = np.array(adj[nodes, :].nonzero()).T
        edges[:, 0] = node_pku[edges[:, 0]]  # 将选择后的node对应到之前的node编号上

        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))

        batch_node_num = len(unique_nodes_list)
        # this dict can map new i to originial node id
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)  # 选择node的编号

        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)  # 第一层涉及到的邻居
        f = self.features(n2)
        new_embeddings = self.out_linear_layer(f)  # self.features(n2)

        original_node_edge = np.array([self.unique_nodes_dict[nodes], self.unique_nodes_dict[nodes]]).T
        edges = np.vstack((edges, original_node_edge))  # 加上自连接的边

        edges = torch.LongTensor(edges).to(DEVICES)

        edge_h_2 = torch.cat((new_embeddings[edges[:, 0], :], new_embeddings[edges[:, 1], :]), dim=1)  # Whi||Whj

        edges_h = torch.exp(F.leaky_relu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), self.slope_ratio))  # attention
        indices = edges
        
        matrix = torch.sparse_coo_tensor(indices.t(), edges_h[:, 0], \
                                         torch.Size([batch_node_num, batch_node_num]), device=DEVICES)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))  # attention求和

        results = torch.sparse.mm(matrix, new_embeddings)  # attention和embedding内积

        output_emb = results.div(row_sum)  # 归一化

        return output_emb[self.unique_nodes_dict[nodes]]

class MeanAggregator(nn.Module):
    def __init__(self, features, in_dim, out_dim, node_num):
        super(MeanAggregator, self).__init__()

        self.features = features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_linear_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
            nn.Linear(self.out_dim, self.out_dim)
        )

        self.unique_nodes_dict = np.zeros(node_num, dtype=np.int32)

    def forward(self, nodes, adj, ind):
        """

        :param nodes:
        :param adj:
        :return:
        """
        mask = [1, 1, 0, 0]
        node_tmp = np.array(nodes)
        edges = np.array(adj[nodes, :].nonzero()).T
        edges[:, 0] = node_tmp[edges[:, 0]]

        unique_nodes_list = np.unique(np.hstack((np.unique(edges), np.array(nodes))))
        batch_node_num = len(unique_nodes_list)
        self.unique_nodes_dict[unique_nodes_list] = np.arange(batch_node_num)

        ## transform 2 new axis
        edges[:, 0] = self.unique_nodes_dict[edges[:, 0]]
        edges[:, 1] = self.unique_nodes_dict[edges[:, 1]]

        n2 = torch.LongTensor(unique_nodes_list).to(DEVICES)
        new_embeddings = self.out_linear_layer(self.features(n2))
        edges = torch.LongTensor(edges).to(DEVICES)

        values = torch.where(edges[:, 0] == edges[:, 1], torch.FloatTensor([mask[ind]]).to(DEVICES), torch.FloatTensor([1]).to(DEVICES))
        # values = torch.ones(edges.shape[0]).to(DEVICES)
        matrix = torch.sparse_coo_tensor(edges.t(), values, torch.Size([batch_node_num, batch_node_num]), device=DEVICES)
        row_sum = torch.spmm(matrix, torch.ones(size=(batch_node_num, 1)).to(DEVICES))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(DEVICES), row_sum)

        results = torch.spmm(matrix, new_embeddings)
        output_emb = results.div(row_sum)

        return output_emb[self.unique_nodes_dict[nodes]]



class SDGNN(nn.Module):

    def __init__(self, enc):
        super(SDGNN, self).__init__()
        self.enc = enc
        self.score_function1 = nn.Sequential(  # 定义status ranking
            nn.Linear(EMBEDDING_SIZE1, 1),
            nn.Sigmoid()
        )
        self.score_function2 = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE1, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(EMBEDDING_SIZE1 * 2, 1)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        return embeds

    def criterion(self, nodes, pos_neighbors, neg_neighbors, adj_lists1_1, adj_lists2_1, weight_dict):
        pos_neighbors_list = [set.union(pos_neighbors[i]) for i in nodes]  # pos的邻居
        neg_neighbors_list = [set.union(neg_neighbors[i]) for i in nodes]  # neg的邻居
        unique_nodes_list = list(set.union(*pos_neighbors_list).union(*neg_neighbors_list).union(nodes))  # 所有涉及到的节点
        unique_nodes_dict = {n: i for i, n in enumerate(unique_nodes_list)}  # 节点编号
        nodes_embs = self.enc(unique_nodes_list)

        loss_total = 0
        for index, node in enumerate(nodes):
            z1 = nodes_embs[unique_nodes_dict[node], :]  # 节点的embedding
            pos_neigs = list([unique_nodes_dict[i] for i in pos_neighbors[node]])  # 节点的positive邻居
            neg_neigs = list([unique_nodes_dict[i] for i in neg_neighbors[node]])  # 节点的negative邻居
            pos_num = len(pos_neigs)
            neg_num = len(neg_neigs)

            sta_pos_neighs = list([unique_nodes_dict[i] for i in adj_lists1_1[node]])  # pos:u->v 对应的index
            sta_neg_neighs = list([unique_nodes_dict[i] for i in adj_lists2_1[node]])  # neg:u->v
            # 平衡三角形weight
            pos_neigs_weight = torch.FloatTensor([weight_dict[node][i] for i in adj_lists1_1[node]]).to(DEVICES)  # u->v之间满足平衡三角形weight
            neg_neigs_weight = torch.FloatTensor([weight_dict[node][i] for i in adj_lists2_1[node]]).to(DEVICES)

            if pos_num > 0:
                pos_neig_embs = nodes_embs[pos_neigs, :]  # pos节点embedding
                loss_pku = F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [pos_neig_embs, z1]),
                                                              torch.ones(pos_num).to(DEVICES))

                if len(sta_pos_neighs) > 0:
                    sta_pos_neig_embs = nodes_embs[sta_pos_neighs, :]  # pos的u->v连接embedding

                    z11 = z1.repeat(len(sta_pos_neighs), 1)  # 重复n次
                    rs = self.fc(torch.cat([z11, sta_pos_neig_embs], 1)).squeeze(-1)  # Z1||pos_emb -> 1
                    loss_pku += F.binary_cross_entropy_with_logits(rs, torch.ones(len(sta_pos_neighs)).to(DEVICES), \
                                                                   weight=pos_neigs_weight
                                                                   )  # 边类别预测
                    s1 = self.score_function1(z1).repeat(len(sta_pos_neighs), 1)  # z1重复n次
                    s2 = self.score_function2(sta_pos_neig_embs)

                    q = torch.where((s1 - s2) > -0.5, torch.Tensor([-0.5]).repeat(s1.shape).to(DEVICES), s1 - s2)  # !!!! 0.5
                    tmp = (q - (s1 - s2))
                    # loss_pku += 5 * torch.einsum("ij,ij->", [tmp, tmp])  # ^2

                loss_total += loss_pku

            if neg_num > 0:
                neg_neig_embs = nodes_embs[neg_neigs, :]
                loss_pku = F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [neg_neig_embs, z1]),
                                                              torch.zeros(neg_num).to(DEVICES))
                if len(sta_neg_neighs) > 0:
                    sta_neg_neig_embs = nodes_embs[sta_neg_neighs, :]

                    z12 = z1.repeat(len(sta_neg_neighs), 1)
                    rs = self.fc(torch.cat([z12, sta_neg_neig_embs], 1)).squeeze(-1)

                    loss_pku += F.binary_cross_entropy_with_logits(rs, torch.zeros(len(sta_neg_neighs)).to(DEVICES), \
                                                                   weight=neg_neigs_weight)

                    s1 = self.score_function1(z1).repeat(len(sta_neg_neighs), 1)
                    s2 = self.score_function2(sta_neg_neig_embs)

                    q = torch.where(s1 - s2 > 0.5, s1 - s2, torch.Tensor([0.5]).repeat(s1.shape).to(DEVICES))

                    tmp = (q - (s1 - s2))
                    # loss_pku += 5 * torch.einsum("ij,ij->", [tmp, tmp])

                loss_total += loss_pku

        return loss_total


def load_data2(filename=''):
    adj_lists1 = defaultdict(set)
    adj_lists1_1 = defaultdict(set)
    adj_lists1_2 = defaultdict(set)
    adj_lists2 = defaultdict(set)
    adj_lists2_1 = defaultdict(set)
    adj_lists2_2 = defaultdict(set)
    adj_lists3 = defaultdict(set)

    with open(filename) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            person1 = int(info[0])
            person2 = int(info[1])
            v = int(info[2])
            adj_lists3[person2].add(person1)
            adj_lists3[person1].add(person2)

            if v == 1:
                adj_lists1[person1].add(person2)
                adj_lists1[person2].add(person1)

                adj_lists1_1[person1].add(person2)
                adj_lists1_2[person2].add(person1)
            else:
                adj_lists2[person1].add(person2)
                adj_lists2[person2].add(person1)

                adj_lists2_1[person1].add(person2)
                adj_lists2_2[person2].add(person1)

    return adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3


def read_emb(num_nodes, fpath):
    dim = 0
    embeddings = 0
    with open(fpath) as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                dim = int(line.split()[1])
                embeddings = np.random.rand(num_nodes, dim)
            else:
                line_l = line.split()
                node = line_l[0]
                emb = [float(j) for j in line_l[1:]]
                assert len(emb) == dim
                embeddings[int(node)] = np.array(emb)
    return embeddings


def run(dataset, k):
    num_nodes = DATASET_NUM_DIC[dataset] + 3
    print(num_nodes, 'data num')
    # adj_lists1, adj_lists2, adj_lists3 = load_data(k, dataset)
    filename = ''.format(dataset, k)
    adj_lists1, adj_lists1_1, adj_lists1_2, adj_lists2, adj_lists2_1, adj_lists2_2, adj_lists3 = load_data2(filename)
    print(k, dataset, 'data load!')

    def csv_to_tensor(csv_file) :
        with open(csv_file, 'r') as file :
            csvreader = csv.reader(file)
            data = [[float(val) for val in line] for line in csvreader]
        tensor_data = torch.tensor(data)
        return tensor_data

    csv_file = ''  # 输入的 CSV 文件路径
    pre_trained_features = csv_to_tensor(csv_file)
    # print(pre_trained_features)

# # #*************************加入新的原始特征
#     pre_trained_features = torch.randn(num_nodes, NODE_FEAT_SIZE)
#     print(pre_trained_features)
# #
#     # 直接使用预训练的嵌入向量初始化嵌入层
    features = nn.Embedding.from_pretrained(pre_trained_features)
# # #**************************************

    # features = nn.Embedding(num_nodes, NODE_FEAT_SIZE)
    features.weight.requires_grad = True

    features = features.to(DEVICES)
    # 有向的四个边; 4个motifs
    adj_lists = [adj_lists1_1, adj_lists1_2,  adj_lists2_1, adj_lists2_2]

  
    weight_dict = defaultdict(dict)
    fea_model = FeaExtra(dataset=dataset, k=k)
    # # u -> v
    for i in adj_lists1_1:
        for j in adj_lists1_1[i]:
            v_list1 = fea_model.feature_part2(i, j)
            mask = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
            counts1 = np.dot(v_list1, mask)  # 构造balance三角形，在后面计算loss时候使用;
            weight_dict[i][j] = counts1

    for i in adj_lists2_1:
        for j in adj_lists2_1[i]:
            v_list1 = fea_model.feature_part2(i, j)
            mask = [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]

            counts1 = np.dot(v_list1, mask)
            weight_dict[i][j] = counts1

    adj_lists = adj_lists


    print(len(adj_lists), 'motifs')
    print(num_nodes)
    def func(adj_list):
        edges = []
        for a in adj_list:
            for b in adj_list[a]:
                edges.append((a, b))
        edges = np.array(edges)
        print(len(edges))
        adj = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
        return adj


    aggregator1 = MeanAggregator

    aggregator2 = AttentionAggregator

    adj_lists = list(map(func, adj_lists))
    aggs = [aggregator2(features, NODE_FEAT_SIZE, NODE_FEAT_SIZE, num_nodes) for adj in adj_lists]  # 构建motifs下节点的邻居聚合
    enc1 = Encoder(features, NODE_FEAT_SIZE, EMBEDDING_SIZE1, adj_lists, aggs)  # 4个motifs+本身节点 + cls => embedding
    enc1 = enc1.to(DEVICES)


    aggs2 = [aggregator1(lambda n: enc1(n), EMBEDDING_SIZE1, EMBEDDING_SIZE1, num_nodes) for _ in adj_lists]
    enc2 = Encoder(lambda n: enc1(n), EMBEDDING_SIZE1, EMBEDDING_SIZE1, adj_lists, aggs2)

    model = SDGNN(enc2)
    model = model.to(DEVICES)

    print(model.train())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters()) + list(enc1.parameters()) \
                                        + list(features.parameters())),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY
                                 )

    for epoch in range(EPOCHS + 2):
        total_loss = []
        if epoch % INTERVAL_PRINT == 1:
            model.eval()
            all_embedding = np.zeros((NUM_NODE, EMBEDDING_SIZE1))
            for i in range(0, NUM_NODE, BATCH_SIZE):
                begin_index = i
                end_index = i + BATCH_SIZE if i + BATCH_SIZE < NUM_NODE else NUM_NODE
                values = np.arange(begin_index, end_index)
                embed = model.forward(values.tolist())
                embed = embed.data.cpu().numpy()
                all_embedding[begin_index: end_index] = embed

            fpath = os.path.join(OUTPUT_DIR, 'embedding-{}-{}-{}.npy'.format(dataset, k, str(epoch)))
            np.save(fpath, all_embedding)
            # pos_ratio, accuracy, f1_score0, f1_score1, f1_score2, auc_score = logistic_embedding(k=k, dataset=dataset,
            #                                                                                      epoch=epoch,
            #                                                                                     dirname=OUTPUT_DIR)
            model.train()

        time1 = time.time()
        nodes_pku = np.random.permutation(NUM_NODE).tolist()
        for batch in range(NUM_NODE // BATCH_SIZE):
            optimizer.zero_grad()
            b_index = batch * BATCH_SIZE
            e_index = (batch + 1) * BATCH_SIZE
            nodes = nodes_pku[b_index:e_index]

            loss = model.criterion(
                nodes, adj_lists1, adj_lists2, adj_lists1_1, adj_lists2_1, weight_dict  # i,j节点的平衡三角形数量
            )
            total_loss.append(loss.data.cpu().numpy())

            loss.backward()
            optimizer.step()
        print(f'epoch: {epoch}, loss: {np.mean(total_loss)}, time: {time.time()-time1}')

def main():
    print('NUM_NODE', NUM_NODE)
    print('WEIGHT_DECAY', WEIGHT_DECAY)
    print('NODE_FEAT_SIZE', NODE_FEAT_SIZE)
    print('EMBEDDING_SIZE1', EMBEDDING_SIZE1)
    print('LEARNING_RATE', LEARNING_RATE)
    print('BATCH_SIZE', BATCH_SIZE)
    print('EPOCHS', EPOCHS)
    print('DROUPOUT', DROUPOUT)
    dataset = args.dataset
    run(dataset=dataset, k=K)


if __name__ == "__main__":
    main()

