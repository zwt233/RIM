import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import random
import copy
import sys
import os
import time
import argparse
import json
import numpy as np
import numpy.linalg as la
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from scipy.sparse import csgraph
from torch.backends import cudnn
from torch.optim import lr_scheduler
from utils import *
from graphConvolution import *

#hyperparameters
num_node = 2708
num_coreset = 140
num_class = 7
oracle_acc = 0.7
th = 0.05
batch_size = 5

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
#num_coreset = int((num_node-1500)*0.01)
hidden_size = 128
num_val = 500
num_test = 1000

def get_reliable_score(similarity):
    return (oracle_acc*similarity)/(oracle_acc*similarity+(1-oracle_acc)*(1-similarity)/(num_class-1))
    
def get_activated_node_dense(node,reliable_score,activated_node): 
    activated_vector=((adj_matrix2[node]*reliable_score)>th)+0
    activated_vector=activated_vector*activated_node
    count=num_ones.dot(activated_vector)
    return count,activated_vector

def get_max_reliable_info_node_dense(idx_used,high_score_nodes,activated_node,train_class,labels): 
    max_ral_node = 0
    max_activated_node = 0
    max_activated_num = 0 
    for node in high_score_nodes:
        reliable_score = oracle_acc
        activated_num,activated_node_tmp =get_activated_node_dense(node,reliable_score,activated_node)
        if activated_num > max_activated_num:
            max_activated_num = activated_num
            max_ral_node = node
            max_activated_node = activated_node_tmp        
    return max_ral_node,max_activated_node,max_activated_num

def update_reliability(idx_used,train_class,labels,num_node):
    activated_node = np.zeros(num_node)
    for node in idx_used:
        reliable_score = 0
        node_label = labels[node].item()
        if node_label in train_class:
            total_score = 0.0
            for tmp_node in train_class[node_label]:
                total_score+=reliability_list[tmp_node]
            for tmp_node in train_class[node_label]:
                reliable_score+=reliability_list[tmp_node]*get_reliable_score(similarity_feature[node][tmp_node])
            reliable_score = reliable_score/total_score
        else:
            reliable_score = oracle_acc
        reliability_list[node]=reliable_score
        activated_node+=((adj_matrix2[node]*reliable_score)>th)+0
    return np.ones(num_node)-((activated_node>0)+0)

def my_cross_entropy(x_pred,x_traget):
    logged_x_pred = torch.log(x_pred)
    cost_value = -torch.sum(x_traget*logged_x_pred)/x_pred.size()[0]
    return cost_value

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -1.0).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()

def random_pick(some_list, probabilities): 
    x = random.uniform(0,1) 
    cumulative_probability = 0.0 
    for item, item_probability in zip(some_list, probabilities): 
        cumulative_probability += item_probability 
        if x < cumulative_probability:
            break 
    return item 


def compute_cos_sim(vec_a,vec_b):
    return (vec_a.dot(vec_b.T))/(la.norm(vec_a)*la.norm(vec_b))
#read dataset

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid,bias=True)
        self.gc2 = GraphConvolution(nhid, nclass,bias=True)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
   
def train(epoch, model,record):

    model.train()
    optimizer.zero_grad()
    output = model(features_GCN, adj)
    output_ = F.softmax(output,dim=1)
    one_hot_labels = F.one_hot(labels, num_classes=num_class)
    weight_one_hot_labels = torch.mul(one_hot_labels,reliability_list)
    loss_train = my_cross_entropy(output_[idx_train],weight_one_hot_labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features_GCN, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    record[acc_val.item()] = acc_test.item()
#read data
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="cora")

reliability_list = np.ones(num_node)
num_zeros = np.zeros(num_node)
num_ones = np.ones(num_node)
labels = list(labels.cpu())
idx_val = list(idx_val.cpu())
idx_test = list(idx_test.cpu())
idx_avaliable = list()
for i in range(num_node):
    if i not in idx_val and i not in idx_test:
        idx_avaliable.append(i)

# add noise
label_list=[]
prob_list = np.full((num_class,num_class),(1-oracle_acc)/(num_class-1)).tolist()
for i in range(num_class):
    label_list.append(i)
    prob_list[i][i]=oracle_acc
for idx in idx_avaliable:
    labels[idx]=torch.tensor(random_pick(label_list,prob_list[labels[idx].item()]))

#compute normalized distance
adj = aug_normalized_adjacency(adj)
features = features.cuda()
adj_matrix = torch.FloatTensor(adj.todense()).cuda()
adj_matrix2 = torch.mm(adj_matrix,adj_matrix).cuda()
aax_feature = torch.mm(adj_matrix2,features)
aax_feature = np.array(aax_feature.cpu())
adj_matrix2 = np.array(adj_matrix2.cpu())
features = features.cpu()
adj = sparse_mx_to_torch_sparse_tensor(adj).float().cuda()
features_GCN = copy.deepcopy(features)
features_GCN = torch.FloatTensor(features_GCN).cuda()

similarity_feature = np.ones((num_node,num_node))
for i in range(num_node-1):
    for j in range(i+1,num_node):
        similarity_feature[i][j] = compute_cos_sim(aax_feature[i],aax_feature[j])
        similarity_feature[j][i] = similarity_feature[i][j]
dis_range = np.max(similarity_feature) - np.min(similarity_feature)
similarity_feature = (similarity_feature - np.min(similarity_feature))/dis_range

#chooose node
print("node selection begin")
activated_node = np.ones(num_node)
idx_train = []
train_class = dict()
idx_avaliable_temp = copy.deepcopy(idx_avaliable)
count = 0
while True:
    max_ral_node,max_activated_node,max_activated_num = get_max_reliable_info_node_dense(idx_train,idx_avaliable_temp,activated_node,train_class,labels) 
    idx_train.append(max_ral_node) 
    idx_avaliable.remove(max_ral_node)
    idx_avaliable_temp.remove(max_ral_node)
    node_label = labels[max_ral_node].item()
    if node_label in train_class:
        train_class[node_label].append(max_ral_node)
    else:
        train_class[node_label]=list()
        train_class[node_label].append(max_ral_node)
    count += 1
    if count%batch_size == 0:
        activated_node = update_reliability(idx_train,train_class,labels,num_node)
    activated_node = activated_node - max_activated_node
    if count >= num_coreset or max_activated_num <= 0:
        break
print("node selection end")

labels = torch.LongTensor(labels).cuda()
idx_train = torch.LongTensor(idx_train).cuda()
idx_val = torch.LongTensor(idx_val).cuda()
idx_test = torch.LongTensor(idx_test).cuda()
reliability_list = torch.FloatTensor(reliability_list).unsqueeze(1).cuda()
#train
print('xxxxxxxxxx Evaluation begin xxxxxxxxxx')
t_total = time.time()
record = {}
model = GCN(nfeat=features_GCN.shape[1],
        nhid=hidden_size,
        nclass=labels.max().item() + 1,
        dropout=0.85)
model.cuda()
optimizer = optim.Adam(model.parameters(),
                        lr=0.05, weight_decay=5e-4)
for epoch in range(400):
    train(epoch,model,record)

bit_list = sorted(record.keys())
bit_list.reverse()
for key in bit_list[:10]:
    value = record[key]
    print(key,value)
print('xxxxxxxxxx Evaluation end xxxxxxxxxx')