"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import math
import time
import pandas as pd
import string

def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data



def rfunc(triple_list, ent_num, rel_num):
    head = dict()
    tail = dict()
    rel_count = dict()
    r_mat_ind = list()
    r_mat_val = list()
    head_r = np.zeros((ent_num, rel_num))
    tail_r = np.zeros((ent_num, rel_num))
    for triple in triple_list:
        head_r[triple[0]][triple[1]] = 1
        tail_r[triple[2]][triple[1]] = 1
        r_mat_ind.append([triple[0], triple[2]])
        r_mat_val.append(triple[1])
        if triple[1] not in rel_count:
            rel_count[triple[1]] = 1
            head[triple[1]] = set()
            tail[triple[1]] = set()
            head[triple[1]].add(triple[0])
            tail[triple[1]].add(triple[2])
        else:
            rel_count[triple[1]] += 1
            head[triple[1]].add(triple[0])
            tail[triple[1]].add(triple[2])
    r_mat = tf.SparseTensor(indices=r_mat_ind, values=r_mat_val, dense_shape=[ent_num, ent_num])

    return head, tail, head_r, tail_r, r_mat


def get_mat(triple_list, ent_num):
    degree = [1] * ent_num
    pos = dict()
    for triple in triple_list:
        if triple[0] != triple[1]:
            degree[triple[0]] += 1
            degree[triple[1]] += 1
        if triple[0] == triple[2]:
            continue
        if (triple[0], triple[2]) not in pos:
            pos[(triple[0], triple[2])] = 1
            pos[(triple[2], triple[0])] = 1

    for i in range(ent_num):
        pos[(i, i)] = 1
    return pos, degree


def get_sparse_tensor(triple_list, ent_num):
    pos, degree = get_mat(triple_list, ent_num)
    ind = []
    val = []
    for fir, sec in pos:
        ind.append((sec, fir))
        val.append(pos[(fir, sec)] / math.sqrt(degree[fir]) / math.sqrt(degree[sec]))
    pos = tf.SparseTensor(indices=ind, values=val, dense_shape=[ent_num, ent_num])

    return pos


def get_neg(ILL, output_layer, k):
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    KG_vec = np.array(output_layer)
    sim = scipy.spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
    for i in range(t):
        rank = sim[i, :].argsort()
        neg.append(rank[0:k])

    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def clear_attribute_triples(attribute_triples):
    print('\nbefore clear:', len(attribute_triples))
    # step 1
    attribute_triples_new = set()
    attr_num = {}
    for (e, a, _) in attribute_triples:
        ent_num = 1
        if a in attr_num:
            ent_num += attr_num[a]
        attr_num[a] = ent_num
    attr_set = set(attr_num.keys())
    attr_set_new = set()
    for a in attr_set:
        if attr_num[a] >= 10:
            attr_set_new.add(a)
    for (e, a, v) in attribute_triples:
        if a in attr_set_new:
            attribute_triples_new.add((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 1:', len(attribute_triples))

    # step 2
    attribute_triples_new = []
    literals_number, literals_string = [], []
    for (e, a, v) in attribute_triples:
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if v.endswith('"@en'):
            v = v[:v.index('"@en')]
        if is_number(v):
            literals_number.append(v)
        else:
            literals_string.append(v)
        v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('"', '')
        v = v.replace('_', ' ').replace('-', ' ').replace('/', ' ')
        if 'http' in v:
            continue
        attribute_triples_new.append((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 2:', len(attribute_triples))
    return attribute_triples, literals_number, literals_string



def get_adj(triple_list, ent_num):
    edges=[]
    for triple in triple_list:
      edges.append([triple[0], triple[2]])
      edges.append([triple[2], triple[0]])
    print(edges)
    adj = nx.adjacency_matrix(nx.from_edgelist(edges))
    print(adj.shape)
    return adj
    


def get_mat(triple_list, ent_num):
    degree = [1] * ent_num
    pos = {}
    for triple in triple_list:
        if triple[0] != triple[1]:
            degree[triple[0]] += 1
            degree[triple[1]] += 1
        if triple[0] == triple[2]:
            continue
        if (triple[0], triple[2]) not in pos:
            pos[(triple[0], triple[2])] = 1
            pos[(triple[2], triple[0])] = 1

    for i in range(ent_num):
        pos[(i, i)] = 1
    return pos, degree


def get_sparse_tensor(triple_list, ent_num):
    pos, degree = get_mat(triple_list, ent_num)
    ind1 = []
    ind2 = []
    val = []
    print(pos)
    for fir, sec in pos:
        ind1.append(fir)
        ind2.append(sec)
        val.append(pos[(fir, sec)] / math.sqrt(degree[fir]) / math.sqrt(degree[sec]))
    print(ent_num)
    print(len(triple_list))
    pos = torch.sparse_coo_tensor(torch.tensor([ind1,ind2]), torch.tensor(val),[ent_num, ent_num])

    return pos


def load_data_1(kgs):
    features=np.load("embeds.npy")
    features = torch.Tensor(features)
    #data={}
    triple_list = kgs.kg1.relation_triples_list + kgs.kg2.relation_triples_list
    ent_num = kgs.entities_num
    #adj=get_adj(triple_list, ent_num)
    #adj = normalize(adj + sp.eye(adj.shape[0]))
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj=get_sparse_tensor(triple_list, ent_num)
    print(adj)
    print(kgs.train_links)
    data = {'adj_train_norm': adj, 'features': features, 'idx_train': kgs.train_links, 'idx_val': (kgs.valid_entities1,kgs.valid_entities2), 'idx_test': (kgs.test_entities1,kgs.test_entities2)}
    return data


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


