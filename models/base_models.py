"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from utils.eval_utils import acc_f1
from utils.train_utils import greedy_alignment

class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self,x, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)
        self.x=nn.Parameter(x)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], self.x], dim=1)
        h = self.encoder.encode(self.x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class LPModel(BaseModel):
    def __init__(self,x, args):
        super(LPModel, self).__init__(x,args)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.x=nn.Parameter(x)
        self.nsn=75

    def get_neg(self,ILL, embeddings, k,test_links):
      neg = []
      t = len(ILL)
      ILL_vec = embeddings[ILL]
      for i in range(t):
          sim=self.manifold.sqdist(ILL_vec[i], embeddings, self.c)
          #print(sim)
          rank = sim[i, :].argsort()
          #print(rank)
          neg.append(rank[0:k])

      neg = np.array(neg)
      neg = neg.reshape((t * k,))
      return neg
    def get_loss(self, embeddings,train_links,test_links):
        train_links = np.array(train_links)
        train_num=len(train_links)
        left = train_links[:, 0]
        right = train_links[:, 1]
        t = len(train_links)
        neg_num=k=75
        gamma=1
        #print(left)
        #print(right)
        left_x = embeddings[left]
        right_x = embeddings[right]
        sqdist = self.manifold.sqdist(left_x, right_x, self.c)

        #print(train_links.shape)
        #print(sqdist.shape)
        #A = torch.sum(sqdist)



        A = sqdist
        D=A+gamma

        pos = np.ones((train_num, neg_num)) * (train_links[:, 0].reshape((train_num, 1)))
        neg_left = pos.reshape((train_num * neg_num,))
        pos = np.ones((train_num, neg_num)) * (train_links[:, 1].reshape((train_num, 1)))
        neg2_right = pos.reshape((train_num * neg_num,))

        neg_right=[]
        for i in range(t):
          sim=self.manifold.sqdist(embeddings[left[i]].repeat(30000,1), embeddings, self.c)
          rank = sim.argsort()
          neg_right.append(rank[0:k])
        #print(neg_right)
        neg_right=torch.stack(neg_right)
        #print(neg_right)
        neg_right=torch.reshape(neg_right,(t * k,))
        neg_l_x = embeddings[neg_left]
        neg_r_x = embeddings[neg_right]

        B = self.manifold.sqdist(neg_l_x, neg_r_x, self.c)
        C = - torch.reshape(B, (t, k))
        #print(D)
        #print(C)
        L1 = torch.relu(torch.add(C, torch.reshape(D, (t, 1))))
        #print(L1)
        neg2_left=[]
        for i in range(t):
          sim=self.manifold.sqdist(embeddings[right[i]].repeat(30000,1), embeddings, self.c)
          rank = sim.argsort()
          neg2_left.append(rank[0:k])
        neg2_left=torch.stack(neg2_left)
        neg2_left=torch.reshape(neg2_left,(t * k,))

        neg2_l_x = embeddings[neg2_left]
        neg2_r_x = embeddings[neg2_right]

        E = self.manifold.sqdist(neg2_l_x, neg2_r_x, self.c)
        F = - torch.reshape(E, (t, k))

        L2 = torch.relu(torch.add(F, torch.reshape(D, (t, 1))))
        return (torch.sum(L1) + torch.sum(L2)) / (2.0 * k * t)

    def valid_metric(self,embeddings,ents):
        ent1,ent2=ents[0],ents[1]
        dis=[]
        ind=0
        #print(torch.cuda.memory_allocated(0))
        for e1 in ent1:
          # print(ind)
          #print(torch.cuda.memory_allocated(0))
          # ind+=1
          temp=self.manifold.sqdist(embeddings[e1], embeddings[ent2], self.c).detach().cpu().numpy()
          dis.append(temp)
        dis_mat=np.array(dis)
        return dis_mat

    def compute_metrics(self, embeddings, data, split):
        loss=None
        if split == 'train':
          train_links = np.array(data["idx_train"])
          test_links=np.array([data["idx_test"][0],data["idx_test"][1]])
          loss=self.get_loss(embeddings,train_links,test_links)
          torch.cuda.empty_cache()
        elif split == 'val':
          dis_mat=self.valid_metric(embeddings,data["idx_val"])
          greedy_alignment(dis_mat, 2)
          del dis_mat
        elif split == 'test':
          dis_mat=self.valid_metric(embeddings,data["idx_test"])
          greedy_alignment(dis_mat, 2)
          del dis_mat
        metrics = {'loss': loss}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

