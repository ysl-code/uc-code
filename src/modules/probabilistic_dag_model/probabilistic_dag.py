import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gumbel_softmax import *
from .soft_sort import gumbel_sinkhorn
from .sinkhorn_net import Sinkhorn_Net

class ProbabilisticDAG(nn.Module):

    def __init__(self, n_nodes, input_dim, hidden_dim, temperature=1.0, hard=True, noise_factor=1.0, device = 'cpu'): #deep_set, set_transformer
        """Base Class for Probabilistic DAG Generator based on topological order sampling

        Args:
            n_nodes (int): Number of nodes
            temperature (float, optional): Temperature parameter for order sampling. Defaults to 1.0.
            hard (bool, optional): If True output hard DAG. Defaults to True.
            noise_factor (float, optional): Noise factor for Sinkhorn sorting. Defaults to 1.0.
        """
        super().__init__()

        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.hard = hard
        self.device = device
        # Mask for ordering
        self.mask = torch.triu(torch.ones(self.n_nodes, self.n_nodes), 1).to(self.device)  #创建上三角矩阵

        # define initial parameters
        self.noise_factor = noise_factor
        
        #permutation network
        self.permutation_net = Sinkhorn_Net(self.input_dim, self.hidden_dim, self.n_nodes)
        
        #egde network
        self.edge_net = nn.Sequential(
                nn.Linear(self.input_dim*self.n_nodes, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.n_nodes*self.n_nodes*2),
        )
        
        self.gs = GumbleSoftmax(device=self.device,temp=temperature).to(self.device)
        self.to(self.device)

    def sample_edges(self, obs, test):
        logits = self.edge_net(obs).view(-1,2)  #(n_nodes*n_nodes, 2)
        dag= self.gs(logits, force_hard = True, test = test)  #（n_agents*nagents,2）
        dag = dag[:,0]  #提取每个元素的第一个值，表示边的存在性
        dag = dag.view(-1, self.n_nodes, self.n_nodes)  #(1,3,3)
        return dag

    def sample_permutation(self, obs, test):
        #perm_weights = self.permutation_net(obs)
        #log_alpha = F.logsigmoid(perm_weights) #do we need this log.sigmoid?
        #P = gumbel_sinkhorn(log_alpha, noise_factor=self.noise_factor, temp=self.temperature, hard=self.hard, test = test)
        batch_size = len(obs)  #这里的batch_size指的应该是并行采样的环境数量
        P = torch.eye(self.n_nodes).view(1, self.n_nodes, self.n_nodes).repeat(batch_size, 1, 1).to(self.device)
        return P

    def sample(self, obs, test):
        P = self.sample_permutation(obs, test = test)  #(batch_size,n_nodes,n_nodes)这是一个单位阵
        P_inv = P.transpose(1, 2)  #转置矩阵
        obs = obs.view(len(obs), -1)  #（1,1904）
        U = self.sample_edges(obs, test = test)        
        dag_adj = U * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking
        return P, U, dag_adj  #这里设计这个P感觉没有必要啊，就是一个n*n的单位向量，根本没有必要学习



class DummyDAG(nn.Module):

    def __init__(self, n_nodes, threshold = 1, device = 'cpu'): #deep_set, set_transformer
        """Base Class for Probabilistic DAG Generator based on topological order sampling

        Args:
            n_nodes (int): Number of nodes
            temperature (float, optional): Temperature parameter for order sampling. Defaults to 1.0.
            hard (bool, optional): If True output hard DAG. Defaults to True.
            noise_factor (float, optional): Noise factor for Sinkhorn sorting. Defaults to 1.0.
        """
        super().__init__()

        self.n_nodes = n_nodes
        self.threshold = threshold
        self.device = device
        # Mask for ordering
        self.mask = torch.triu(torch.ones(self.n_nodes, self.n_nodes), 1).to(self.device)
        self.to(self.device)

    def get_random_permutation(self, batch_size):
        Ps = []
        idx = torch.randperm(self.n_nodes)
        for j in range(batch_size):
            P = torch.zeros(self.n_nodes, self.n_nodes)
            idx = torch.randperm(self.n_nodes)
            for i in range(self.n_nodes):
                P[i][idx[i]]=1
            Ps.append(P)
        Ps = torch.stack(Ps)
        return Ps
        
    def get_random_edge_matrix(self, batch_size):
        random_edge_matrix = torch.ones(batch_size, self.n_nodes, self.n_nodes) * self.threshold
        return torch.bernoulli(random_edge_matrix)

    def sample_edges(self, obs, fixed_noise = None):
        batch_size = len(obs)
        dag = self.get_random_edge_matrix(batch_size)
        return dag.to(self.device)

    def sample_permutation(self, obs, fixed_noise = None):
        batch_size = len(obs)
        #P = self.get_random_permutation(batch_size)
        P = torch.eye(self.n_nodes).view(1, self.n_nodes, self.n_nodes).repeat(batch_size, 1, 1)
        return P.to(self.device)

    def sample(self, obs, fixed_edge_noise = None, fixed_permutation_noise = None, test = False):
        P = self.sample_permutation(obs, fixed_noise = fixed_permutation_noise)
        P_inv = P.transpose(1, 2)
        U = self.sample_edges(obs, fixed_noise = fixed_edge_noise)
        dag_adj = U * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking  dag_adj=U*self.mask,其中mask是上三角矩阵，从而保证了生成的邻接矩阵是有向无环图
        return P, U, dag_adj
