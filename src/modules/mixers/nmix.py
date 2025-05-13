import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from modules.mixers.transformer_module import TransformerModule

class Encoder_s(nn.Module):
    def __init__(self, state_shape, hidden_dim=64):
        super(Encoder_s, self).__init__()
        self.fc1 = nn.Linear(state_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, s):
        emb_s = F.relu(self.fc1(s), inplace=True)
        out = F.tanh(self.fc2(emb_s)+emb_s)
        return out

class Mixer(nn.Module):
    def __init__(self, args, abs=True):
        super(Mixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape)) 

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        self.encoder_s = Encoder_s(self.state_dim,64)

        #hyper Ni weight
        '''self.hyper_Ni_w = nn.ModuleList(nn.Sequential(nn.Linear(self.state_dim+self.args.hypernet_embed, 64),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(64, 1 * self.embed_dim))
                                      for i in range(self.n_agents))'''
        self.transformer = TransformerModule(64,64)
        '''self.fc1_Ni = nn.Sequential(
            nn.Linear(self.args.hypernet_embed * self.n_agents, self.args.hypernet_embed),
            nn.ReLU(inplace=True),
        )'''
        self.fc2_Ni = nn.ModuleList(nn.Sequential(nn.Linear(self.args.hypernet_embed, 1 * (self.n_agents-1)),)
                                        for i in range(self.n_agents))
        self.norm = nn.LayerNorm(1 * (self.n_agents-1))
        self.sigmoid = nn.Sigmoid()

        # hyper w1 b1   self.input_dim
        self.hyper_w1 = nn.Sequential(nn.Linear(64+64, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(64+64, self.embed_dim))
        
        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(64 + 64, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(64 + 64, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))

        '''self.pre_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim + args.n_actions + args.n_actions, 64),
                            nn.ReLU(inplace=True),
                            nn.Linear(64, 64),
                            nn.ReLU(inplace=True),
                            nn.Linear(64, args.obs_shape),
                            )'''

        self.pre_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim + args.n_actions + args.n_actions, 64),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(64, args.obs_shape),
                                     )


        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, states, hidden_state=None):
        # reshape
        b, t, _ = qvals.size()  #([128, 71, 5])
        states = states.reshape(-1, self.state_dim)
        encoder_s = self.encoder_s(states)  # (b*t,64)
        encoder_s_detach = encoder_s.detach().unsqueeze(-2)  # (b*t,1,64)

        #print('qvals.shape',qvals.shape)
        #state_hidden_detach = th.cat((states, hidden_state.detach()), dim=-
        hidden_detach = hidden_state.detach().reshape(-1,self.n_agents,64)
        eye_mask = th.eye(self.n_agents, dtype=th.bool).unsqueeze(0).unsqueeze(0).expand(b, t, self.n_agents, self.n_agents)

        eye_temp = th.eye(self.n_agents, device=qvals.device).unsqueeze(0).unsqueeze(0).expand(b, t, self.n_agents, self.n_agents)

        #print('hidden_detach.shape:',hidden_detach.shape)  #torch.Size([128, 71, 5, 64])
        #Ni_1 = self.fc1_Ni(hidden_detach.reshape(b * t, -1)).view(b*t,-1)  #(b,t,n_agents * obs_fs) -> (b,t,32)
        h_cat_ens = th.cat((hidden_detach,encoder_s_detach),dim=-2)  #(-1,n+1,64)
        h_trans = self.transformer(h_cat_ens)[:,:-1].view(-1,self.n_agents,64) #(-1,n+1,64) -> #(-1,n,64)
        en_s_h_trans = th.cat((encoder_s,h_trans.detach().mean(-2)), dim=-1) #(-1,64 + 64)
        N_list = []
        for i in range(self.n_agents):
            Ni_temp = self.fc2_Ni[i](h_trans[:,i]).view(b,t,1,-1)  # (b*t, 32)-->(b,t,1,n) #b,l,n,32
            Ni_temp = self.norm(Ni_temp)
            Ni_temp = self.sigmoid(Ni_temp)
            #print('Ni_temp[:,:,:,:i].shape:',Ni_temp[:,:,:,:i].shape)  #Ni_temp[:,:,:,:i].shape: torch.Size([128, 71, 1, 0]) ...
            Ni_2 = th.cat((Ni_temp[:,:,:,:i],eye_temp[:,:,i,i].view(b,t,1,1),Ni_temp[:,:,:,i:]),dim=-1)
            #print('Ni_2.shape:',Ni_2.shape)  #Ni_2.shape: torch.Size([128, 71, 1, 5])
            #print(Ni_2[i,i,:,:])
            N_list.append(Ni_2)
        N_nn = th.stack(N_list, dim=-2).view(b,t, self.n_agents, -1)  # (b*l,6,32)
        #print('N_nn.shape:',N_nn.shape)  #N_nn.shape: torch.Size([128, 71, 5, 5])
        #N_nn_clone = th.abs(N_nn.clone())  #add abs operation for all N_nn elements
        #N_nn_clone = N_nn.clone()
        #N_nn_clone[eye_mask] = th.abs(N_nn[eye_mask])

        temp = qvals.unsqueeze(dim=-2).repeat(1, 1, self.n_agents, 1)
        '''temp_detach = temp.detach()
        temp[~eye_mask] = temp_detach[~eye_mask]'''

        co_idv_q_taken_t = temp * N_nn  # torch.Size([12, 1, 8, 8])
        # print('co_idv_q_taken_t.shape:', co_idv_q_taken_t.shape)  #(12,1,8,8)  co_idv_q_taken_t.shape: torch.Size([128, 70, 5, 5])
        sum_co_idv_q_taken_t = co_idv_q_taken_t.sum(dim=-1)  # (bz,1,n_agents)=(12,1,8)
        # print('sum_co_idv_q_taken_t.shape:', sum_co_idv_q_taken_t.shape)  #(12,1,8)
        #mean_co_idv_q_taken_t = sum_co_idv_q_taken_t / eye_G_s.sum(dim=-1)  # 保留梯度 (12,1,8)

        qvals_Ni = sum_co_idv_q_taken_t.reshape(b * t, 1, self.n_agents)    #qvals.reshape(b * t, 1, self.n_agents)

        #state_hidden_detach = state_hidden_detach.reshape(-1,self.state_dim + self.args.hypernet_embed)

        # First layer
        w1 = self.hyper_w1(en_s_h_trans.detach()).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
        b1 = self.hyper_b1(en_s_h_trans.detach()).view(-1, 1, self.embed_dim)
        
        # Second layer
        w2 = self.hyper_w2(en_s_h_trans).view(-1, self.embed_dim, 1) # b * t, emb, 1
        b2= self.hyper_b2(en_s_h_trans).view(-1, 1, 1)
        
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
            
        # Forward
        hidden = F.elu(th.matmul(qvals_Ni, w1) + b1) # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2 # b * t, 1, 1

        #temp = y.unsqueeze(dim=-2).repeat(1, self.n_agents, 1)

        return y.view(b, t, -1), N_nn.view(b, t, self.n_agents, -1), eye_mask.view(b, t, self.n_agents, -1)  #(b,t,n,1),(b,t,n,n),(b,t,n,n)

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)

    def pre_obs_next(self, hi):  #b, t, n, fs
        b, t, n, n_1, _ = hi.size()
        predict_obs_next = self.pre_net(hi.reshape(b*t*n,-1))
        return predict_obs_next.view(b,t,n,n-1,-1)




'''
        loss = loss.sum(dim=-1, keepdim=False)  # bs,t , self.n_agents-1
        loss = loss.sum(dim=-1, keepdim=True)  # bs,t , 1
        loss = (loss * mask).sum() / mask.sum()'''




        
