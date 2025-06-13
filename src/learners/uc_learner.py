import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.uc_mix import UC_Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
import torch.nn as nn

def parameters_num(param_list):
    return sum(p.numel() for p in param_list) / 1000

class UCLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        elif args.mixer == "uc_mix":
            self.mixer = UC_Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        print('All net size')
        #print('map=', args.env_args['map_name'])
        print('learner=', args.learner)
        mix_net_size = parameters_num(self.mixer.parameters())
        agent_net_size = parameters_num(self.mac.parameters())
        print(str(mix_net_size)+'K' + '//' + str(agent_net_size)+'K')
        print(str(mix_net_size + agent_net_size)+'K')

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        obs_next = batch["obs"][:, 1:]
        mask_next = batch["filled"][:, 1:].float()
        mask_next[:, 1:] = mask_next[:, 1:] * (1 - terminated[:, 1:])
        actions_onehot = batch["actions_onehot"][:, :-1]
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        hidden_state = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, hi = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            hidden_state.append(hi)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        hidden_state = th.stack(hidden_state, dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            target_hidden_state = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, target_hi = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
                target_hidden_state.append(target_hi)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
            target_hidden_state = th.stack(target_hidden_state, dim=1)

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals, target_Gs_nn, target_eye_mask = self.target_mixer(target_max_qvals, batch["state"], target_hidden_state)

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals, target_Gs_nn, target_eye_mask = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals, Gs_nn, eye_mask = self.mixer(chosen_action_qvals, batch["state"][:, :-1], hidden_state[:, :-1])
        b, t, n, _ = Gs_nn.shape
        tao_t = hidden_state[:, :-1]
        influence_inputs = []
        obs_next_no_i = []
        '''for i in range(self.args.n_agents):
            hidden_out_without_i = th.cat((tao_t[:, :, :i], tao_t[:, :, i + 1:]), dim=2) #(b,t,n-1,64)
            actions_onehot_without_i = th.cat((actions_onehot[:, :, :i], actions_onehot[:, :, i + 1:]), dim=2)  # (b,t,n-1,action_dim)
            actions_onehot_i = actions_onehot[:, :, i].unsqueeze(-2).repeat(1, 1, self.args.n_agents - 1, 1)  # (b,t,n-1,action_dim)
            influence_input = th.cat((hidden_out_without_i, actions_onehot_without_i, actions_onehot_i), dim=-1) #(b,t,n-1,fs)
            obs_next_without_i = th.cat((obs_next[:, :, :i], obs_next[:, :, i + 1:]), dim=2) #(b,t,n-1,obs_fs)
            influence_inputs.append(influence_input)
            obs_next_no_i.append(obs_next_without_i)

        influence_inputs = th.stack(influence_inputs, dim=-3)  #(b,t,n,n-1,fs1)
        obs_next_no_i = th.stack(obs_next_no_i, dim=-3) #(b,t,n,n-1,fs2)'''

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)


        Gs_nn_no_i = Gs_nn[~eye_mask].view(b,t,-1)
        #print('Gs_nn_no_i.shape:', Gs_nn_no_i.shape)  #Gs_nn_no_i.shape: torch.Size([179200]) -> Gs_nn_no_i.shape: torch.Size([128, 70, 20])
        temp_mask = mask.expand_as(Gs_nn_no_i)
        masked_abs_Gs_nn_no_i = th.abs(Gs_nn_no_i) * temp_mask

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        '''pre_mask = mask.unsqueeze(-1).unsqueeze(-1).expand(b, t, n, n - 1, obs_next.shape[-1])  # (b,t,n,n-1,fs2)
        pre_obs_next = self.mixer.pre_net(influence_inputs)
        criterion = nn.HuberLoss(reduction='mean', delta=1.0)
        pre = pre_obs_next * pre_mask  # (b,t,n,n-1,fs2)
        target = obs_next_no_i * pre_mask  # (b,t,n,n-1,fs2)
        loss_pre = criterion(pre.view(b * t * n, n - 1, -1), target.view(b * t * n, n - 1, -1))'''



        #sum_chosen_action_qvals = (chosen_action_qvals*mask).sum() / mask.sum()
        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        L_sd = masked_abs_Gs_nn_no_i.sum() / temp_mask.sum()
        #loss = L_td = masked_td_error.sum() / mask.sum() + self.args.pre_alpha * loss_pre + self.args.sd_alpha * L_sd #- self.args.alph_*sum_chosen_action_qvals
        loss = L_td = masked_td_error.sum() / mask.sum() + self.args.sd_alpha * L_sd  # - self.args.alph_*sum_chosen_action_qvals

        #print('loss_pre:',loss_pre) #tensor(0.0433, device='cuda:0', grad_fn=<HuberLossBackward0>)
        #print('obs_next_no_i:',obs_next_no_i.shape)  #obs_next_no_i: torch.Size([128, 70, 5, 4, 55])
        #self.args.sd_alpha=0.2
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("L_sd", L_sd.item(), t_env)
            self.logger.log_stat("sd_alpha * L_sd", self.args.sd_alpha * L_sd.item(), t_env)
            #self.logger.log_stat("loss_pre", loss_pre.item(), t_env)
            #self.logger.log_stat("pre_alpha * loss_pre", self.args.pre_alpha * loss_pre.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
