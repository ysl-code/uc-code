from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
from modules.cb.consensus_builder import ConsensusBuilder
from modules.embedding.embedding_net import Embedding_net
import torch.nn.functional as F
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class COLANMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(COLANMAC, self).__init__(scheme, groups, args)
        input_shape = self._get_input_shape(scheme)
        if self.args.input == 'hidden':
            self._build_consensus_builder(self.args.rnn_hidden_dim)
        elif self.args.input == 'obs':
            self._build_consensus_builder(input_shape)
        self._build_embedding_net()
        self.obs_center = th.zeros(1, self.args.consensus_builder_dim).cuda()
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
        agent_inputs = self._build_inputs(ep_batch, t) #(b,n,fs)
        avail_actions = ep_batch["avail_actions"][:, t]
        self.hidden_states = self.agent.calc_hidden(agent_inputs, self.hidden_states) #(b,n,64)

        with th.no_grad():
            if self.args.input == 'hidden':
                latent_state = self.consensus_builder.calc_student(self.hidden_states)
            elif self.args.input == 'obs':
                latent_state = self.consensus_builder.calc_student(agent_inputs) #(128,3,32)

            # latent_state = latent_state - latent_state.max(-1, keepdim=True)[0].detach()
            latent_state_id = F.softmax(latent_state, dim=-1).detach().max(-1)[1].unsqueeze(-1)
            latent_state_id[
                ep_batch['alive_allies'][:, t].reshape(*latent_state_id.size()) == 0] = self.args.consensus_builder_dim
            latent_state_embedding = self.embedding_net(latent_state_id.squeeze(-1)) #(b,n,4)
        agent_outs = self.agent.calc_value(latent_state_embedding, self.hidden_states)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def parameters(self):
        return list(self.agent.parameters()) + list(self.embedding_net.parameters())

    def consensus_builder_update_parameters(self):
        return self.consensus_builder.update_parameters()

    def consensus_builder_all_parameters(self):
        return self.consensus_builder.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.embedding_net.load_state_dict(other_mac.embedding_net.state_dict())

    def load_consensus_builder_state(self, other_mac):
        self.consensus_builder.load_state_dict(other_mac.consensus_builder.state_dict())
        self.obs_center = other_mac.obs_center.detach().clone()

    def cuda(self):
        self.agent.cuda()
        self.consensus_builder.cuda()
        self.embedding_net.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.consensus_builder.state_dict(), "{}/consensus_builder.th".format(path))
        th.save(self.embedding_net.state_dict(), "{}/embedding.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.consensus_builder.load_state_dict(
            th.load("{}/consensus_builder.th".format(path), map_location=lambda storage, loc: storage))
        self.embedding_net.load_state_dict(
            th.load("{}/embedding.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_embedding_net(self):
        self.embedding_net = Embedding_net(self.args)

    def _build_consensus_builder(self, obs_shape):
        state_dim = int(np.prod(self.args.state_shape))
        self.consensus_builder = ConsensusBuilder(state_dim, obs_shape, self.args)
