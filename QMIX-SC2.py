import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from smac.env import StarCraft2Env

from copy import deepcopy as dc
from functools import partial
from types import SimpleNamespace as SN

# Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018, July).
# Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning. In International Conference on Machine Learning (pp. 4295-4304). PMLR.

# QMIX is a multi agent RL model. It is being applied here to learn Star Craft 2.


class OneHot:
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32


class DecayThenFlatSchedule:

    def __init__(self, start, finish, time_length, decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))


class EpsilonGreedyActionSelector:

    def __init__(self, epsilon_start, epsilon_finish, epsilon_anneal_time):
        self.schedule = DecayThenFlatSchedule(epsilon_start, epsilon_finish, epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_outs, avail_actions, t_env):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        # mask actions that are excluded from selection
        masked_q_values = agent_outs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_outs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = th.distributions.Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device='cpu'):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length, self.preprocess)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())


class RNNAgent(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        hidden = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, hidden)
        q = self.fc2(h)

        return q, h


class BasicMAC:
    def __init__(self, scheme, n_actions, n_agents, action_selector):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)

        self.action_selector = action_selector

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None)):
        avail_actions = ep_batch['avail_actions'][:, t_ep]
        agent_outs = self.forward(ep_batch, t_ep)
        chosen_actions = self.action_selector.select_action(agent_outs[bs], avail_actions[bs], t_env)
        return chosen_actions

    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch['avail_actions'][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
        agent_outs[reshaped_avail_actions == 0] = -1e10

        agent_outs = F.softmax(agent_outs, dim=-1)
        epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
        agent_outs = (1 - self.action_selector.epsilon) * agent_outs + th.ones_like(agent_outs) * (self.action_selector.epsilon / epsilon_action_num)
        agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def _build_agents(self, input_shape):
        self.agent = RNNAgent(input_shape, 256, self.n_actions)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch['obs'][:, t])

        if t == 0:
            inputs.append(th.zeros_like(batch['actions_onehot'][:, t]))
        else:
            inputs.append(batch['actions_onehot'][:, t - 1])

        inputs.append(th.eye(self.n_agents, device='cuda').unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        input_shape += self.n_actions
        input_shape += self.n_agents

        return input_shape


class QMixer(nn.Module):
    def __init__(self, state_dim, n_agents, embed_dim, hypernet_embed):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim * n_agents)
        )
        self.hyper_wf = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim)
        )

        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.V = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, states, agent_qs):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        # First Layer
        w1 = th.abs(self.hyper_w1(states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # Second Layer
        wf = th.abs(self.hyper_wf(states)).view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)

        # Out
        q_tot = (th.bmm(hidden, wf) + v).view(bs, -1, 1)
        return q_tot


class QLearner:
    def __init__(self, mac, mixer, tgt_update_interval, gamma, lr):
        self.mac = mac
        self.mixer = mixer
        self.gamma = gamma

        self.tgt_mac = dc(mac)
        self.tgt_mixer = dc(mixer)

        self.last_tgt_update_ep = 0
        self.tgt_update_interval = tgt_update_interval

        self.params = list(mac.parameters()) + list(mixer.parameters())
        self.optimizer = th.optim.RMSprop(self.params, lr)

    def train(self, batch, ep_number):
        rewards = batch['reward'][:, :-1]
        actions = batch['actions'][:, :-1]
        terminated = batch['terminated'][:, :-1].float()
        mask = batch['filled'][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch['avail_actions']

        # Estimated Qs
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Tgt Qs
        tgt_mac_out = []
        self.tgt_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            tgt_agent_outs = self.tgt_mac.forward(batch, t=t)
            tgt_mac_out.append(tgt_agent_outs)

        # tgt_outs = tgt_mac_out[1:]
        # if not isinstance(tgt_outs, list):
        #     tgt_outs = list(tgt_outs)

        tgt_mac_out = th.stack(tgt_mac_out[1:], dim=1)
        tgt_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over tgt Qs
        mac_out_detach = mac_out.clone().detach()
        mac_out_detach[avail_actions == 0] = -9999999
        cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        tgt_max_qvals = th.gather(tgt_mac_out, dim=3, index=cur_max_actions).squeeze(3)

        # Mix
        chosen_action_qvals = self.mixer(batch['state'][:, :-1], chosen_action_qvals)
        tgt_max_qvals = self.tgt_mixer(batch['state'][:, :-1], tgt_max_qvals)

        # 1 step Q-Learning targets
        targets = rewards + self.gamma * (1 - terminated) * tgt_max_qvals

        # Td error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)
        masked_td_error = mask * td_error

        # loss
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if ep_number - self.last_tgt_update_ep >= self.tgt_update_interval:
            self._update_targets()
            self.last_tgt_update_ep = ep_number

    def _update_targets(self):
        self.tgt_mac.load_state(self.mac)
        self.tgt_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.mac.cuda()
        self.tgt_mac.cuda()
        self.mixer.cuda()
        self.tgt_mixer.cuda()


class EpisodeRunner:

    def __init__(self, env):
        self.batch_size = 1

        self.env = env
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device='cuda')
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env)
        self.batch.update({"actions": actions}, ts=self.t)

        self.t_env += self.t

        return self.batch


def main():
    t_max = 1000
    batch_size = 32
    env = StarCraft2Env('5m_vs_6m')
    # Init runner so we can get env info
    runner = EpisodeRunner(env)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, buffer_size=1000, max_seq_length=(env_info["episode_limit"] + 1),
                          preprocess=preprocess,
                          device="cuda")

    # Setup multiagent controller here
    action_selector = EpsilonGreedyActionSelector(1, .05, 50000)  # epsilon_start, epsilon_finish, epsilon_anneal_time):
    mac = BasicMAC(scheme=scheme, n_actions=n_actions, n_agents=n_agents, action_selector=action_selector)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    mixer = QMixer(state_dim=int(np.prod(state_shape)), n_agents=n_agents, embed_dim=32, hypernet_embed=64)
    learner = QLearner(mac=mac, mixer=mixer, tgt_update_interval=200, gamma=.99, lr=1e-4)

    learner.cuda()

    # start training
    episode = 0

    while runner.t_env <= t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run()
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(batch_size):
            episode_sample = buffer.sample(batch_size)

            # Truncate batch to only filled timesteps
            # max_ep_t = episode_sample.max_t_filled()
            # episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != 'cuda':
                episode_sample.to('cuda')

            learner.train(episode_sample, episode)

        episode += 1

    runner.close_env()

if __name__ == '__main__':
    main()


# Instantiate:
# Qlearner - includes the mixer, along with agents.
# Multi agent controller
# Episode runner
# Buffer
# Smac environment

# need to repurpose all arguments to function without the internal arguments
# write main loop
