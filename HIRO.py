import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gym
from gym.wrappers import AtariPreprocessing as atari
from gym.wrappers import FrameStack
import numpy as np

import random
import matplotlib.pyplot as plt
from collections import deque
import os

# Nachum, O., Gu, S., Lee, H., & Levine, S. (2018).
# Data-efficient hierarchical reinforcement learning. arXiv preprint arXiv:1805.08296.

# Hierarchical RL model. Simply playing with openai gym tasks.


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = th.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.body = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.avg_head = nn.Linear(hidden_size, num_actions)
        self.std_head = nn.Linear(hidden_size, num_actions)

        self.avg_head.weight.data.uniform_(-init_w, init_w)
        self.avg_head.bias.data.uniform_(-init_w, init_w)

        self.std_head.weight.data.uniform_(-init_w, init_w)
        self.std_head.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = self.body(state)

        mean = self.avg_head(x)
        log_std = self.std_head(x)
        log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = th.distributions.Normal(mean, std)
        z = normal.rsample()

        action = th.tanh(z)
        log_pi = (normal.log_prob(z) - th.log(1 - (action.pow(2) + epsilon))).sum(dim=1, keepdim=True)

        return action, log_pi

    def log_prob(self, state, action, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = th.distributions.Normal(mean, std)
        z = th.arctanh(action)

        log_prob = (normal.log_prob(z) - th.log(1 - (action.pow(2) + epsilon))).sum(dim=1).cpu()
        return log_prob


class SACAgent(nn.Module):
    def __init__(self, env, hi, gamma, tau, alpha, q_lr, policy_lr, a_lr, buffer_maxlen):
        super().__init__()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        self.env = env
        self.hi = hi
        self.action_range = [env.action_space.low, env.action_space.high]
        if hi:
            self.obs_dim = env.observation_space.shape[0]
            self.action_dim = self.obs_dim
        else:
            self.obs_dim = 2 * env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0]

        self.gamma = gamma
        self.tau = tau

        self.q_net1 = QNet(self.obs_dim, self.action_dim).to(self.device)
        self.q_net2 = QNet(self.obs_dim, self.action_dim).to(self.device)
        self.tgt_q_net1 = QNet(self.obs_dim, self.action_dim).to(self.device)
        self.tgt_q_net2 = QNet(self.obs_dim, self.action_dim).to(self.device)
        self.policy_net = PolicyNet(self.obs_dim, self.action_dim).to(self.device)

        for tgt_param, param in zip(self.tgt_q_net1.parameters(), self.q_net1.parameters()):
            tgt_param.data.copy_(param)

        for tgt_param, param in zip(self.tgt_q_net2.parameters(), self.q_net2.parameters()):
            tgt_param.data.copy_(param)

        self.q1_optimizer = th.optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = th.optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = th.optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.alpha = alpha
        self.tgt_entropy = -self.action_dim  # -th.prod(th.Tensor(self.env.action_space.shape[0]).to(self.device)).item()
        self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = th.optim.Adam([self.log_alpha], lr=a_lr)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def get_action(self, state):
        state = th.FloatTensor(state).unsqueeze(0).to(self.device)

        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()

        normal = th.distributions.Normal(mean, std)
        z = normal.rsample()
        action = th.tanh(z).detach().squeeze(0).cpu().numpy()

        return action

    def rescale_action(self, action):
        avg_action = (self.action_range[1] + self.action_range[0]) / 2
        return avg_action + avg_action * action

    def update(self, states, actions, rewards, next_states, dones):
        states = th.Tensor(states).to(self.device)
        actions = th.Tensor(actions).to(self.device)
        rewards = th.Tensor(rewards).to(self.device)
        next_states = th.Tensor(next_states).to(self.device)
        dones = th.FloatTensor(dones).to(self.device).unsqueeze(1)

        # Q loss
        next_actions, next_log_pis = self.policy_net.sample(next_states)
        next_q1 = self.tgt_q_net1(next_states, next_actions)
        next_q2 = self.tgt_q_net2(next_states, next_actions)
        next_tgt_q = th.min(next_q1, next_q2) - self.alpha * next_log_pis
        expected_q = rewards + (1 - dones) * self.gamma * next_tgt_q

        curr_q1 = self.q_net1(states, actions)
        curr_q2 = self.q_net2(states, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy loss
        new_actions, new_log_pi = self.policy_net.sample(states)
        tgt_q = th.min(self.q_net1(states, new_actions), self.q_net2(states, new_actions))
        policy_loss = (self.alpha * new_log_pi - tgt_q.detach()).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # tgt networks
        for tgt_param, param in zip(self.tgt_q_net1.parameters(), self.q_net1.parameters()):
            tgt_param.data.copy_(self.tau * param + (1 - self.tau) * tgt_param)

        for tgt_param, param in zip(self.tgt_q_net2.parameters(), self.q_net2.parameters()):
            tgt_param.data.copy_(self.tau * param + (1 - self.tau) * tgt_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-new_log_pi - self.tgt_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()


class HIRO:
    def __init__(self, env, gamma=.99, tau=.005, alpha=.1, q_lr=3e-4, policy_lr=3e-4, a_lr=3e-4, buffer_maxlen=5000):
        self.env = env

        self.hi_agent = SACAgent(self.env, hi=True, gamma=gamma, tau=tau, alpha=alpha, q_lr=q_lr, policy_lr=policy_lr, a_lr=a_lr, buffer_maxlen=buffer_maxlen)
        self.lo_agent = SACAgent(self.env, hi=False, gamma=gamma, tau=tau, alpha=alpha, q_lr=q_lr, policy_lr=policy_lr, a_lr=a_lr, buffer_maxlen=buffer_maxlen)

    def h(self, state, goal, next_state):
        return state + goal - next_state

    def intrinsic_reward(self, state, goal, next_state):
        return -np.linalg.norm(state + goal - next_state)

    def off_policy_correction(self, state_seq, action_seq):
        states = th.Tensor(state_seq[:-1])
        state_dim = states.size(-1)
        actions = th.Tensor(action_seq[1:])
        goal = action_seq[0]

        std = th.Tensor(.25 * (self.env.observation_space.high - self.env.observation_space.low)).clamp(-1e6, 1e6)
        loc = th.Tensor(state_seq[-1] - state_seq[0])
        dist = th.distributions.Normal(loc, std)

        candidate_goals = dist.sample(th.Size([8]))
        extras = th.Tensor([state_seq[-1] - state_seq[0], goal])
        candidate_goals = th.cat([candidate_goals, extras], dim=0)

        goals = [candidate_goals]
        for t in range(states.size(0) - 1):
            goals.append(self.h(states[t], goals[-1], states[t + 1]))

        goals = th.stack(goals, dim=1).reshape(-1, state_dim)
        states = th.cat([states.repeat(10, 1), goals], dim=1)
        actions = actions.repeat(10, 1)

        log_prob = self.lo_agent.policy_net.log_prob(states.cuda(), actions.cuda()).reshape(10, -1).sum(dim=1)
        max_goal = candidate_goals[log_prob.argmax()].numpy()
        return max_goal

    def train(self, max_episodes, max_steps, hi_update_interval, batch_size):
        episode_rewards = []

        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0

            for step in range(max_steps):
                init_goal = self.hi_agent.get_action(state)
                state_seq = [state]
                action_seq = [init_goal]
                hi_rewards = 0

                done = None
                goal = init_goal
                for t in range(hi_update_interval):
                    action = self.lo_agent.get_action(np.concatenate([state, goal]))
                    next_state, hi_reward, done, _ = self.env.step(self.lo_agent.rescale_action(action))

                    state_seq.append(next_state)
                    action_seq.append(action)
                    hi_rewards += hi_reward

                    next_goal = self.h(state, goal, next_state)
                    lo_reward = self.intrinsic_reward(state, goal, next_state)
                    self.lo_agent.replay_buffer.push(np.concatenate([state, goal]), action, lo_reward, np.concatenate([next_state, next_goal]), done)

                    state = next_state
                    goal = next_goal
                    episode_reward += hi_reward

                    self.env.render()

                    if len(self.lo_agent.replay_buffer) > batch_size:
                        states, actions, rewards, next_states, dones = self.lo_agent.replay_buffer.sample(batch_size)
                        self.lo_agent.update(states, actions, rewards, next_states, dones)

                    if done:
                        break

                self.hi_agent.replay_buffer.push(state_seq, action_seq, hi_rewards, state_seq[-1], done)

                if len(self.hi_agent.replay_buffer) > batch_size:
                    state_seqs, action_seqs, rewards, next_states, dones = self.hi_agent.replay_buffer.sample(batch_size)

                    states = []
                    actions = []
                    for state_seq, action_seq in zip(state_seqs, action_seqs):
                        actions.append(self.off_policy_correction(state_seq, action_seq))
                        states.append(state_seq[0])

                    self.hi_agent.update(states, actions, rewards, next_states, dones)

                if done or step == max_steps - 1:
                    episode_rewards.append(episode_reward)
                    break

        self.env.close()
        return episode_rewards


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    agent = HIRO(env)
    returns = agent.train(max_episodes=5, max_steps=1000, hi_update_interval=10, batch_size=64)
    # th.save(agent, 'HIRO_v0.00')
    print(returns)
