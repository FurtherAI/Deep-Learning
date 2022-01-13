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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018, July).
# Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning (pp. 1861-1870). PMLR.

# Soft Actor Critic built to play asteroids, using openai gym.

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


class ACNet(nn.Module):
    def __init__(self, channels, num_actions, q_net, hidden_size=256, init_w=3e-3):
        super().__init__()
        self.q_net = q_net

        self.Q = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, hidden_size, kernel_size=10, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        self.Q[-1].weight.data.uniform_(-init_w, init_w)
        self.Q[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        if self.q_net:
            return self.Q(state)
        else:
            pis = F.softmax(self.Q(state), dim=1)
            pis = pis + th.full(pis.size(), 1e-8, device='cuda').where(pis == 0, th.zeros(pis.size(), device='cuda'))
            return pis


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


class SACAgent(nn.Module):
    def __init__(self, env, gamma, tau, alpha, q_lr, policy_lr, a_lr, buffer_maxlen):
        super().__init__()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        self.env = env
        # self.action_range = [env.action_space.low, env.action_space.high]
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n  # env.action_space.shape[0]

        self.gamma = gamma
        self.tau = tau

        self.q_net1 = ACNet(self.obs_dim, self.action_dim, True).to(self.device)
        self.q_net2 = ACNet(self.obs_dim, self.action_dim, True).to(self.device)
        self.tgt_q_net1 = ACNet(self.obs_dim, self.action_dim, True).to(self.device)
        self.tgt_q_net2 = ACNet(self.obs_dim, self.action_dim, True).to(self.device)
        self.policy_net = ACNet(self.obs_dim, self.action_dim, False).to(self.device)

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

        pi = th.distributions.Categorical(probs=self.policy_net(state).squeeze())
        action = pi.sample().detach().cpu().item()
        print(action, 'action')

        # mean, log_std = self.policy_net.forward(state)
        # std = log_std.exp()
        #
        # normal = th.distributions.Normal(mean, std)
        # z = normal.rsample()
        # action = th.tanh(z).detach().squeeze().cpu().numpy()

        return action

    # def rescale_action(self, action):
    #     avg_action = (self.action_range[1] + self.action_range[0]) / 2
    #     return avg_action + avg_action * action

    def update(self, batch_size):
        # print(th.any(self.policy_net.Q[0].weight.data.isnan()))

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = th.FloatTensor(states).to(self.device)  # may need to convert to numpy before torch
        actions = th.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = th.FloatTensor(rewards).to(self.device)
        next_states = th.FloatTensor(next_states).to(self.device)  # may need to convert to numpy before torch
        dones = th.FloatTensor(dones).to(self.device)
        dones = dones.unsqueeze(1)

        # Q loss
        next_pis = self.policy_net(next_states)  # next_actions, next_log_pis = self.policy_net.sample(next_states)
        next_q1 = self.tgt_q_net1(next_states)  # continuous - (next_actions, next_states)
        next_q2 = self.tgt_q_net2(next_states)
        next_tgt_q = next_pis * (th.min(next_q1, next_q2) - self.alpha * th.log(next_pis))
        expected_q = rewards + (1 - dones) * self.gamma * (next_tgt_q.sum(dim=1).unsqueeze(1))  # continuous - exclude expectation

        curr_q1 = self.q_net1(states).gather(1, actions)
        curr_q2 = self.q_net2(states).gather(1, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy loss
        new_pis = self.policy_net(states)  # new_actions, new_log_pi = self.policy_net.sample(states)]
        tgt_q = th.min(self.q_net1(states), self.q_net2(states))
        policy_loss = (new_pis * (self.alpha * th.log(new_pis) - tgt_q.detach())).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # tgt networks
        for tgt_param, param in zip(self.tgt_q_net1.parameters(), self.q_net1.parameters()):
            tgt_param.data.copy_(self.tau * param + (1 - self.tau) * tgt_param)

        for tgt_param, param in zip(self.tgt_q_net2.parameters(), self.q_net2.parameters()):
            tgt_param.data.copy_(self.tau * param + (1 - self.tau) * tgt_param)

        # update temperature
        # alpha_loss = (new_pis.detach() * (self.log_alpha * (-th.log(new_pis) - self.tgt_entropy).detach())).sum(dim=1).mean()
        alpha_loss = (self.log_alpha * (-th.log(new_pis) - self.tgt_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()


def mini_batch_train(env, agent, max_episodes, max_steps, gradient_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            print(step, 'step')
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                for grad_step in range(gradient_steps):
                    agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                break

            if step % 100 == 0:
                th.save(agent, 'sac-asteroids_v0.00')

            state = next_state
            print(episode_reward, 'return')
            env.render()

    env.close()
    return episode_rewards


def make_env(env):
    env = gym.make(env)
    env = atari(env, frame_skip=5, scale_obs=True)
    env = FireResetEnv(env)
    env = FrameStack(env, 4)
    return env


if __name__ == '__main__':
    env = make_env('AsteroidsNoFrameskip-v4')
    # agent = SACAgent(env, gamma=.99, tau=.005, alpha=1, q_lr=3e-4, policy_lr=3e-4, a_lr=3e-4, buffer_maxlen=5000)
    agent = th.load('sac-asteroids_v0.00')
    returns = mini_batch_train(env, agent, max_episodes=15, max_steps=3000, gradient_steps=1, batch_size=64)
    th.save(agent, 'sac-asteroids_v0.00')
    print(returns)
    # plt.plot(np.arange(0, len(returns)), returns)
    # plt.show()

# discrete action space (if applicable to