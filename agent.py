#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Implements a DDPG Agent

    Args:
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        device (str, optional): device used for tensor operations
        buffer_size (int, optional): size of the experience replay buffer
        batch_size (int, optional): size of batch sampled for experience replay
        lr (float, optional): learning rate of both actor and critic models
        lr_steps (int, optional): number of steps between each scheduler step
        lr_gamma (float, optional): LR multiplier applied at each scheduler step
        gamma (float, optional): discount factor
        tay (float, optional): soft update rate
        noise_mean (float, optional): mean of Ornstein-Uhlenbeck process
        noise_theta (float, optional): theta parameter Ornstein-Uhlenbeck process
        noise_sigma (float, optional): sigma parameter of Ornstein-Uhlenbeck process
        grad_clip (float, optional): gradient clip
    """

    def __init__(self, state_size, action_size, train=False, device=None, buffer_size=1e6, batch_size=128,
                 lr=1e-3, gamma=0.99, tau=1e-3, update_freq=20, nb_updates=10,
                 noise_mean=0, noise_theta=0.05, noise_sigma=0.15, eps=1.0, eps_decay=1e-6,
                 grad_clip=1.0):

        self.state_size = state_size
        self.action_size = action_size
        self.train = train
        self.bs = batch_size
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.update_freq = update_freq
        self.nb_updates = nb_updates
        self.eps = eps
        self.eps_decay = eps_decay

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
        self.device = torch.device(device)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(self.device)
        if self.train:
            self.actor_target = Actor(state_size, action_size).to(self.device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size).to(self.device)
        if self.train:
            self.critic_target = Critic(state_size, action_size).to(self.device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr, weight_decay=0.)

            # Noise process
            self.noise = OUNoise(action_size, noise_mean, noise_theta, noise_sigma)

            # Replay memory
            self.memory = ReplayBuffer(action_size, int(buffer_size), batch_size, self.device)

    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        if not self.train:
            raise ValueError('agent cannot be trained if constructor argument train=False')

        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.bs and timestep % self.update_freq == 0:
            for _ in range(self.nb_updates):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Resolves action for given state as per current policy.

        Args:
            state (numpy.ndarray): current state representation
            add_noise (bool, optional): should noise be add to action value
        Returns:
            numpy.ndarray: clipped action value
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        if self.train:
            self.actor_local.train()

            if add_noise:
                action += self.eps * self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        if not self.train:
            raise ValueError('agent cannot be trained if constructor argument train=False')
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions.to(dtype=torch.float32))
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # gradient clipping for critic
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        # --------------------- and update epsilon decay ----------------------- #
        if self.eps_decay > 0:
            self.eps -= self.eps_decay
            self.noise.reset()

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Implements a noise sampler based on Ornstein-Uhlenbeck process.

    Args:
        size (int): expected 1-dimension size of noise
        mu (float, optional): mean of the distribution
        theta (float, optional): theta parameter of OU process
        sigma (float, optional): sigma parameter of OU process
    """

    def __init__(self, size, mu=0., theta=0.05, sigma=0.15):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample.

        Returns:
            numpy.ndarray: noisy state
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Args:
        action_size (int): dimension of each action
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        device (torch.device): device to use for tensor operations
    """

    def __init__(self, action_size, buffer_size, batch_size, device):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory.

        Returns:
            tuple: tuple of vectorized sampled experiences
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
