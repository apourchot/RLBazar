import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

from drl.models import GaussianActor, CriticTD3 as Critic, ValueTD3 as Value


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class Virel(object):
    """
    VIREL-inspired Actor-Critic Algorithm: https://arxiv.org/abs/1811.01132
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Actor stuff
        self.actor = GaussianActor(state_dim, action_dim, max_action, args)
        self.actor_t = GaussianActor(state_dim, action_dim, max_action, args)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)

        # Critic stuff
        self.critic = Critic(state_dim, action_dim, max_action, args)
        self.critic_t = Critic(state_dim, action_dim, max_action, args)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # Value stuff
        self.value = Value(state_dim, action_dim, max_action, args)
        self.value_t = Value(state_dim, action_dim, max_action, args)
        self.value_t.load_state_dict(self.value.state_dict())
        self.value_opt = torch.optim.Adam(
            self.value.parameters(), lr=args.critic_lr)

        # Env stuff
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Hyperparams
        self.tau = args.tau
        self.n_steps = args.n_steps
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.policy_noise = args.policy_noise
        self.reward_scale = args.reward_scale
        self.weights = FloatTensor(
            [self.discount ** i for i in range(self.n_steps)])

        # cuda
        if args.use_cuda:
            self.actor.cuda()
            self.actor_t.cuda()
            self.critic.cuda()
            self.critic_t.cuda()
            self.value.cuda()
            self.value_t.cuda()

    def action(self, state):
        """
        Returns action given state
        """
        state = FloatTensor(state.reshape(1, -1))
        action, mu, sigma = self.actor(state)
        # print("mu", mu)
        # print("sigma", sigma ** 2)
        return action.cpu().data.numpy().flatten()

    def train(self, memory, n_iter):
        """
        Trains the model for n_iter stepsd
        """

        for it in range(n_iter):

            # Sample replay buffer
            states, actions, n_states, rewards, steps, dones, stops = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards * self.weights
            rewards = rewards.sum(dim=1, keepdim=True)

            # Select action according to policy
            n_actions = self.actor_t(n_states)[0]

            # V target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_V1, target_V2 = self.value_t(n_states)
                target_V = torch.min(target_V1, target_V2)
                target_V = target_V * self.discount ** (steps + 1)
                target_V = rewards + (1 - stops) * target_V

            # Get current Q estimates
            current_V1, current_V2 = self.value(states)

            # Compute critic loss
            value_loss = nn.MSELoss()(current_V1, target_V) + \
                nn.MSELoss()(current_V2, target_V)

            # Optimize the critic // M Step
            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_Q1, target_Q2 = self.critic_t(n_states, n_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = target_Q * self.discount ** (steps + 1)
                target_Q = rewards + (1 - stops) * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
                nn.MSELoss()(current_Q2, target_Q)

            # Optimize the critic // M Step
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # actions, mus, sigmas
                actions, mus, sigmas = self.actor(states)

                # Compute actor loss with entropy
                actor_loss = -(self.critic(states, actions)[0] - self.value(states)[0])
                actor_loss -= torch.log(sigmas ** 2).mean() * critic_loss.detach()
                actor_loss = actor_loss.mean()

                # Optimize the actor // E Steps
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                # Update the frozen actor models
                for param, target_param in zip(self.actor.parameters(), self.actor_t.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

            # Update the frozen critic models
            for param, target_param in zip(self.critic.parameters(), self.critic_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            # Update the frozen value models
            for param, target_param in zip(self.value.parameters(), self.value_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, directory):
        """
        Save the model in given folder
        """
        self.actor.save_model(directory, "actor")
        self.critic.save_model(directory, "critic")

    def load(self, directory):
        """
        Load model from folder
        """
        self.actor.load_model(directory, "actor")
        self.critic.load_model(directory, "critic")
