import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

from models import Actor, CriticTD3 as Critic, ValueTD3 as Value


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class TD3(object):
    """
    Twin Delayed Deep Deterministic Policy Gradient Algorithm
    https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Actor stuff
        self.actor = Actor(state_dim, action_dim, max_action, args)
        self.actor_t = Actor(state_dim, action_dim, max_action, args)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)

        # Critic stuff
        self.critic = Critic(state_dim, action_dim, max_action, args)
        self.critic_t = Critic(state_dim, action_dim, max_action, args)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # Env stuff
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Hyperparams
        self.tau = args.tau
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.policy_noise = args.policy_noise
        self.reward_scale = args.reward_scale

        # cuda
        if args.use_cuda:
            self.actor.cuda()
            self.actor_t.cuda()
            self.critic.cuda()
            self.critic_t.cuda()

    def action(self, state):
        """
        Returns action given state
        """
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, memory, n_iter):
        """
        Trains the model for n_iter steps
        """

        for it in range(n_iter):

            # Sample replay buffer
            states, n_states, actions, rewards, dones = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards

            # Select action according to policy and add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            n_actions = self.actor_t(n_states) + FloatTensor(noise)
            n_actions = n_actions.clamp(-self.max_action, self.max_action)

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_Q1, target_Q2 = self.critic_t(n_states, n_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards + (1 - dones) * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
                nn.MSELoss()(current_Q2, target_Q)

            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic(states, self.actor(states))[0].mean()

                # Optimize the actor
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


class NTD3(object):
    """
    Twin Delayed Deep Deterministic Policy Gradient Algorithm with n-step return
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Actor stuff
        self.actor = Actor(state_dim, action_dim, max_action, args)
        self.actor_t = Actor(state_dim, action_dim, max_action, args)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)

        # Critic stuff
        self.critic = Critic(state_dim, action_dim, max_action, args)
        self.critic_t = Critic(state_dim, action_dim, max_action, args)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

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
        self.weights = FloatTensor([self.discount ** i for i in range(self.n_steps)])

        # cuda
        if args.use_cuda:
            self.actor.cuda()
            self.actor_t.cuda()
            self.critic.cuda()
            self.critic_t.cuda()

    def action(self, state):
        """
        Returns action given state
        """
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, memory, n_iter):
        """
        Trains the model for n_iter steps
        """

        for it in range(n_iter):

            # Sample replay buffer
            states, n_states, actions, rewards, steps, dones, stops = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards * self.weights

            # Select action according to policy and add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            n_actions = self.actor_t(n_states) + FloatTensor(noise)
            n_actions = n_actions.clamp(-self.max_action, self.max_action)

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_Q1, target_Q2 = self.critic_t(n_states, n_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards.sum(dim=1, keepdim=True) + (1 - stops) * target_Q * self.discount ** (steps + 1)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
                nn.MSELoss()(current_Q2, target_Q)

            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic(states, self.actor(states))[0].mean()

                # Optimize the actor
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


class TD3exp(object):
    """
    Twin Delayed Deep Deterministic Policy Gradient Algorithm
    https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Actor stuff
        self.actor = Actor(state_dim, action_dim, max_action, args)
        self.actor_t = Actor(state_dim, action_dim, max_action, args)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)

        # Critic stuff
        self.critic = Critic(state_dim, action_dim, max_action, args)
        self.critic_t = Critic(state_dim, action_dim, max_action, args)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # Env stuff
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Hyperparams
        self.tau = args.tau
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.policy_noise = args.policy_noise
        self.reward_scale = args.reward_scale

        # cuda
        if args.use_cuda:
            self.actor.cuda()
            self.actor_t.cuda()
            self.critic.cuda()
            self.critic_t.cuda()

    def action(self, state):
        """
        Returns action given state
        """
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, memory, n_iter):
        """
        Trains the model for n_iter steps
        """

        for it in range(n_iter):

            # Sample replay buffer
            states, n_states, actions, rewards, dones = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards

            # Select action according to policy and add clipped noise
            mean = self.actor_t(n_states)
            noise = FloatTensor(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)))
            n_actions = torch.tanh(mean + noise) * self.max_action * 2 / np.pi

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_Q1, target_Q2 = self.critic_t(n_states, mean)
                entropy = - 0.5 * torch.sum((noise / self.policy_noise) ** 2, dim=1, keepdim=True) \
                    - torch.sum(torch.log(self.max_action -
                                          n_actions ** 2), dim=1, keepdim=True)
                # print(entropy)
                target_Q = torch.min(target_Q1, target_Q2) + 100
                target_Q = rewards + (1 - dones) * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
                nn.MSELoss()(current_Q2, target_Q)

            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic(states, self.actor(states))[0].mean()

                # Optimize the actor
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


class POPTD3(object):
    """
    Twin Delayed Deep Deterministic Policy Gradient Algorithm
    https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Mean Actor and sigma stuff
        self.actor = Actor(state_dim, action_dim, max_action, args)
        self.actor_t = Actor(state_dim, action_dim, max_action, args)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.sigma = 1e-3 * \
            torch.nn.Parameter(torch.ones(self.actor.get_size()))
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)
        # self.actor_opt.add_param_group({"sigma": self.sigma})

        # Value stuff
        self.value = Value(state_dim, action_dim, max_action, args)
        self.value_t = Value(state_dim, action_dim, max_action, args)
        self.value_t.load_state_dict(self.value.state_dict())
        self.value_opt = torch.optim.Adam(
            self.value.parameters(), lr=args.critic_lr)

        # Critic stuff
        self.critic = Critic(state_dim, action_dim, max_action, args)
        self.critic_t = Critic(state_dim, action_dim, max_action, args)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # Env stuff
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Hyperparams
        self.tau = args.tau
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.policy_noise = args.policy_noise
        self.reward_scale = args.reward_scale

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
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def sample(self):
        """
        Returns the actors to be evaluated
        """
        self.epsilon = np.random.randn(1, self.actor.get_size())
        return self.actor.get_params() + self.epsilon * to_numpy(self.sigma.data)

    def train(self, memory, n_iter):
        """
        Trains the model for n_iter steps
        """

        for it in range(n_iter):

            # Sample replay buffer off-policy
            states, n_states, actions, rewards, dones = memory.sample(
                self.batch_size, debug=False)
            rewards = self.reward_scale * rewards

            # Select action according to policy and add clipped noise
            # noise = np.clip(np.random.normal(0, self.policy_noise, size=(
            #     self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            # n_actions = self.actor_t(n_states) + FloatTensor(noise)
            # n_actions = n_actions.clamp(-self.max_action, self.max_action)

            n_actions = self.actor_t(n_states)

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_Q1, target_Q2 = self.critic_t(n_states, n_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards + (1 - dones) * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
                nn.MSELoss()(current_Q2, target_Q)

            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # V target = reward + discount * min_i(Vi(next state))
            with torch.no_grad():
                target_V1, target_V2 = self.value_t(n_states)
                target_V = torch.min(target_V1, target_V2)
                target_V = rewards + (1 - dones) * self.discount * target_V

            # Get current Q estimates
            current_V1, current_V2 = self.value(states)

            # Compute critic loss
            value_loss = nn.MSELoss()(current_V1, target_V) + \
                nn.MSELoss()(current_V2, target_V)

            # Optimize the value
            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Sample replay buffer on-policy
                states, n_states, actions, rewards, dones = memory.sample(
                    self.batch_size, debug=False)

                # Compute actor loss
                actor_loss = self.value(states)[0].mean(
                ) - self.critic(states, self.actor(states))[0].mean()
                # actor_loss -= self.critic(states, self.actor(states) + self.sigma * self.epsilon)

                # Optimize the actor
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
