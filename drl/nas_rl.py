import numpy as np
import torch
import torch.nn as nn

from drl.models import MetaActor, MetaActorv2, MetaActorv3, CriticTD3 as Critic, MultiCriticTD3 as MultiCritic
from tqdm import tqdm
from copy import deepcopy

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class NASRL(object):
    """
    Neural Architecture Search for Reinforcement Learning
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Actor stuff
        self.actor = MetaActorv2(state_dim, action_dim, max_action, args)
        self.actor_t = MetaActorv2(state_dim, action_dim, max_action, args)
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

        # Hyper parameters
        self.tau = args.tau
        self.ops = [torch.nn.modules.Tanh(), torch.nn.modules.ReLU(
        ), torch.nn.modules.ELU(), torch.nn.modules.LeakyReLU(), None]
        self.n_ops = len(self.ops)
        self.n_steps = args.n_steps
        self.n_cells = args.n_cells
        self.actor_lr = args.actor_lr
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.noise_clip = args.noise_clip
        self.hidden_size = args.hidden_size
        self.policy_freq = args.policy_freq
        self.policy_noise = args.policy_noise
        self.reward_scale = args.reward_scale
        self.weights = FloatTensor(
            [self.discount ** i for i in range(self.n_steps)])

        # Sampling stuff
        self.alphas = 1 / self.n_ops * \
            np.ones((self.n_cells - 1, self.n_cells, self.n_ops))

        # Cuda
        if args.use_cuda:
            self.critic.cuda()
            self.critic_t.cuda()
            self.actor.cuda()
            self.actor_t.cuda()

    def action(self, state, ops_mat):
        """
        Returns action given state
        """
        if ops_mat is None:
            ops_mat = self.sample_actor()
        state = FloatTensor(state.reshape(1, -1))
        return self.actor(state, ops_mat).cpu().data.numpy().flatten()

    def train(self, memory, n_iter, ops_mat):
        """
        Trains the model for n_iter steps with the current sampled architecture
        """
        critic_losses = []
        actor_losses = []

        # Update Q to get Q-function corresponding to sampled architecture
        for _ in tqdm(range(n_iter)):

            # Sample from replay buffer
            states, actions, n_states, rewards, steps, dones, stops = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards * self.weights
            rewards = rewards.sum(dim=1, keepdim=True)

            # Select action according to policy and add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            n_actions = self.actor_t(n_states, ops_mat) + FloatTensor(noise)
            n_actions = n_actions.clamp(-self.max_action, self.max_action)

           # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_q1, target_q2 = self.critic_t(n_states, n_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1 - stops) * \
                    target_q * self.discount ** (steps + 1)

            # Get current Q estimates
            current_q1, current_q2 = self.critic(states, actions)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_q1, target_q) + \
                nn.MSELoss()(current_q2, target_q)
            critic_losses.append(critic_loss.data.cpu().numpy())

            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # Update the frozen critic model
            for param, target_param in zip(self.critic.parameters(), self.critic_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        for _ in tqdm(range(n_iter)):

            # Sample from replay buffer
            states, _, _, _, _, _, _ = memory.sample(self.batch_size)

            # Compute actor loss
            actor_loss = -self.critic(states, self.actor(states, ops_mat))[0].mean()
            actor_losses.append(actor_loss.data.cpu().numpy())

            # Optimize the actor
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Update the frozen actor model
            for param, target_param in zip(self.actor.parameters(), self.actor_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        return np.mean(critic_losses), np.mean(actor_losses)

    def sample_actor(self):
        """
        Sample an actor according to the parameters
        of the distribution.
        """
        ops_mat = self.actor.sample_ops()
        return ops_mat

    def sample_pop(self, pop_size):
        """
        Samples pop_size actors
        """
        ops_mats = []
        for _ in range(pop_size):
            ops_mats.append(self.sample_actor())
        return ops_mats

    def update_dist(self, fitness, ops_mats):
        """
        Updates the distribution of the NAS with the last score
        """
        self.actor.update_dist(fitness, ops_mats)

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


class NASRLv2(object):
    """
    Neural Architecture Search for Reinforcement Learning added the multi-critic
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Actor stuff
        self.actor = MetaActorv2(state_dim, action_dim, max_action, args)
        self.actor_t = MetaActorv2(state_dim, action_dim, max_action, args)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr)

        # Critic stuff
        self.critic = MultiCritic(state_dim, action_dim, max_action, args)
        self.critic_t = MultiCritic(state_dim, action_dim, max_action, args)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr)

        # Env stuff
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Hyper parameters
        self.tau = args.tau
        self.ops = [torch.nn.modules.Tanh(), torch.nn.modules.ReLU(
        ), torch.nn.modules.ELU(), torch.nn.modules.LeakyReLU(), None]
        self.n_ops = len(self.ops)
        self.n_steps = args.n_steps
        self.n_cells = args.n_cells
        self.actor_lr = args.actor_lr
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.noise_clip = args.noise_clip
        self.hidden_size = args.hidden_size
        self.policy_freq = args.policy_freq
        self.policy_noise = args.policy_noise
        self.reward_scale = args.reward_scale
        self.weights = FloatTensor(
            [self.discount ** i for i in range(self.n_steps)])

        # Sampling stuff
        self.alphas = 1 / self.n_ops * \
            np.ones((self.n_cells - 1, self.n_cells, self.n_ops))

        # Cuda
        if args.use_cuda:
            self.critic.cuda()
            self.critic_t.cuda()
            self.actor.cuda()
            self.actor_t.cuda()

    def action(self, state, ops_mat):
        """
        Returns action given state
        """
        if ops_mat is None:
            ops_mat = self.sample_actor()
        state = FloatTensor(state.reshape(1, -1))
        return self.actor(state, ops_mat).cpu().data.numpy().flatten()

    def train(self, memory, n_iter, ops_mat):
        """
        Trains the model for n_iter steps with the current sampled architecture
        """
        critic_losses = []
        actor_losses = []

        # Getting operations and corresponding key
        key = list(ops_mat[np.triu_indices(self.n_cells, k=1)])
        key = ["{}".format(i) for i in key]
        key = "_".join(key)

        # Update Q to get Q-function corresponding to sampled architecture
        for _ in tqdm(range(n_iter)):

            # Sample from replay buffer
            states, actions, n_states, rewards, steps, dones, stops = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards * self.weights
            rewards = rewards.sum(dim=1, keepdim=True)

            # Select action according to policy and add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            n_actions = self.actor_t(n_states, ops_mat) + FloatTensor(noise)
            n_actions = n_actions.clamp(-self.max_action, self.max_action)

           # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_q1, target_q2 = self.critic_t(n_states, n_actions, key)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1 - stops) * \
                    target_q * self.discount ** (steps + 1)

            # Get current Q estimates
            current_q1, current_q2 = self.critic(states, actions, key)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_q1, target_q) + \
                nn.MSELoss()(current_q2, target_q)
            critic_losses.append(critic_loss.data.cpu().numpy())

            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # Update the frozen critic model
            for param, target_param in zip(self.critic.parameters(), self.critic_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update actor
        for _ in tqdm(range(n_iter)):

            # Sample from replay buffer
            states, _, _, _, _, _, _ = memory.sample(self.batch_size)

            # Compute actor loss
            actor_loss = -self.critic(states, self.actor(states, ops_mat), key)[0].mean()
            actor_losses.append(actor_loss.data.cpu().numpy())

            # Optimize the actor
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Update the frozen actor model
            for param, target_param in zip(self.actor.parameters(), self.actor_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        return np.mean(critic_losses), np.mean(actor_losses)

    def sample_actor(self):
        """
        Sample an actor according to the parameters
        of the distribution.
        """
        ops_mat = self.actor.sample_ops()
        return ops_mat

    def sample_pop(self, pop_size):
        """
        Samples pop_size actors
        """
        ops_mats = []
        for _ in range(pop_size):
            ops_mats.append(self.sample_actor())
        return ops_mats

    def update_dist(self, fitness, ops_mats):
        """
        Updates the distribution of the NAS with the last score
        """
        self.actor.update_dist(fitness, ops_mats)

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


class NASRLv3(object):
    """
    Neural Architecture Search for Reinforcement Learning changed sampling strategy
    to be scale invariant but no multi-critic
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Actor stuff
        self.actor = MetaActorv3(state_dim, action_dim, max_action, args)
        self.actor_t = MetaActorv3(state_dim, action_dim, max_action, args)
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

        # Hyper parameters
        self.tau = args.tau
        self.ops = [torch.nn.modules.Tanh(), torch.nn.modules.ReLU(
        ), torch.nn.modules.ELU(), torch.nn.modules.LeakyReLU(), None]
        self.n_ops = len(self.ops)
        self.n_steps = args.n_steps
        self.n_cells = args.n_cells
        self.actor_lr = args.actor_lr
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.noise_clip = args.noise_clip
        self.hidden_size = args.hidden_size
        self.policy_freq = args.policy_freq
        self.policy_noise = args.policy_noise
        self.reward_scale = args.reward_scale
        self.weights = FloatTensor(
            [self.discount ** i for i in range(self.n_steps)])

        # Sampling stuff
        self.alphas = 1 / self.n_ops * \
            np.ones((self.n_cells - 1, self.n_cells, self.n_ops))

        # Cuda
        if args.use_cuda:
            self.critic.cuda()
            self.critic_t.cuda()
            self.actor.cuda()
            self.actor_t.cuda()

    def action(self, state, ops_mat):
        """
        Returns action given state
        """
        if ops_mat is None:
            ops_mat = self.sample_actor()
        state = FloatTensor(state.reshape(1, -1))
        return self.actor(state, ops_mat).cpu().data.numpy().flatten()

    def train(self, memory, n_iter, ops_mat):
        """
        Trains the model for n_iter steps with the current sampled architecture
        """
        critic_losses = []
        actor_losses = []

        # Update Q to get Q-function corresponding to sampled architecture
        for _ in tqdm(range(n_iter)):

            # Sample from replay buffer
            states, actions, n_states, rewards, steps, dones, stops = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards * self.weights
            rewards = rewards.sum(dim=1, keepdim=True)

            # Select action according to policy and add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            n_actions = self.actor_t(n_states, ops_mat) + FloatTensor(noise)
            n_actions = n_actions.clamp(-self.max_action, self.max_action)

           # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_q1, target_q2 = self.critic_t(n_states, n_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1 - stops) * \
                    target_q * self.discount ** (steps + 1)

            # Get current Q estimates
            current_q1, current_q2 = self.critic(states, actions)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_q1, target_q) + \
                nn.MSELoss()(current_q2, target_q)
            critic_losses.append(critic_loss.data.cpu().numpy())

            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # Update the frozen critic model
            for param, target_param in zip(self.critic.parameters(), self.critic_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        # update actor
        for _ in tqdm(range(n_iter)):

            # Sample from replay buffer
            states, _, _, _, _, _, _ = memory.sample(self.batch_size)

            # Compute actor loss
            actor_loss = -self.critic(states, self.actor(states, ops_mat))[0].mean()
            actor_losses.append(actor_loss.data.cpu().numpy())

            # Optimize the actor
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Update the frozen actor model
            for param, target_param in zip(self.actor.parameters(), self.actor_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        return np.mean(critic_losses), np.mean(actor_losses)

    def sample_actor(self):
        """
        Sample an actor according to the parameters
        of the distribution.
        """
        ops_mat = self.actor.sample_ops()
        return ops_mat

    def sample_pop(self, pop_size):
        """
        Samples pop_size actors
        """
        ops_mats = []
        for _ in range(pop_size):
            ops_mats.append(self.sample_actor())
        return ops_mats

    def update_dist(self, fitness, ops_mats):
        """
        Updates the distribution of the NAS with the last score
        """
        self.actor.update_dist(fitness, ops_mats)

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
