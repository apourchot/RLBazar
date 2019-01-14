import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from drl.models import Actor, NASActor, CriticTD3 as Critic


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class TD3(object):
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
        self.weights = FloatTensor(
            [self.discount ** i for i in range(self.n_steps)])

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
        state = FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, memory, n_iter):
        """
        Trains the model for n_iter steps
        """
        critic_losses = []
        actor_losses = []
        for it in range(n_iter):

            # Sample replay buffer
            states, actions, n_states, rewards, steps, dones, stops = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards * self.weights
            rewards = rewards.sum(dim=1, keepdim=True)

            # Select action according to policy and add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            n_actions = self.actor_t(n_states) + FloatTensor(noise)
            n_actions = n_actions.clamp(-self.max_action, self.max_action)

           # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_q1, target_q2 = self.critic_t(n_states, n_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1 - stops) * target_q * self.discount ** (steps + 1)

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

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic(states, self.actor(states))[0].mean()
                actor_losses.append(actor_loss.data.cpu().numpy())

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

        return np.mean(critic_losses), np.mean(actor_losses)

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


class NASTD3(object):
    """
    Twin Delayed Deep Deterministic Policy Gradient Algorithm with n-step return
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Actor stuff
        self.actor = NASActor(state_dim, action_dim, max_action, args)
        self.actor_t = NASActor(state_dim, action_dim, max_action, args)
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
        self.weights = FloatTensor(
            [self.discount ** i for i in range(self.n_steps)])

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
        state = FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, memory, n_iter):
        """
        Trains the model for n_iter steps
        """
        critic_losses = []
        actor_losses = []
        for it in tqdm(range(n_iter)):

            # Sample replay buffer
            states, actions, n_states, rewards, steps, dones, stops = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards * self.weights
            rewards = rewards.sum(dim=1, keepdim=True)

            # Select action according to policy
            n_actions = self.actor_t(n_states)

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_q1, target_q2 = self.critic_t(n_states, n_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1 - stops) * target_q * self.discount ** (steps + 1)

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

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic(states, self.actor(states))[0].mean()
                actor_losses.append(actor_loss.data.cpu().numpy())

                # Optimize the actor
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                # Normalize alphas
                self.actor.normalize_alpha()
                self.actor_t.normalize_alpha()

                # Update the frozen actor models
                for param, target_param in zip(self.actor.parameters(), self.actor_t.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

            # Update the frozen critic models
            for param, target_param in zip(self.critic.parameters(), self.critic_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        return np.mean(critic_losses), np.mean(actor_losses)

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


class STD3(object):
    """
    Smoothed Twin Delayed Deep Deterministic Policy Gradient Algorithm
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
        self.weights = FloatTensor(
            [self.discount ** i for i in range(self.n_steps)])

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
        state = FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, memory, n_iter):
        """
        Trains the model for n_iter steps
        """

        for it in range(n_iter):

            # Sample replay buffer
            states, actions, n_states, rewards, steps, dones, stops = memory.sample(
                self.batch_size)
            print("before:", rewards)
            rewards = self.reward_scale * rewards * self.weights
            rewards = rewards.sum(dim=1, keepdim=True)
            print("after:", rewards)

            # Select action according to policy and add clipped noise
            noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
            n_actions = self.actor_t(n_states)  # + FloatTensor(noise)
            n_actions = n_actions.clamp(-self.max_action, self.max_action)

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_Q1, target_Q2 = self.critic_t(n_states, n_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = target_Q * self.discount ** (steps + 1)
                target_Q = rewards.sum + (1 - stops) * target_Q

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
                # noise = np.clip(np.random.normal(0, self.policy_noise, size=(
                #     self.batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
                # n_actions = self.actor(states) + FloatTensor(noise)
                # n_actions = n_actions.clamp(-self.max_action, self.max_action)
                # actor_loss = -self.critic(states, n_actions)[0].mean()

                actor_params = self.actor.get_params()
                grads = np.zeros(self.actor.get_size())

                for _ in range(5):

                    noise = np.random.normal(
                        0, self.policy_noise, size=(self.actor.get_size()))
                    self.actor.set_params(
                        actor_params + noise * self.policy_noise)

                    n_actions = self.actor(states)  # + FloatTensor(noise)
                    n_actions = n_actions.clamp(-self.max_action,
                                                self.max_action)

                    self.actor_opt.zero_grad()
                    actor_loss = -self.critic(states, n_actions)[0].mean()
                    actor_loss.backward()

                    # * np.exp(- noise ** 2 / (2 * self.policy_noise ** 2)
                    grads += self.actor.get_grads()
                    #        ) / np.sqrt(2 * np.pi) / self.policy_noise

                self.actor_opt.zero_grad()
                self.actor.set_params(actor_params)
                self.actor.set_grads(grads / 5)
                self.actor_opt.step()

                # Optimize the actor
                # self.actor_opt.zero_grad()
                # actor_loss.backward()
                # self.actor_opt.step()

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
    Population-based Twin Delayed Deep Deterministic Policy Gradient Algorithm
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Mu stuff
        self.mu = Actor(state_dim, action_dim, max_action, args)
        self.mu_t = Actor(state_dim, action_dim, max_action, args)
        self.mu_t.load_state_dict(self.mu.state_dict())

        # Sigma stuff
        self.sigma = torch.nn.Parameter(
            args.sigma_init * torch.ones(self.mu.get_size()))
        self.sigma_t = torch.nn.Parameter(
            args.sigma_init * torch.ones(self.mu.get_size()))

        # Optimizer
        self.opt = torch.optim.Adam(self.mu.parameters(), lr=args.actor_lr)
        self.opt.add_param_group({"params": self.sigma})

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
        self.pop_size = args.pop_size
        self.batch_size = args.batch_size
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.policy_noise = args.policy_noise
        self.reward_scale = args.reward_scale
        self.n_actor_params = self.mu.get_size()
        self.weights = FloatTensor(
            [self.discount ** i for i in range(self.n_steps)])

        # cuda
        if USE_CUDA:
            self.mu.cuda()
            self.mu_t.cuda()
            self.sigma.cuda()
            self.sigma_t.cuda()
            self.critic.cuda()
            self.critic_t.cuda()

    def train(self, memory, n_iter):
        """
        Trains the model for n_iter steps
        """

        for it in range(n_iter):

            # Sample replay buffer
            states, actions, n_states, rewards, steps, dones, stops = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards * self.weights
            rewards = rewards.sum(dim=1, keepdim=True)

            # Select action according to policy
            n_actions = self.mu_t(n_states)

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

            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Compute actor loss
                mu = self.mu.get_params()
                sigma = self.sigma.data.cpu().numpy()
                noise = np.random.normal(size=(self.n_actor_params))
                pi = mu + noise * sigma ** 2

                self.opt.zero_grad()
                self.mu.set_params(pi)
                pi_loss = -self.critic(states, self.mu(states))[0].mean()
                pi_loss.backward()

                # this is bs but necessary
                if self.sigma.grad is None:
                    sigma_loss = self.sigma.sum()
                    sigma_loss.backward()

                grad_pi = self.mu.get_grads()
                grad_mu = grad_pi
                grad_sigma = 2 * grad_pi * sigma * noise

                self.opt.zero_grad()
                self.mu.set_params(mu)
                self.mu.set_grads(grad_mu)
                self.sigma.grad.data = FloatTensor(grad_sigma)
                self.opt.step()

                # Update the frozen mu
                for param, target_param in zip(self.mu.parameters(), self.mu_t.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                # Update the frozen sigma
                self.sigma_t = self.tau * self.sigma + \
                    (1 - self.tau) * self.sigma_t

            # Update the frozen critic models
            for param, target_param in zip(self.critic.parameters(), self.critic_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, directory):
        """
        Save the model in given folder
        """
        self.mu.save_model(directory, "actor")
        self.critic.save_model(directory, "critic")

    def load(self, directory):
        """
        Load model from folder
        """
        self.mu.load_model(directory, "actor")
        self.critic.load_model(directory, "critic")


class D2TD3(object):
    """
    Double-Smoothed Twin Delayed Deep Deterministic Policy Gradient Algorithm
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Mu stuff
        self.mu = Actor(state_dim, action_dim, max_action, args)
        self.mu_t = Actor(state_dim, action_dim, max_action, args)
        self.mu_t.load_state_dict(self.mu.state_dict())

        # Sigma stuff
        self.log_sigma = FloatTensor(
            np.log(args.sigma_init) * np.ones(self.mu.get_size()))
        self.log_sigma_t = FloatTensor(
            np.log(args.sigma_init) * np.ones(self.mu.get_size()))

        # Optimizer
        self.opt = torch.optim.Adam(self.mu.parameters(), lr=args.actor_lr)
        self.opt.add_param_group({"params": self.log_sigma})

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
        self.pop_size = args.pop_size
        self.batch_size = args.batch_size
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq
        self.policy_noise = args.policy_noise
        self.reward_scale = args.reward_scale
        self.n_actor_params = self.mu.get_size()
        self.weights = FloatTensor(
            [self.discount ** i for i in range(self.n_steps)])

        # cuda
        if USE_CUDA:
            self.mu.cuda()
            self.mu_t.cuda()
            self.log_sigma.cuda()
            self.log_sigma_t.cuda()
            self.critic.cuda()
            self.critic_t.cuda()

    def train(self, memory, n_iter):
        """
        Trains the model for n_iter steps
        """

        for it in range(n_iter):

            # Sample replay buffer
            states, actions, n_states, rewards, steps, dones, stops = memory.sample(
                self.batch_size)
            rewards = self.reward_scale * rewards * self.weights
            rewards = rewards.sum(dim=1, keepdim=True)

            # Select policy according to noise
            # mu_t = self.mu_t.get_params()
            # log_sigma_t = self.log_sigma_t.data.cpu().numpy()
            # noise = np.random.randn(self.n_actor_params)
            # pi_t = mu_t + noise * np.exp(log_sigma_t)

            # self.mu_t.set_params(pi_t)
            n_actions = self.mu_t(n_states)
            # self.mu.set_params(mu_t)

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

            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # Delayed policy updates
            if it % self.policy_freq == 0:

                # Creating random policy
                mu = self.mu.get_params()
                log_sigma = self.log_sigma.data.cpu().numpy()
                noise = np.random.randn(self.n_actor_params)
                pi = mu + noise * np.exp(log_sigma)

                # Computing loss
                self.mu.set_params(pi)
                pi_loss = -self.critic(states, self.mu(states))[0].mean()

                # Computing gradient wrt noisy policy
                pi_loss.backward()
                pi_grad = self.mu.get_grads()
                self.mu.set_params(mu)

                # Setting gradients
                self.opt.zero_grad()
                self.mu.set_params(mu)
                self.mu.set_grads(pi_grad)
                self.log_sigma.grad = FloatTensor(
                    pi_grad * noise * np.exp(log_sigma))
                self.opt.step()

                # Update the frozen mu
                for param, target_param in zip(self.mu.parameters(), self.mu_t.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                # Update the frozen sigma
                self.log_sigma_t = self.tau * self.log_sigma + \
                    (1 - self.tau) * self.log_sigma_t

            # Update the frozen critic models
            for param, target_param in zip(self.critic.parameters(), self.critic_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, directory):
        """
        Save the model in given folder
        """
        self.mu.save_model(directory, "actor")
        self.critic.save_model(directory, "critic")

    def load(self, directory):
        """
        Load model from folder
        """
        self.mu.load_model(directory, "actor")
        self.critic.load_model(directory, "critic")
