import torch
import torch.nn as nn

from drl.models import GaussianActor, CriticTD3 as Critic

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class MPO(object):
    """
    MPO-inspired Actor-Critic Algorithm: https://arxiv.org/pdf/1806.06920.pdf
    """

    def __init__(self, state_dim, action_dim, max_action, args):

        # Actor stuff
        self.pi = GaussianActor(state_dim, action_dim, max_action, args)
        self.pi_t = GaussianActor(state_dim, action_dim, max_action, args)
        self.pi_t.load_state_dict(self.pi.state_dict())
        self.pi_opt = torch.optim.Adam(
            self.pi.parameters(), lr=args.actor_lr)

        # Variational policy stuff
        self.q = GaussianActor(state_dim, action_dim, max_action, args)
        self.q_t = GaussianActor(state_dim, action_dim, max_action, args)
        self.q.load_state_dict(self.pi.state_dict())
        self.q_t.load_state_dict(self.pi.state_dict())
        self.q_opt = torch.optim.Adam(
            self.q.parameters(), lr=args.actor_lr)

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
        self.alpha = args.alpha
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
            self.pi.cuda()
            self.pi_t.cuda()
            self.q.cuda()
            self.q_t.cuda()
            self.critic.cuda()
            self.critic_t.cuda()

    def action(self, state):
        """
        Returns action given state
        """
        state = FloatTensor(state.reshape(1, -1))
        action, mu, sigma = self.pi(state)
        # print("mu", mu)
        # print("sigma", sigma)
        return action.cpu().data.numpy().flatten()

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

            # Select action according to policy pi
            n_actions = self.pi(n_states)[0]

            # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
            with torch.no_grad():
                target_q1, target_q2 = self.critic_t(n_states, n_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = target_q * self.discount ** (steps + 1)
                target_q = rewards + (1 - stops) * target_q

            # Get current Q estimates
            current_q1, current_q2 = self.critic(states, actions)

            # Compute critic loss
            critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

            # Optimize the critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # E-Step

            # actions, mus, sigmas
            pi_a, pi_mus, pi_sigmas = self.pi(states)
            q_a, q_mus, q_sigmas = self.q(states)

            # KL div between pi and q
            kl_div = torch.log(pi_sigmas ** 2 / q_sigmas ** 2)
            kl_div += (q_sigmas ** 4 + (q_mus - pi_mus) ** 2) / (2 * pi_sigmas ** 4)
            kl_div = kl_div.mean(dim=1, keepdim=True)

            # q loss
            loss = self.critic(states, q_a)[0] - kl_div
            loss = -loss.mean()

            # SGD
            self.q_opt.zero_grad()
            self.pi_opt.zero_grad()
            loss.backward()
            self.q_opt.step()
            self.pi_opt.step()

            # M-Step

            # actions, mus, sigmas
            # pi_a, pi_mus, pi_sigmas = self.pi(states)
            # q_a, q_mus, q_sigmas = self.q(states)
            # q_a.detach(), q_mus.detach(), q_sigmas.detach()
#
            # # KL div between pi and q
            # kl_div = torch.log(pi_sigmas ** 2 / q_sigmas ** 2)
            # kl_div += (q_sigmas ** 4 + (q_mus - pi_mus) ** 2) / \
            #     (2 * pi_sigmas ** 4)
            # kl_div = kl_div.mean(dim=1, keepdim=True)
#
            # # pi_loss
            # pi_loss = kl_div.mean()
#
            # # SGD
            # self.pi_opt.zero_grad()
            # pi_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1)
            # self.pi_opt.step()

            # print(pi_loss.data, loss.data)

            # Update the frozen actor models
            for param, target_param in zip(self.pi.parameters(), self.pi_t.parameters()):
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
        self.pi.save_model(directory, "actor")
        self.critic.save_model(directory, "critic")

    def load(self, directory):
        """
        Load model from folder
        """
        self.pi.load_model(directory, "actor")
        self.critic.load_model(directory, "critic")
