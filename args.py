import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str,)
parser.add_argument('--env', default='HalfCheetah-v2', type=str)
parser.add_argument('--start_steps', default=10000, type=int)

# DDPG parameters
parser.add_argument('--actor_lr', default=0.001, type=float)
parser.add_argument('--critic_lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--discount', default=0.99, type=float)
parser.add_argument('--reward_scale', default=1., type=float)
parser.add_argument('--tau', default=0.005, type=float)
parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')

# TD3 parameters
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_freq', default=2, type=int)
parser.add_argument('--n_steps', default=1, type=int)

# Gaussian noise parameters
parser.add_argument('--gauss_sigma', default=0.1, type=float)

# OU process parameters
parser.add_argument('--ou_noise', dest='ou_noise', action='store_true')
parser.add_argument('--ou_theta', default=0.15, type=float)
parser.add_argument('--ou_sigma', default=0.2, type=float)
parser.add_argument('--ou_mu', default=0.0, type=float)

# Training parameters
parser.add_argument('--n_episodes', default=1, type=int)
parser.add_argument('--max_steps', default=1000000, type=int)
parser.add_argument('--mem_size', default=1000000, type=int)
parser.add_argument('--n_noisy', default=0, type=int)

# Testing parameters
parser.add_argument('--filename', default="", type=str)
parser.add_argument('--n_test', default=1, type=int)

# misc
parser.add_argument('--output', default='results/', type=str)
parser.add_argument('--period', default=5000, type=int)
parser.add_argument('--n_eval', default=10, type=int)
parser.add_argument('--save_all_models',
                    dest="save_all_models", action="store_true")
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--seed', default=-1, type=int)
parser.add_argument('--render', dest='render', action='store_true')