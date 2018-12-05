from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import gym
import gym.spaces
import numpy as np
from tqdm import tqdm

from drl.random_process import GaussianNoise
from utils.utils import evaluate, get_output_folder, prLightPurple, prRed
from utils.memory import Memory
from utils.logger import Logger
from utils.args import parser

from drl.TD3 import TD3, NTD3, STD3, POPTD3, D2TD3
from drl.Virel import Virel

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

if __name__ == "__main__":

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    args.use_cuda = USE_CUDA
    with open(args.output + "/parameters.txt", 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))

    # environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # Memory
    memory = Memory(args.mem_size, state_dim, action_dim, n_steps=args.n_steps)

    # Algorithm
    drla = Virel(state_dim, action_dim, max_action, args)

    # Action noise
    a_noise = None # GaussianNoise(action_dim, sigma=args.gauss_sigma)

    # Logger
    fields = ["eval_score", "total_steps"]
    logger = Logger(args.output, fields)

    # Train
    ite = 0
    K = 5000
    total_steps = 0
    while total_steps < args.max_steps:

        ite += 1
        actor_steps = 0
        while actor_steps < K:

            fitness, steps = evaluate(
                drla, env, memory, noise=a_noise, random=total_steps <= args.start_steps, n_steps=args.n_steps)
            drla.train(memory, steps)

            actor_steps += steps
            total_steps += steps

            prLightPurple(
                "Iteration {}; Noisy Actor fitness:{}".format(ite, fitness))

        fitness, steps = evaluate(
            drla, env, memory=None, noise=None, n_episodes=10)
        logger.append([fitness, total_steps])
        print("---------------------------------")
        prRed("Total steps: {}; Actor fitness:{} \n".format(
            total_steps, fitness))
        drla.save(args.output)
