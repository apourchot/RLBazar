import torch
import gym.spaces
import numpy as np

from utils.utils import evaluate_nasrl as evaluate, get_output_folder, prLightPurple, prRed
from utils.memory import Memory
from utils.logger import Logger
from utils.args import parser
from utils.random_process import GaussianNoise

from drl.nas_rl import NASRL, NASRLv2, NASRLv3, NASReptiLe, NASReptiLev2

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

    # Environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # Memory
    memory = Memory(args.mem_size, state_dim, action_dim, args)

    # Algorithm
    drla = NASRL(state_dim, action_dim, max_action, args)

    # Action noise
    a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)

    # Logger
    fields = ["eval_score", "total_steps", "train_scores", "sampled_archi", "log_alphas"]
    logger = Logger(args.output, fields)

    # Train
    ite = 0
    total_steps = 0
    while total_steps < args.max_steps:

        ite += 1

        fitness = []
        c_losses = []
        a_losses = []
        pop = drla.sample_pop(args.pop_size)

        # Update actors and critic
        if total_steps >= args.start_steps:
            for i in range(args.pop_size):
                c_loss, a_loss = drla.train(memory, 1000, pop[i])
                c_losses.append(c_loss)
                a_losses.append(a_loss)

        # Evaluate actors
        actor_steps = 0
        for i in range(args.pop_size):
            score, steps = evaluate(
                drla, pop[i], env, memory, noise=a_noise, random=total_steps <= args.start_steps, n_steps=args.n_steps,
                render=args.render)
            fitness.append(score)

            actor_steps += steps
            total_steps += steps

            prLightPurple(
                "Iteration {}; Noisy Actor fitness:{}".format(ite, score))
            print(pop[i])

        # Update distribution parameters
        if total_steps >= args.start_steps:
            drla.update_dist(fitness, pop)

        # Log scores
        score, _ = evaluate(
            drla, None, env, memory=None, noise=None, n_episodes=10)
        # "eval_score", "total_steps", "train_scores", "sampled_archi", "log_alphas"
        logger.append([score, total_steps, fitness, pop, drla.actor.log_alphas])
        print("---------------------------------")
        prRed("Total steps: {}; Actor fitness:{} \n".format(
            total_steps, score))
        drla.save(args.output)

        # Save models
        if args.save_all_models and total_steps % 100000 == 0:
            drla.actor.save_model(args.output, "actor_{}".format(total_steps))
            drla.critic.save_model(args.output, "critic_{}".format(total_steps))
            memory.save(args.output)

