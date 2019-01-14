import torch
import gym.spaces

from utils.utils import evaluate, get_output_folder, prLightPurple, prRed
from utils.memory import Memory
from utils.logger import Logger
from utils.args import parser
from utils.random_process import GaussianNoise

from drl.td3 import NASTD3

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
    drla = NASTD3(state_dim, action_dim, max_action, args)

    # Action noise
    a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)

    # Logger
    fields = ["eval_score", "critic_loss", "actor_loss", "total_steps"]
    logger = Logger(args.output, fields)

    # Train
    ite = 0
    K = 5000
    total_steps = 0
    c_loss, a_loss = None, None
    while total_steps < args.max_steps:

        ite += 1
        actor_steps = 0
        while actor_steps < K:

            fitness, steps = evaluate(
                drla, env, memory, noise=a_noise, random=total_steps <= args.start_steps, n_steps=args.n_steps, render=args.render)

            if total_steps > args.start_steps:
                c_loss, a_loss = drla.train(memory, steps)

            actor_steps += steps
            total_steps += steps

            prLightPurple(
                "Iteration {}; Noisy Actor fitness:{}; Q-Loss:{}; A-Loss:{}".format(ite, fitness, c_loss, a_loss))

        fitness, steps = evaluate(
            drla, env, memory=None, noise=None, n_episodes=10)
        logger.append([fitness, c_loss, a_loss, total_steps])
        print("---------------------------------")
        prRed("Total steps: {}; Actor fitness:{} \n".format(
            total_steps, fitness))
        drla.save(args.output)
        print(torch.exp(drla.actor.log_alphas))

        if args.save_all_models and total_steps % 100000 == 0:
            drla.actor.save_model(args.output, "actor_{}".format(total_steps))
            drla.critic.save_model(args.output, "critic_{}".format(total_steps))
            memory.save(args.output)

