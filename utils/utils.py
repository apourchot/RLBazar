import os
import torch
import numpy as np

from copy import deepcopy

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor
    

def evaluate(actor, env, memory=None, n_steps=1, n_episodes=1, random=False, noise=None, render=False, max_action=1):
    """
    Computes the score of an actor on a given number of runs,
    fills the memory if needed
    """

    if not random:
        def policy(state):
            state = FloatTensor(state.reshape(-1))
            action = actor.action(state)

            if noise is not None:
                action += noise.sample()

            return np.clip(action, -max_action, max_action)

    else:
        def policy(state):
            return env.action_space.sample()

    scores = []
    total_steps = 0

    for _ in range(n_episodes):

        states = []
        actions = []
        rewards = []

        score = 0
        steps = 0
        done = False
        obs = deepcopy(env.reset())
        states.append(obs)

        while not done:

            # get next action and act
            action = policy(obs)
            n_obs, reward, done, _ = env.step(action)
            is_done = 0 if steps + 1 == env._max_episode_steps else float(done)
            score += reward
            steps += 1

            # update obs
            obs = n_obs
            states.append(obs)
            actions.append(action)
            rewards.append(reward)

            # render if needed
            if render:
                env.render()

            # reset when done
            if done:
                env.reset()

        # store everything into memory if needed
        if memory is not None:

            for i in range(steps):
                
                d = min(n_steps, steps - i)

                state = states[i]
                action = actions[i]
                n_state = states[i+1]

                reward = np.zeros(n_steps)
                reward[:d] = rewards[i:i + d]
                stop = is_done if i == steps - 1 else 0
                done = True if i == steps - 1 else False

                memory.add((state, action, n_state, reward, d, done, stop))

        scores.append(score)
        total_steps += steps

    return np.mean(scores), total_steps

def prRed(prt):
    print("\033[91m{}\033[00m" .format(prt))


def prGreen(prt):
    print("\033[92m{}\033[00m" .format(prt))


def prYellow(prt):
    print("\033[93m{}\033[00m" .format(prt))


def prLightPurple(prt):
    print("\033[94m{}\033[00m" .format(prt))


def prPurple(prt):
    print("\033[95m{}\033[00m" .format(prt))


def prCyan(prt):
    print("\033[96m{}\033[00m" .format(prt))


def prLightGray(prt):
    print("\033[97m{}\033[00m" .format(prt))


def prBlack(prt):
    print("\033[98m{}\033[00m" .format(prt))


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(x, dtype="float"):
    """
    Numpy array to tensor
    """

    FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return FloatTensor(x)
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return LongTensor(x)
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return ByteTensor(x)
    else:
        x = np.array(x, dtype=np.float64).tolist()

    return FloatTensor(x)


def soft_update(target, source, tau):
    """
    Performs a soft target update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Performs a hard target update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir
