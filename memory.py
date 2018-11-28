import numpy as np
import torch
import torch.multiprocessing as mp

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

# Code based on https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# and https://github.com/jingweiz/pytorch-distributed/blob/master/core/memories/shared_memory.py


class Memory():

    def __init__(self, memory_size, state_dim, action_dim, n_steps=1):

        # rl params
        self.n_steps = n_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # memory params
        self.pos = 0
        self.full = False
        self.memory_size = memory_size

        if USE_CUDA:
            self.stops = torch.zeros(self.memory_size, 1).cuda()
            self.dones = torch.zeros(self.memory_size, 1).cuda()
            self.steps = torch.zeros(self.memory_size, 1).cuda()
            self.rewards = torch.zeros(self.memory_size, n_steps).cuda()
            self.states = torch.zeros(self.memory_size, self.state_dim).cuda()
            self.actions = torch.zeros(self.memory_size, self.action_dim).cuda()
            self.n_states = torch.zeros(self.memory_size, self.state_dim).cuda()

        else:
            self.stops = torch.zeros(self.memory_size, 1)
            self.dones = torch.zeros(self.memory_size, 1)
            self.steps = torch.zeros(self.memory_size, 1)
            self.rewards = torch.zeros(self.memory_size, n_steps)
            self.states = torch.zeros(self.memory_size, self.state_dim)
            self.actions = torch.zeros(self.memory_size, self.action_dim)
            self.n_states = torch.zeros(self.memory_size, self.state_dim)

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def get_pos(self):
        return self.pos

    def add(self, datum):

        state, n_state, action, reward, step, done, stop = datum

        self.dones[self.pos] = FloatTensor([done])
        self.stops[self.pos] = FloatTensor([stop])
        self.steps[self.pos] = FloatTensor([step])
        self.states[self.pos] = FloatTensor(state)
        self.actions[self.pos] = FloatTensor(action)
        self.rewards[self.pos] = FloatTensor(reward)
        self.n_states[self.pos] = FloatTensor(n_state)

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):

        upper_bound = self.memory_size if self.full else self.pos
        batch_inds = torch.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))

        return (self.states[batch_inds],
                self.n_states[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.steps[batch_inds],
                self.dones[batch_inds],
                self.stops[batch_inds])