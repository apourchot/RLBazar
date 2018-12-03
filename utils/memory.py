import numpy as np
import torch
import torch.multiprocessing as mp

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

class Memory():
    """
    Simple Memory class
    """

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
            self.states = torch.zeros(self.memory_size, self.state_dim).cuda()
            self.actions = torch.zeros(self.memory_size, self.action_dim).cuda()
            self.n_states = torch.zeros(self.memory_size, self.state_dim).cuda()
            self.rewards = torch.zeros(self.memory_size, n_steps).cuda()
            self.steps = torch.zeros(self.memory_size, 1).cuda()
            self.dones = torch.zeros(self.memory_size, 1).cuda()
            self.stops = torch.zeros(self.memory_size, 1).cuda()

        else:
            self.states = torch.zeros(self.memory_size, self.state_dim)
            self.actions = torch.zeros(self.memory_size, self.action_dim)
            self.n_states = torch.zeros(self.memory_size, self.state_dim)
            self.rewards = torch.zeros(self.memory_size, n_steps)
            self.steps = torch.zeros(self.memory_size, 1)
            self.dones = torch.zeros(self.memory_size, 1)
            self.stops = torch.zeros(self.memory_size, 1)

    def size(self):
        """
        Returns the current size of the memory
        """
        if self.full:
            return self.memory_size
        return self.pos

    def get_pos(self):
        """
        Returns the current position of the cursors
        """
        return self.pos

    def add(self, datum):
        """
        Adds the given sample into memory
        """

        state, action, n_state, reward, step, done, stop = datum

        self.states[self.pos] = FloatTensor(state)
        self.actions[self.pos] = FloatTensor(action)
        self.n_states[self.pos] = FloatTensor(n_state)
        self.rewards[self.pos] = FloatTensor(reward)
        self.steps[self.pos] = LongTensor([step])
        self.dones[self.pos] = FloatTensor([done])
        self.stops[self.pos] = FloatTensor([stop])

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        """
        Sample a mini-batch from memory
        """

        upper_bound = self.memory_size if self.full else self.pos
        batch_inds = LongTensor(np.random.randint(0, upper_bound, size=batch_size))

        states = self.states[batch_inds]
        actions = self.actions[batch_inds]
        
        tmp = FloatTensor(batch_size, 1).uniform_()
        rand_steps = self.steps[batch_inds] * tmp 
        rand_steps = torch.ceil(rand_steps).type(LongTensor).view(-1)

        rewards = self.rewards[batch_inds]
        n_states = self.states[batch_inds + rand_steps]
        stops = self.stops[batch_inds + rand_steps]
        dones = self.dones[batch_inds + rand_steps]

        return (states, actions, n_states, rewards, rand_steps.type(FloatTensor).view(-1, 1), dones, stops)

