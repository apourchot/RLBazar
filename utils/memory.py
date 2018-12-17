import numpy as np
import torch

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

    def __init__(self, memory_size, state_dim, action_dim, args):

        # rl params
        self.n_steps = args.n_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # memory params
        self.pos = 0
        self.full = False
        self.memory_size = memory_size

        if USE_CUDA:
            self.states = torch.zeros(self.memory_size, self.n_steps + 1, self.state_dim).cuda()
            self.actions = torch.zeros(self.memory_size, self.n_steps, self.action_dim).cuda()
            self.rewards = torch.zeros(self.memory_size, self.n_steps).cuda()
            self.steps = torch.zeros(self.memory_size, self.n_steps).cuda()
            self.dones = torch.zeros(self.memory_size, self.n_steps).cuda()
            self.stops = torch.zeros(self.memory_size, self.n_steps).cuda()

        else:
            self.states = torch.zeros(self.memory_size, self.n_steps + 1, self.state_dim)
            self.actions = torch.zeros(self.memory_size, self.n_steps, self.action_dim)
            self.rewards = torch.zeros(self.memory_size, self.n_steps)
            self.steps = torch.zeros(self.memory_size, self.n_steps)
            self.dones = torch.zeros(self.memory_size, self.n_steps)
            self.stops = torch.zeros(self.memory_size, self.n_steps)

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

        states, actions, rewards, dones, stops = datum

        self.states[self.pos] = FloatTensor(states)
        self.actions[self.pos] = FloatTensor(actions)
        self.rewards[self.pos] = FloatTensor(rewards)
        self.dones[self.pos] = FloatTensor(dones)
        self.stops[self.pos] = FloatTensor(stops)

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        """
        Sample a mini-batch from memory
        """

        upper_bound = max(1, self.memory_size - self.n_steps if self.full else self.pos - self.n_steps)
        batch_inds = LongTensor(np.random.randint(0, upper_bound, size=batch_size))
        
        tmp = FloatTensor(batch_size, 1).uniform_()
        rand_steps = self.n_steps * tmp
        rand_steps = torch.ceil(rand_steps).type(LongTensor).view(-1) - 1

        states = self.states[batch_inds, 0]
        n_states = self.states[batch_inds, rand_steps + 1]
        actions = self.actions[batch_inds, 0]
        rewards = self.rewards[batch_inds]
        dones = self.dones[batch_inds, rand_steps].view(-1, 1)
        stops = self.stops[batch_inds, rand_steps].view(-1, 1)

        return states, actions, n_states, rewards, rand_steps.type(FloatTensor).view(-1, 1), dones, stops

    def load(self, input_dir):
        """
        Loads the content of the replay buffer from the given folder
        """
        self.states = torch.load(input_dir + "/buffer/states.pkl")
        self.actions = torch.load(input_dir + "/buffer/actions.pkl")
        self.rewards = torch.load(input_dir + "/buffer/rewards.pkl")
        self.dones = torch.load(input_dir + "/buffer/dones.pkl")
        self.stops = torch.load(input_dir + "/buffer/stops.pkl")

    def save(self, output_dir):
        """
        Save the content of the replay buffer
        """
        torch.save(self.states, output_dir + "/states.pkl")
        torch.save(self.actions, output_dir + "/actions.pkl")
        torch.save(self.rewards, output_dir + "/rewards.pkl")
        torch.save(self.dones, output_dir + "/dones.pkl")
        torch.save(self.stops, output_dir + "/stops.pkl")