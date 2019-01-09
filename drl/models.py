from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import itertools

from mab.mab import UCB

from utils.optimizers import Adam, BasicSGD
from utils.utils import to_numpy, to_base_10, to_base_n

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class NullOp(nn.Module):
    def forward(self, input):
        return torch.zeros_like(input)


class RLNN(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(RLNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def set_grads(self, params):
        """
        Sets the current gradient
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.grad.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.grad.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        """
        Loads the model
        """
        if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/{}.pkl'.format(filename, net_name),
                       map_location=lambda storage, loc: storage)
        )

    def save_model(self, output, net_name):
        """
        Saves the model
        """
        torch.save(
            self.state_dict(),
            '{}/{}.pkl'.format(output, net_name)
        )


class MultiCriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(MultiCriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3_list = nn.ModuleDict()

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6_list = nn.ModuleDict()

        # Filling Module Dicts
        for ops in itertools.product(range(args.n_ops), repeat=args.n_cells):
            key = "_".join(str(ops)[1:-1].replace(" ", "").split(','))
            self.l3_list[key] = nn.Linear(300, 1)
            self.l6_list[key] = nn.Linear(300, 1)

        self.layer_norm = args.layer_norm
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u, key):

        x = torch.cat([x, u], 1)

        x1 = F.leaky_relu(self.l1(x))
        x1 = F.leaky_relu(self.l2(x1))
        x1 = self.l3_list[key](x1)

        x2 = F.leaky_relu(self.l4(x))
        x2 = F.leaky_relu(self.l5(x2))
        x2 = self.l6_list[key](x2)

        return x1, x2


class MetaActor(RLNN):

    def __init__(self, input_dim, output_dim, max_output, args):
        super(MetaActor, self).__init__(input_dim, output_dim, max_output)

        # Misc
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_output = max_output

        # Hyper parameters
        self.hidden_size = args.hidden_size
        self.n_cells = args.n_cells
        self.ops = [torch.nn.modules.Tanh(), torch.nn.modules.ReLU(), torch.nn.modules.ELU(), torch.nn.modules.LeakyReLU()] # , NullOp()]
        self.n_ops = len(self.ops)

        # Sampling params
        self.log_alphas = np.zeros((self.n_cells - 1, self.n_cells, self.n_ops))
        self.normalize_alpha()
        self.alphas_opt = BasicSGD(args.distrib_lr)

        # Layers
        self.fc_list = nn.ModuleDict()
        for i in range(self.n_cells - 1):
            for j in range(self.n_cells):

                input_ = self.hidden_size
                output = self.hidden_size

                if i == 0:
                    input_ = self.input_dim

                for o in range(self.n_ops):
                    self.fc_list["{}_{}_{}".format(i, j, o)] = nn.Sequential(nn.Linear(input_, output), self.ops[o])

        self.out = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x, ops_mat):

        batch_size = x.shape[0]

        # Forward pass
        inputs = FloatTensor(self.n_cells, batch_size, self.hidden_size).fill_(0)
        outputs = FloatTensor(self.n_cells - 1, self.n_cells, batch_size, self.hidden_size).fill_(0)

        # Edges starting from first node
        for j in range(1, self.n_cells):
            outputs[0, j] = self.fc_list["{}_{}_{}".format(0, j, ops_mat[0, j])](x)

        # Other edges
        for i in range(1, self.n_cells):
            tmp = outputs[:i, i].clone().reshape(-1, batch_size, self.hidden_size)
            tmp_sum = torch.sum(tmp, dim=0)
            inputs[i] = tmp_sum

            for j in range(i + 1, self.n_cells):
                outputs[i, j] = self.fc_list["{}_{}_{}".format(i, j, ops_mat[i, j])](tmp_sum)

        # Output edges
        result = inputs[-1]
        result = self.max_output * torch.tanh(self.out(result))

        return result

    def sample_ops(self):
        """
        Sample the current architecture
        """
        ops_mat = np.zeros((self.n_cells - 1, self.n_cells), dtype=int)

        # Sampling operation at each edge
        for i in range(self.n_cells - 1):
            for j in range(i + 1, self.n_cells):
                ops_mat[i, j] = np.random.choice(self.n_ops, p=np.exp(self.log_alphas[i][j]))

        return ops_mat

    def update_dist(self, fitness, ops_mats):
        """
        Update the distribution with Adam
        """

        # computing gradient
        grad = np.zeros((self.n_cells - 1, self.n_cells, self.n_ops))
        for n in range(len(fitness)):
            for i in range(self.n_cells - 1):
                for j in range(i + 1, self.n_cells):
                    grad[i, j, ops_mats[n][i, j]] += fitness[n]
        grad = grad / len(fitness)

        # update distribution
        step = self.alphas_opt.step(grad.reshape(-1,))
        self.log_alphas = self.log_alphas - step.reshape(self.log_alphas.shape)
        self.normalize_alpha()
        print(np.exp(self.log_alphas))

    def normalize_alpha(self):
        """
        Transforms log_alphas into log_probabilities
        """
        self.log_alphas -= np.log(np.sum(np.exp(self.log_alphas), axis=2, keepdims=True))


class MetaActorv2(RLNN):

    def __init__(self, input_dim, output_dim, max_output, args):
        super(MetaActorv2, self).__init__(input_dim, output_dim, max_output)

        # Misc
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_output = max_output

        # Hyper parameters
        self.hidden_size = args.hidden_size
        self.n_cells = args.n_cells
        self.ops = [torch.nn.modules.Tanh(), torch.nn.modules.ReLU(), torch.nn.modules.ELU(), torch.nn.modules.LeakyReLU()] # , NullOp()]
        self.n_ops = len(self.ops)

        # Sampling params
        self.pop_size = args.pop_size
        self.log_alphas = np.zeros((self.n_cells - 1, self.n_cells, self.n_ops))
        self.normalize_alpha()
        self.alphas_opt = BasicSGD(args.distrib_lr)
        self.weights = np.array([np.log((self.pop_size + 1) / i) for i in range(1, self.pop_size + 1)])

        # Layers
        self.fc_list = nn.ModuleDict()
        for i in range(self.n_cells - 1):
            for j in range(self.n_cells):

                input_ = self.hidden_size
                output = self.hidden_size

                if i == 0:
                    input_ = self.input_dim

                for o in range(self.n_ops):
                    self.fc_list["{}_{}_{}".format(i, j, o)] = nn.Sequential(nn.Linear(input_, output), self.ops[o])

        self.out = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x, ops_mat):

        batch_size = x.shape[0]

        # Forward pass
        inputs = FloatTensor(self.n_cells, batch_size, self.hidden_size).fill_(0)
        outputs = FloatTensor(self.n_cells - 1, self.n_cells, batch_size, self.hidden_size).fill_(0)

        # Edges starting from first node
        for j in range(1, self.n_cells):
            outputs[0, j] = self.fc_list["{}_{}_{}".format(0, j, ops_mat[0, j])](x)

        # Other edges
        for i in range(1, self.n_cells):
            tmp = outputs[:i, i].clone().reshape(-1, batch_size, self.hidden_size)
            tmp_sum = torch.sum(tmp, dim=0)
            inputs[i] = tmp_sum

            for j in range(i + 1, self.n_cells):
                outputs[i, j] = self.fc_list["{}_{}_{}".format(i, j, ops_mat[i, j])](tmp_sum)

        # Output edges
        result = inputs[-1]
        result = self.max_output * torch.tanh(self.out(result))

        return result

    def sample_ops(self):
        """
        Sample the current architecture
        """
        ops_mat = np.zeros((self.n_cells - 1, self.n_cells), dtype=int)

        # Sampling operation at each edge
        for i in range(self.n_cells - 1):
            for j in range(i + 1, self.n_cells):
                ops_mat[i, j] = np.random.choice(self.n_ops, p=np.exp(self.log_alphas[i][j]))

        return ops_mat

    def update_dist(self, fitness, ops_mats):
        """
        Update the distribution with Adam
        """

        # sorting by fitness
        arg_sort = np.argsort(fitness)[::-1]

        grad = np.zeros((self.n_cells - 1, self.n_cells, self.n_ops))
        for n in arg_sort:
            for i in range(self.n_cells - 1):
                for j in range(i + 1, self.n_cells):
                    grad[i, j, ops_mats[n][i, j]] += self.weights[n]
        grad = grad / len(fitness)

        step = self.alphas_opt.step(grad.reshape(-1,))
        self.log_alphas = self.log_alphas - step.reshape(self.log_alphas.shape)
        self.normalize_alpha()
        print(np.exp(self.log_alphas))

    def normalize_alpha(self):
        """
        Transforms log_alphas into log_probabilities
        """
        self.log_alphas -= np.log(np.sum(np.exp(self.log_alphas), axis=2, keepdims=True))


class MetaActorv3(RLNN):

    def __init__(self, input_dim, output_dim, max_output, args):
        super(MetaActorv3, self).__init__(input_dim, output_dim, max_output)

        # Misc
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_output = max_output

        # Hyper parameters
        self.hidden_size = args.hidden_size
        self.n_cells = args.n_cells
        self.ops = [torch.nn.modules.Tanh(), torch.nn.modules.ReLU(), torch.nn.modules.ELU(), torch.nn.modules.LeakyReLU()] # , NullOp()]
        self.n_ops = len(self.ops)
        self.n_links = (self.n_cells * (self.n_cells - 1)) // 2

        # Sampling params
        self.pop_size = args.pop_size
        self.n_archi = self.n_ops ** self.n_links
        self.log_alphas = np.zeros(self.n_archi)
        self.normalize_alpha()
        self.alphas_opt = BasicSGD(args.distrib_lr)
        self.weights = np.array([np.log((self.pop_size + 1) / i) for i in range(1, self.pop_size + 1)])
        print(self.weights)

        # Layers
        self.fc_list = nn.ModuleDict()
        for i in range(self.n_cells - 1):
            for j in range(self.n_cells):

                input_ = self.hidden_size
                output = self.hidden_size

                if i == 0:
                    input_ = self.input_dim

                for o in range(self.n_ops):
                    self.fc_list["{}_{}_{}".format(i, j, o)] = nn.Sequential(nn.Linear(input_, output), self.ops[o])

        self.out = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x, ops_mat):

        batch_size = x.shape[0]

        # Forward pass
        inputs = FloatTensor(self.n_cells, batch_size, self.hidden_size).fill_(0)
        outputs = FloatTensor(self.n_cells - 1, self.n_cells, batch_size, self.hidden_size).fill_(0)

        # Edges starting from first node
        for j in range(1, self.n_cells):
            outputs[0, j] = self.fc_list["{}_{}_{}".format(0, j, ops_mat[0, j])](x)

        # Other edges
        for i in range(1, self.n_cells):
            tmp = outputs[:i, i].clone().reshape(-1, batch_size, self.hidden_size)
            tmp_sum = torch.sum(tmp, dim=0)
            inputs[i] = tmp_sum

            for j in range(i + 1, self.n_cells):
                outputs[i, j] = self.fc_list["{}_{}_{}".format(i, j, ops_mat[i, j])](tmp_sum)

        # Output edges
        result = inputs[-1]
        result = self.max_output * torch.tanh(self.out(result))

        return result

    def sample_ops(self):
        """
        Sample the current architecture
        """

        # selecting index
        ind = np.random.choice(self.n_archi, p=np.exp(self.log_alphas))
        key = to_base_n(ind, self.n_ops, self.n_links)

        # sampling architecture
        ops_mat = np.zeros((self.n_cells - 1, self.n_cells), dtype=np.int)
        ops_mat[np.triu_indices(self.n_cells, k=1)] = key

        return ops_mat

    def update_dist(self, fitness, ops_mats):
        """
        Update the distribution
        """

        # sort by fitness
        arg_sort = np.argsort(fitness)[::-1]

        # compute gradient
        grad = np.zeros(self.n_archi)
        for i in arg_sort:
            key = str(ops_mats[i][np.triu_indices(self.n_cells, k=1)])[1:-1].split(" ")
            key = to_base_10(key, self.n_ops)
            grad[key] += self.weights[i]

        # update distribution
        step = self.alphas_opt.step(grad.reshape(-1,))
        print(-step)

        self.log_alphas = self.log_alphas - step.reshape(self.log_alphas.shape)
        self.normalize_alpha()

        print(np.exp(self.log_alphas))

    def normalize_alpha(self):
        """
        Transforms log_alphas into log_probabilities
        """
        self.log_alphas -= np.log(np.sum(np.exp(self.log_alphas)))


class MetaActorMAB(RLNN):

    def __init__(self, input_dim, output_dim, max_output, args):
        super(MetaActorMAB, self).__init__(input_dim, output_dim, max_output)

        # Misc
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_output = max_output

        # Hyper parameters
        self.hidden_size = args.hidden_size
        self.n_cells = args.n_cells
        self.ops = [torch.nn.modules.Tanh(), torch.nn.modules.ReLU(), torch.nn.modules.ELU(), torch.nn.modules.LeakyReLU()] # , NullOp()]
        self.n_ops = len(self.ops)
        self.n_links = (self.n_cells * (self.n_cells - 1)) // 2

        # Sampling params
        self.pop_size = args.pop_size
        self.ucbs = [UCB(self.n_ops) for _ in range(self.n_links)]
        self.log_alphas = None

        # Layers
        self.fc_list = nn.ModuleDict()
        for i in range(self.n_cells - 1):
            for j in range(self.n_cells):

                input_ = self.hidden_size
                output = self.hidden_size

                if i == 0:
                    input_ = self.input_dim

                for o in range(self.n_ops):
                    self.fc_list["{}_{}_{}".format(i, j, o)] = nn.Sequential(nn.Linear(input_, output), self.ops[o])

        self.out = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x, ops_mat):

        batch_size = x.shape[0]

        # Forward pass
        inputs = FloatTensor(self.n_cells, batch_size, self.hidden_size).fill_(0)
        outputs = FloatTensor(self.n_cells - 1, self.n_cells, batch_size, self.hidden_size).fill_(0)

        # Edges starting from first node
        for j in range(1, self.n_cells):
            outputs[0, j] = self.fc_list["{}_{}_{}".format(0, j, ops_mat[0, j])](x)

        # Other edges
        for i in range(1, self.n_cells):
            tmp = outputs[:i, i].clone().reshape(-1, batch_size, self.hidden_size)
            tmp_sum = torch.sum(tmp, dim=0)
            inputs[i] = tmp_sum

            for j in range(i + 1, self.n_cells):
                outputs[i, j] = self.fc_list["{}_{}_{}".format(i, j, ops_mat[i, j])](tmp_sum)

        # Output edges
        result = inputs[-1]
        result = self.max_output * torch.tanh(self.out(result))

        return result

    def sample_ops(self):
        """
        Sample the current architecture
        """
        ops_mat = np.zeros((self.n_cells - 1, self.n_cells), dtype=int)
        tmp = 0

        # Sampling operation at each edge
        for i in range(self.n_cells - 1):
            for j in range(i + 1, self.n_cells):
                ops_mat[i, j] = self.ucbs[tmp].sample()
                tmp += 1

        return ops_mat

    def update_dist(self, fitness, ops_mats):
        """
        Update the distribution with Adam
        """
        for n in range(self.pop_size):
            tmp = 0
            for i in range(self.n_cells - 1):
                for j in range(i + 1, self.n_cells):
                    arm = ops_mats[n][i, j]
                    self.ucbs[tmp].update_arm(arm, fitness[n])
                    tmp += 1

        tmp = 0
        for i in range(self.n_cells - 1):
            for j in range(i + 1, self.n_cells):
                print(self.ucbs[tmp].means)
                print(self.ucbs[tmp].compute_confidences())
                tmp += 1



class NASActor(RLNN):

    def __init__(self, input_dim, output_dim, max_output, args):
        super(NASActor, self).__init__(input_dim, output_dim, max_output)

        # Misc
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_output = max_output

        # hyper-parameters
        self.hidden_size = args.hidden_size
        self.n_cells = args.n_cells
        self.ops = [torch.nn.modules.Tanh(), torch.nn.modules.ReLU(), torch.nn.modules.ELU(), torch.nn.modules.LeakyReLU(), NullOp()]
        self.n_ops = len(self.ops)

        # Layers
        self.fc_list = nn.ModuleDict()
        for i in range(self.n_cells - 1):
            for j in range(self.n_cells):

                input = self.hidden_size
                output = self.hidden_size

                if i == 0:
                    input = self.input_dim

                for o in range(self.n_ops):
                    self.fc_list["{}_{}_{}".format(i, j, o)] = nn.Sequential(nn.Linear(input, output), self.ops[o])

        self.out = nn.Linear(self.hidden_size, self.output_dim)

        # Sampling params
        self.tau = 1
        self.log_alphas = nn.Parameter(FloatTensor(self.n_cells - 1, self.n_cells, self.n_ops).fill_(-np.log(self.n_ops)))
        self.cpt = 1

    def forward(self, x):

        batch_size = x.shape[0]

        # Sampling stuff
        self.sample()
        z = torch.exp((self.log_alphas + self.eps) / self.tau)
        z = z / torch.sum(z, dim=2, keepdim=True)

        self.cpt += 1

        # Forward pass
        inputs = FloatTensor(self.n_cells, batch_size, self.hidden_size).fill_(0)
        outputs = FloatTensor(self.n_cells - 1, self.n_cells, self.n_ops, batch_size, self.hidden_size).fill_(0)

        # Edges starting from first node
        for j in range(1, self.n_cells):
            for o in range(self.n_ops):
                outputs[0, j, o] = self.fc_list["{}_{}_{}".format(0, j, o)](x)

        # Other edges
        for i in range(1, self.n_cells):
            tmp = outputs[:i, i].clone().reshape(-1, batch_size, self.hidden_size)
            tmp = tmp * z[:i, i].reshape(-1, 1, 1)
            tmp_sum = torch.sum(tmp, dim=0)
            inputs[i] = tmp_sum

            for j in range(i + 1, self.n_cells):
                for o in range(self.n_ops):
                    outputs[i, j, o] = self.fc_list["{}_{}_{}".format(i, j, o)](tmp_sum)

        # Output edges
        result = inputs[-1]
        # for i in range(self.n_cells - 1):
        #     result = result + inputs[i] * torch.prod(z[i, (i + 1):, -1])
        result = self.max_output * torch.tanh(self.out(result))

        return result

    def sample(self):
        self.eps = FloatTensor(self.n_cells - 1, self.n_cells, self.n_ops).uniform_()
        self.eps = -torch.log(-torch.log(self.eps))

    def reduce_temp(self, rate):
        self.tau = rate * self.tau + (1 - self.tau) * 1e-5

    def normalize_alpha(self):
        with torch.no_grad():
            self.log_alphas -= torch.log(torch.sum(torch.exp(self.log_alphas), dim=2, keepdim=True))


class GaussianActor(RLNN):
    """
    Gaussian Policy
    """

    def __init__(self, state_dim, action_dim, max_action, args):
        super(GaussianActor, self).__init__(state_dim, action_dim, max_action)

        # Mean parameters
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        # Sigma parameters
        self.l4 = nn.Linear(state_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, action_dim)

        # Misc
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sigma_max = 0.3

    def forward(self, x, rand=True):

        mu = torch.tanh(self.l1(x))
        mu = torch.tanh(self.l2(mu))
        mu = self.max_action * torch.tanh(self.l3(mu))

        s = torch.tanh(self.l4(x))
        s = torch.tanh(self.l5(s))
        s = self.sigma_max * torch.tanh(self.l6(s))

        if rand:
            noise = FloatTensor(self.action_dim).normal_()
        else:
            noise = FloatTensor(self.action_dim).fill_(0)

        return mu + noise * s ** 2, mu, s


class FoundActor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(FoundActor, self).__init__(state_dim, action_dim, max_action)

        self.l12 = nn.Linear(state_dim, 400)
        self.l13 = nn.Linear(state_dim, 400)
        self.l23 = nn.Linear(400, 400)
        self.lout = nn.Linear(400, action_dim)

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x1):

        x2 = F.relu(self.l12(x1))
        x3 = torch.tanh(self.l13(x1)) + F.leaky_relu(self.l23(x2))
        out = self.max_action * torch.tanh(self.lout(x3))

        return out


class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.layer_norm = args.layer_norm
        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x):

        if not self.layer_norm:
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
            x = self.max_action * torch.tanh(self.l3(x))

        else:
            x = torch.tanh(self.n1(self.l1(x)))
            x = torch.tanh(self.n2(self.l2(x)))
            x = self.max_action * torch.tanh(self.l3(x))

        return x


class Value(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Value, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.layer_norm = args.layer_norm
        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x):

        if not self.layer_norm:
            x = torch.leaky_relu(self.l1(x))
            x = torch.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = torch.leaky_relu(self.n1(self.l1(x)))
            x = torch.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x


class ValueTD3(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(ValueTD3, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        self.l4 = nn.Linear(state_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n3 = nn.LayerNorm(400)
            self.n4 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x):

        if not self.layer_norm:
            x1 = F.leaky_relu(self.l1(x))
            x1 = F.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = F.relu(self.n1(self.l1(x)))
            x1 = F.relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = F.leaky_relu(self.l4(x))
            x2 = F.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = F.relu(self.n3(self.l4(x)))
            x2 = F.relu(self.n4(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim, 200)
        self.l1_ = nn.Linear(action_dim, 200)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u):

        if not self.layer_norm:
            x = F.leaky_relu(self.l1(x))
            u = F.leaky_relu(self.l1_(u))
            x = F.leaky_relu(self.l2(torch.cat([x, u], 1)))
            x = self.l3(x)

        else:
            x = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n3 = nn.LayerNorm(400)
            self.n4 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u):

        x = torch.cat([x, u], 1)

        if not self.layer_norm:
            x1 = F.leaky_relu(self.l1(x))
            x1 = F.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = F.relu(self.n1(self.l1(x)))
            x1 = F.relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = F.leaky_relu(self.l4(x))
            x2 = F.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = F.relu(self.n3(self.l4(x)))
            x2 = F.relu(self.n4(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2
