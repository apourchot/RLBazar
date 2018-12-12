import numpy as np
import networkx as nx
from drl.models import RLNN

class Neat(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Neat, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.nn_graph = nx.DiGraph()