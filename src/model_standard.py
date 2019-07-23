import torch
import torch.nn as nn
import torch.nn.functional as F


class QStandardNetwork(nn.Module):
    """A (standard) Deep Q-Network (DQN) that represents an actor's policy model."""


    def __init__(self, state_size, action_size):
        """Builds the model and initializes its parameters.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(QStandardNetwork, self).__init__()

        self.layer_1 = nn.Linear(in_features=state_size, out_features=64)
        self.layer_2 = nn.Linear(in_features=64, out_features=32)
        self.layer_3 = nn.Linear(in_features=32, out_features=16)
        self.layer_4 = nn.Linear(in_features=16, out_features=action_size)     


    def forward(self, state):
        """Passes a state through the network.
        
        Params
        ======
            state (array_like): the state
            
        Returns
        =======
            The result of the state passed through the network.
        """

        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = self.layer_4(x)

        return x
