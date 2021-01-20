import numpy as np
import torch
import torch.nn as nn


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """
    Actor (Policy) Model.

    Attributes
    ----------
    action_dim
    seed
    """
    def __init__(self, action_dim, seed=42):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=6, stride=6)
        self.bnorm1 = nn.BatchNorm2d(64)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=4)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.SiLU()
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=4, stride=1)
        self.act3 = nn.SiLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, 128)
        self.act_fc = nn.SiLU()
        self.out = nn.Linear(128, action_dim)
        self.act_out = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.fc.weight.data.uniform_(*hidden_init(self.fc))
        self.out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.conv1(state)
        x = self.bnorm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.act_fc(x)
        x = self.out(x)
        output = self.act_out(x)
        return output


class Critic(nn.Module):
    """Critic (Value) Model.

    Attributes
    ----------
    action_dim
    seed
    """
    def __init__(self,
                 action_dim,
                 seed=42):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=6, stride=6)
        self.bnorm1 = nn.BatchNorm2d(64)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=4)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.SiLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=4, stride=1)
        self.act3 = nn.SiLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256+action_dim, 128)
        self.act_fc = nn.SiLU()
        self.out = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.fc.weight.data.uniform_(*hidden_init(self.fc))
        self.out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps (state, action) pairs to
        Q-values.
        """
        x = self.conv1(state)
        x = self.bnorm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.flatten(x)
        x = torch.cat((x, action), dim=1)
        x = self.fc(x)
        x = self.act_fc(x)
        output = self.out(x)
        return output
