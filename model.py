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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=4, stride=4)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.act2 = nn.SiLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=4)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.act3 = nn.SiLU()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.bnorm4 = nn.BatchNorm2d(256)
        self.act4 = nn.SiLU()
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=4, stride=4)
        self.act5 = nn.SiLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 256)
        self.act_fc = nn.SiLU()
        self.out = nn.Linear(256, action_dim)
        self.act_out = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.conv4.weight.data.uniform_(*hidden_init(self.conv4))
        self.conv5.weight.data.uniform_(*hidden_init(self.conv5))
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
        x = self.bnorm3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bnorm4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.act5(x)
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
    def __init__(self, action_dim, seed=42):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=4, stride=4)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.act2 = nn.SiLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=4)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.act3 = nn.SiLU()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=1, padding=1)
        self.bnorm4 = nn.BatchNorm2d(256)
        self.act4 = nn.SiLU()
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=4, stride=4)
        self.act5 = nn.SiLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512+action_dim, 256)
        self.act_fc = nn.SiLU()
        self.out = nn.Linear(256, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.conv4.weight.data.uniform_(*hidden_init(self.conv4))
        self.conv5.weight.data.uniform_(*hidden_init(self.conv5))
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
        x = self.bnorm3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bnorm4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.act5(x)
        x = self.flatten(x)
        x = torch.cat((x, action), dim=1)
        x = self.fc(x)
        x = self.act_fc(x)
        output = self.out(x)
        return output
