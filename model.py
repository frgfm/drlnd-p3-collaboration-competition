#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn


def make_layer(in_feats, out_feats, activation=False, bn=False, dropout=0.):
    """Create model layer

    Args:
        in_feats (int): number of input noes
        out_feats (int): number of output nodes
        activation (bool, optional): whether activation should be used
        bn (bool, optional): whether 1D batch norm should be applied
        dropout (float, optional): dropout probability

    Returns:
        list<torch.nn.Module>: list of PyTorch layers
    """

    layers = [nn.Linear(in_feats, out_feats)]
    if bn:
        layers.append(nn.BatchNorm1d(out_feats))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    return layers


class Actor(nn.Module):
    """Actor (Policy) Model.

    Args:
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        fc_units (int, optional): number of nodes in hidden layers
        bn (bool, optional): whether 1D batch norm should be applied
        dropout (float, optional): dropout probability
    """

    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300, bn=True, dropout_prob=0.):

        super(Actor, self).__init__()

        # Number of nodes in FC layers
        lin_features = [state_size, fc1_units, fc2_units, action_size]
        dropout_probs = [dropout_prob / 2] * (len(lin_features) - 2) + [dropout_prob]

        layers = []
        for idx, out_feats in enumerate(lin_features[1:]):
            layers.extend(make_layer(lin_features[idx], out_feats,
                                     idx + 2 < len(lin_features),
                                     bn and idx == 0,
                                     dropout_probs[idx] if idx + 2 < len(lin_features) else 0.))

        self.model = nn.Sequential(*layers)
        # Init last FC differently
        self.model[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        return torch.tanh(self.model(state))


class Critic(nn.Module):
    """Critic (Value) Model.

    Args:
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        fc_units (int, optional): number of nodes in hidden layers
        bn (bool, optional): whether 1D batch norm should be applied
        dropout (float, optional): dropout probability
    """

    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300, bn=False, dropout_prob=0.):

        super(Critic, self).__init__()

        # Number of nodes in FC layers
        base_feats, head_feats = [state_size, fc1_units], [fc1_units + action_size, fc2_units, 1]
        base_drops, head_drops = [dropout_prob / 2], [dropout_prob / 2] * (len(head_feats) - 2) + [dropout_prob]

        layers = []
        for idx, out_feats in enumerate(base_feats[1:]):
            layers.extend(make_layer(base_feats[idx], out_feats,
                                     True, bn and idx == 0, base_drops[idx]))
        self.base = nn.Sequential(*layers)

        layers = []
        for idx, out_feats in enumerate(head_feats[1:]):
            layers.extend(make_layer(head_feats[idx], out_feats,
                                     idx + 2 < len(head_feats),
                                     False,
                                     head_drops[idx] if idx + 2 < len(head_feats) else 0.))

        self.head = nn.Sequential(*layers)
        # Init last FC differently
        self.head[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.base(state)
        x = torch.cat((x, action), dim=1)
        return self.head(x)
