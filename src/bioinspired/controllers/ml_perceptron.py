"""Multilayer Perceptron Controller

This module implements a multilayer perceptron (MLP) controller for an arbitrary agent.
"""

import numpy as np
import torch
import torch.nn as nn

from overrides import override

from .controller_base import ControllerBase


class MLPController(nn.Module, ControllerBase):
    """Multilayer Perceptron Controller for an agent.
    This neural net takes a state vector (from the simulator) and outputs a thrust vector.
    """

    def __init__(self, hidden_sizes, output_size, simulator, lander_name, target_name):
        super().__init__()
        ControllerBase.__init__(self, simulator, lander_name, target_name)
        layers = []
        last_size = self.get_input_size()
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.ReLU())
            last_size = h
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)

    def _(self, state_vector):
        """Forward pass: takes a state vector and returns thrust vector."""
        if not isinstance(state_vector, torch.Tensor):
            state_vector = torch.tensor(state_vector, dtype=torch.float32)
        return self.network(state_vector)

    def get_weights(self):
        """Returns all weights as a single flat tensor."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_weights(self, flat_weights):
        """Sets weights from a flat tensor (for evolutionary algorithms)."""
        pointer = 0
        for p in self.parameters():
            numel = p.data.numel()
            p.data.copy_(flat_weights[pointer : pointer + numel].view_as(p.data))
            pointer += numel

    @override
    def get_control_action(self):
        """Get the control action based on the current state of the simulation."""
        state_vector = self.extract_state_vector()
        thrust_vector = self._(state_vector)
        return thrust_vector.detach().numpy()