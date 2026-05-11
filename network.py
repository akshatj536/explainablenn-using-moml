"""
network.py
==========
Defines the InterpretableNN — a shallow neural network whose forward pass is
gated by binary connection matrices (masks).

Architecture:
    input (n_input)  →  [mask1]  →  hidden (n_hidden_max)  →  [mask2]  →  output (n_output)

The binary mask is NOT a learnable parameter; it is a registered buffer.
During each forward pass the weight matrices are element-wise multiplied by
their corresponding mask, which zeros out inactive connections so that:
  - Inactive paths contribute nothing to the output.
  - Gradient updates for masked weights are zero, keeping them at whatever
    value they were initialised to (they are effectively frozen/ignored).

Interpretability metrics (all computed from the masks):
  - Active connections  : total number of 1-bits across both masks.
  - Active neurons      : hidden units that have ≥1 incoming AND ≥1 outgoing
                          connection (i.e. they actually process information).
  - Active input features: input dimensions that feed into at least one
                          hidden neuron.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class InterpretableNN(nn.Module):
    """
    Interpretable neural network controlled by binary connection matrices.

    Parameters
    ----------
    n_input : int
        Number of input features.
    n_hidden_max : int
        Maximum (physical) width of the single hidden layer.
        The *effective* width depends on the masks.
    n_output : int
        Number of output neurons (1 for binary classification / regression).
    mask1 : np.ndarray, shape (n_input, n_hidden_max), dtype int/bool
        Connection mask between input and hidden layer.
    mask2 : np.ndarray, shape (n_hidden_max, n_output), dtype int/bool
        Connection mask between hidden and output layer.
    task : str
        'classification' → sigmoid output + BCE loss
        'regression'     → linear output + MSE loss
    """

    def __init__(self,
                 n_input: int,
                 n_hidden_max: int,
                 n_output: int,
                 mask1: np.ndarray,
                 mask2: np.ndarray,
                 task: str = 'classification'):
        super().__init__()

        self.n_input      = n_input
        self.n_hidden_max = n_hidden_max
        self.n_output     = n_output
        self.task         = task

        # Learnable parameters — shape mirrors the masks
        # Kaiming-uniform style initialisation scaled to mask density
        bound1 = np.sqrt(6.0 / max(1, int(mask1.sum(axis=0).mean() + 1e-9)))
        bound2 = np.sqrt(6.0 / max(1, int(mask2.sum(axis=0).mean() + 1e-9)))

        self.W1 = nn.Parameter(
            torch.empty(n_input, n_hidden_max).uniform_(-bound1, bound1))
        self.b1 = nn.Parameter(torch.zeros(n_hidden_max))
        self.W2 = nn.Parameter(
            torch.empty(n_hidden_max, n_output).uniform_(-bound2, bound2))
        self.b2 = nn.Parameter(torch.zeros(n_output))

        # Binary masks stored as non-learnable buffers
        m1 = torch.tensor(mask1, dtype=torch.float32)
        m2 = torch.tensor(mask2, dtype=torch.float32)
        self.register_buffer('mask1', m1)
        self.register_buffer('mask2', m2)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Masked forward pass.

        Effective weights are W * mask, so connections with mask=0
        never affect the output — equivalently, those synapses do not exist.

        Shapes:
            x          : (batch, n_input)
            h_pre      : (batch, n_hidden_max)
            h          : (batch, n_hidden_max)  after ReLU
            out        : (batch, n_output)
        """
        # Mask weights in-place (element-wise product; no gradient for mask)
        W1_eff = self.W1 * self.mask1   # (n_input, n_hidden_max)
        W2_eff = self.W2 * self.mask2   # (n_hidden_max, n_output)

        h   = torch.relu(x @ W1_eff + self.b1)
        out = h @ W2_eff + self.b2

        if self.task == 'classification':
            out = torch.sigmoid(out)

        return out

    # ------------------------------------------------------------------
    # Interpretability metrics
    # ------------------------------------------------------------------

    def get_active_connections(self) -> int:
        """Total number of active (non-zero mask) connections across both layers."""
        return int(self.mask1.sum().item() + self.mask2.sum().item())

    def get_active_neurons(self) -> int:
        """
        Count hidden neurons that are truly 'alive':
        they must have at least one incoming connection from the input layer
        AND at least one outgoing connection to the output layer.
        Dead neurons — those with all-zero rows in mask1 or all-zero rows in
        mask2 — are excluded; they can never contribute to the output.
        """
        has_incoming = (self.mask1.sum(dim=0) > 0)   # shape (n_hidden_max,)
        has_outgoing = (self.mask2.sum(dim=1) > 0)   # shape (n_hidden_max,)
        return int((has_incoming & has_outgoing).sum().item())

    def get_active_features(self) -> int:
        """
        Number of input features that actually participate in the network
        (i.e. connected to at least one hidden neuron).
        """
        return int((self.mask1.sum(dim=1) > 0).sum().item())

    def get_complexity_tuple(self) -> Tuple[int, int, int]:
        """Return (active_connections, active_neurons, active_features)."""
        return (self.get_active_connections(),
                self.get_active_neurons(),
                self.get_active_features())

    def update_masks(self, mask1: np.ndarray, mask2: np.ndarray):
        """
        Replace the connection masks in-place (used after mutation to keep
        the same model object but with a different topology).
        """
        self.mask1.copy_(torch.tensor(mask1, dtype=torch.float32))
        self.mask2.copy_(torch.tensor(mask2, dtype=torch.float32))
