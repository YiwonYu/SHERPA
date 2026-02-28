import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2, residual=False):
        super(SEBlock, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.residual = residual

    def forward(self, x):
        B, C, H, W = x.size()
        squeezed = x.view(B, C, -1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(squeezed))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeezed.size()
        out = torch.mul(x, fc_out_2.view(a, b, 1, 1))
        if self.residual:
            out = out + x
        return out, None, None