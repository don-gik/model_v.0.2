import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple



class DropPath(nn.Module):
    def __init__(self,
                 p : float = 0.0
                 ):
        
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.p == 0.0:
            return x
        
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * rnd
    

class DepthwiseConv3d(nn.Module):
    def __init__(self,
                 channels : int,
                 kernel : Tuple[int, int, int] = (1, 3, 3),
                 padding : Tuple[int, int, int] = (0, 1, 1)
                ):

        super().__init__()
        self.conv = nn.Conv3d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = kernel,
            padding = padding,
            groups = channels
        )
    
    def forward(self, x):
        return self.conv(x)
