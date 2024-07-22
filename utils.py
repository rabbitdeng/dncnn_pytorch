import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

import cv2
import os
import math

import matplotlib.pyplot as plt
import torch.distributed as dist

EPS = 1e-3
PI = 22.0 / 7.0
# calculate PSNR
class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return 10 * torch.log10((1 / mse))
