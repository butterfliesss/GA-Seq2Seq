# Copied from https://github.com/alibaba-edu/simple-effective-text-matching-pytorch
__author__ = "Alibaba Group"

# Modified by:
# Hui Ma

# Changelog:
# Add a new fusion way

# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
from ..utils.registry import register
from . import Linear

registry = {}
register = partial(register, registry=registry)


@register('simple')
class Fusion(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fusion = Linear(input_size * 2, hidden_size, activations=True)

    def forward(self, x, align):
        return self.fusion(torch.cat([x, align], dim=-1))


@register('full')
class FullFusion(nn.Module):
    def __init__(self, input_size, hidden_size, drop):
        super().__init__()
        self.dropout = drop
        self.fusion1 = Linear(input_size * 2, hidden_size, activations=True)
        self.fusion2 = Linear(input_size * 2, hidden_size, activations=True)
        self.fusion3 = Linear(input_size * 2, hidden_size, activations=True)
        self.fusion = Linear(hidden_size * 3, hidden_size, activations=True)
        # # self.training = training

    def forward(self, x, align):
        x1 = self.fusion1(torch.cat([x, align], dim=-1))
        x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        x3 = self.fusion3(torch.cat([x, x * align], dim=-1))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = f.dropout(x, self.dropout, self.training)
        return self.fusion(x)

@register('my')
class FullFusion(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fusion = Linear(input_size * 4, hidden_size, activations=True)
        # # self.training = training

    def forward(self, x, align):
        return self.fusion(torch.cat([x, align, x - align, x * align], dim=-1))


