# Copied from https://github.com/alibaba-edu/simple-effective-text-matching-pytorch
__author__ = "Alibaba Group"

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

import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
from ..utils.registry import register
from . import Linear, Module

registry = {}
register = partial(register, registry=registry)


@register('identity')
class Alignment(Module):
    def __init__(self, __, hidden_size, _):
        super().__init__()

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2))

    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float()).byte()#.bool()
        attn.masked_fill_(~mask, -1e7)
        attn_a = f.softmax(attn, dim=1)
        attn_b = f.softmax(attn, dim=2)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b, b)
        return feature_a, feature_b


@register('linear')
class MappedAlignment(Alignment):
    def __init__(self, input_size, hidden_size, drop):
        super().__init__(input_size, hidden_size, drop)
        self.dropout = drop
        self.projection = Linear(input_size, hidden_size, activations=True)
        # self.training = training

    def _attention(self, a, b):
        a = f.dropout(a, self.dropout, self.training)
        b = f.dropout(b, self.dropout, self.training)
        a = self.projection(a)
        b = self.projection(b)
        return super()._attention(a, b)

