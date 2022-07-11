# -*- coding: utf-8 -*-
# !@time: 2022/5/31 上午6:51
# !@author: superMC @email: 18758266469@163.com
# !@fileName: inference.py


import os
import torch
import torch.nn as nn
from model import SessionGraph, forward

model = torch.load('./models/SessionGraph.pt')
model.eval()
forward(model)
