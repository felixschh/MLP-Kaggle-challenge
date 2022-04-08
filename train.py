from collections import OrderedDict
import enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms


trainset, testset, trainloader, testloader = get_data()
model = Classify()

criterion =  nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 50
train = []
starting_time = time.time()

for epoch in range(epochs):
    running_loss = 0
    running_loss_test = 0

    print(f'Epoch: {epoch+1}/{epochs}')
    for i, (images, labels) in enumerate(iter(trainloader)):



