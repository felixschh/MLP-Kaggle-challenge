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
from data_handler import get_data
from model import Classify


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
        images.resize_(images.size()[0], 784)
        optimizer.zero_grad()

        prediction = model.forward(images)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
    
    train.append((running_loss/64))

train_time = time.time() - starting_time
print(f'Time: {train_time}')

plt.plot(train, label = 'Trainloss')
plt.legend()
plt.show()



