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
import data_handler as dh
from model import Classify


trainset, testset, trainloader, testloader = dh.get_data()

model = Classify(784)

criterion =  nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 100
train = []
print_every = 40
starting_time = time.time()

for epoch in range(epochs):
    running_loss = 0
    running_loss_test = 0

    if epoch%10 == 0:
        torch.save(model.state_dict(), f'checkpoint_{epoch}.pth')

    print(f'Epoch: {epoch+1}/{epochs}')
    
    for i, (images, labels) in enumerate(iter(trainloader)):

        images.resize_(images.size()[0], 784)

        optimizer.zero_grad()

        logits = model.forward(images)

        prediction = F.log_softmax(logits, dim=1)#model.forward(images)

        loss = criterion(prediction, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()  


        if i%print_every ==0:
                print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
                running_loss=0

    train.append((running_loss/64))
# train_time = time.time() - starting_time
# print(f'Time: {train_time}')

torch.save(model.state_dict(), 'checkpoint.pth')

plt.plot(train, label = 'Testloss')
plt.plot()
plt.legend()
plt.show()

