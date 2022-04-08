import torch
import torch.nn.functional as F
from torch import nn
# from torchsummary import summary

class Classify(nn.Module):
    def __init__(self, input):
        super(Classify, self).__init__()
        self.input_layer = nn.Linear(input, 400)  
        self.hidden1 = nn.Linear(400, 200) 
        self.hidden2 = nn.Linear(200, 100) 
        self.hidden3 = nn.Linear(100, 75)
        self.hidden4 = nn.Linear(75, 50)
        self.output = nn.Linear(50, 10) 

    # Batch normalisation n Dropout    
        # self.fc1 = nn.Linear(input, 400)
        # self.bn1 = nn.BatchNorm1d(400) # Batcn Norm for the first layer
        # self.fc2 = nn.Linear(400, 200)
        # self.bn2 = nn.BatchNorm1d(200) # Batcn Norm for the secon layer
        # self.fc3 = nn.Linear(200, 100)
        # self.bn3 = nn.BatchNorm1d(100)  # Batcn Norm for the third layer
        # self.fc4 = nn.Linear(100, 75)
        # self.bn4 = nn.BatchNorm1d(75)
        # self.fc5 = nn.Linear(75, 50)
        # self.bn5 = nn.BatchNorm1d(50)
        # self.output = nn.Linear(50, 10)
        # self.do  = nn.Dropout(0.2, inplace=True)

    
    def forward(self, x):
        first_layer = self.input_layer(x)
        act1 = F.relu(first_layer)
        second_layer = self.hidden1(act1)
        act2 = F.relu(second_layer)
        third_layer = self.hidden2(act2)
        act3 = F.relu(third_layer)
        fourth_layer = self.hidden3(act3)
        act4 = F.relu(fourth_layer)
        third_layer = self.hidden4(act4)
        act5 = F.relu(third_layer)
        out_layer = self.output(act5)
        x = F.softmax(out_layer, dim=1)
        return out_layer

        # x = F.relu(self.bn1(self.do(self.fc1(x))))
        # x = F.relu(self.bn2(self.do(self.fc2(x))))
        # x = F.relu(self.bn3(self.do(self.fc3(x))))
        # x = F.relu(self.bn4(self.do(self.fc4(x))))
        # x = F.relu(self.bn5(self.do(self.fc5(x))))
        # x = F.softmax(x, dim=1)
        # return self.output(x)

