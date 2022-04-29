import torch
import torch.nn as nn
import torch.nn.functional as F


class FcNeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, p = 0.5):
        super(FcNeuralNet, self).__init__()
        self.fc1= nn.Linear(input_dim, hidden_dim)
        # self.fc2= nn.Linear(hidden_dim, hidden_dim*2)
        #self.fc3= nn.Linear(hidden_dim*2, hidden_dim)
        self.fc4= nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p) 

    
    def forward(self, x):
      """
      The forward pass of the fully connected layer
      """
      x = F.relu(self.fc1(x))
    #   x=self.fc2(x)
      #x=F.sigmoid(self.fc3(x))
      x=self.fc4(x)
      x=self.dropout(x)

      out=F.log_softmax(x, dim =1)
      return out



