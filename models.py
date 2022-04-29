import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class FcNeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FcNeuralNet, self).__init__()
        self.fc1= nn.Linear(input_dim, hidden_dim)
        self.fc2= nn.Linear(hidden_dim, hidden_dim*2)
        # self.fc3= nn.Linear(hidden_dim*2, hidden_dim)
        self.fc4= nn.Linear(hidden_dim*2, num_classes)

    
    def forward(self, x):
      """
      The forward pass of the fully connected layer
      """
      x = F.tanh(self.fc1(x))
      x=self.fc2(x)
      # x=F.tanh(self.fc3(x))
      x=self.fc4(x)

      out=F.log_softmax(x, dim =1)
      return out


def train(model,training_data,optimizer,criterion,num_epochs=10):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i in training_data:
            sentences, labels = i

            # Move tensors to the configured device
            # sentences = sentences.to(device)
            # labels = labels.to(device)
                
            # Forward pass
            outputs = model(sentences)
            loss = criterion(outputs, labels-1)    
            # Backprpagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item() 
                    
        loss_values = (running_loss / len(training_data))
        losses.append(loss_values)
            
        print("=" * 25)
        print('| Epoch {:3d}| Loss: {:.4f}'.format(epoch+1,loss.item()))
    return model,losses

def eval(model,test_data):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i in test_data:
            sentences, labels = i
            outputs = model(sentences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted+1 == labels).sum().item()
            print(predicted,labels)
        
    print("="* 33)
    print('Accuracy test: {:.2f} %'.format(100 * correct / total))
    print("="* 33)
    return correct