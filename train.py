from config import *
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