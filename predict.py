import torch
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
            #print(predicted,labels)
        
    print("="* 33)
    print('Accuracy : {:.2f} %'.format(100 * correct / total))
    print("="* 33)
    return correct