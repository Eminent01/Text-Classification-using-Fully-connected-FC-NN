import torch.nn as nn


vector_lenght=50
sentence_lenght=20
input_dim=vector_lenght*sentence_lenght
hidden_dim=10
num_classes=5
criterion = nn.CrossEntropyLoss()
num_epochs=5
