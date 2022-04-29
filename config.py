import torch.nn as nn


vector_lenght=100
sentence_lenght=20
input_dim=vector_lenght*sentence_lenght
hidden_dim=100
num_classes=5
criterion = nn.CrossEntropyLoss()
num_epochs=5
weight_decay=1e-4
lr=1e-4