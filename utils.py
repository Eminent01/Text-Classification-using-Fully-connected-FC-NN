import torch

import torch
import matplotlib.pyplot as plt


def get_batch(data,index,batch_size):
  size=batch_size
  length=len(data)
  input_length=len(data[0][0])
  if index+batch_size>length:
    size=length-index
  sentences=[]
  labels=torch.zeros((size,1))
  batch_data=data[index:index+size]
  for i in range(len(batch_data)):

    sentence,label=batch_data[i]
    #print("s",label.shape)
    sentences.append(sentence)
    labels[i]=label
  
  return torch.stack(sentences),labels



def plot(cost,method):
    plt.title(f'{method} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(cost, "r",  label= f'{method} Loss')
    plt.legend()
    plt.savefig('./figures/fig.png')
    plt.show()

def get_max_sentence_length(dataset):
    max_sentence_length=0
    for i in range(len(dataset)):
        doc=dataset.iloc[i,0]
        list_words=doc.split(" ")
        # print(len(list_words))
        number_of_words=len(list_words)
        if number_of_words>max_sentence_length:
            max_sentence_length=number_of_words
    return max_sentence_length