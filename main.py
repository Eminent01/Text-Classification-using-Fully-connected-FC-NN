from cgi import test
import sys
import torch
import torch.nn as nn

from config import *
from models import *
from utils import plot,get_max_sentence_length
from predict import eval
from train import train
from data.PreprocessData import PreprocessData as Preprocess
from data.WordRepresentation import WordRepresentation
from sklearn.model_selection import train_test_split


#The main will be called using name.py



path="./data/reviews.csv"
dataset=Preprocess(path)


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-w" ,"--word_rep", help = "Word  representtion metho", required = True)
parser.add_argument("-n" ,"--num_epochs", help = "Number of epochs", type = int,required = True)
parser.add_argument("-max" ,"--take_max", help = "Take sentence length as the maximum length", action='store_true')
parser.add_argument("-sml" ,"--set_max", help = "Set max length", type = int)
mains_args = vars(parser.parse_args())



method= mains_args["word_rep"].lower()
if method in ["word2vec","glove"]:

    consider_all=mains_args["take_max"]
    if consider_all:
        sentence_lenght=get_max_sentence_length(dataset.dataset)
        input_dim=vector_lenght*sentence_lenght
    else:
        given_max=mains_args["set_max"]
        if given_max:
            sentence_lenght=given_max
            input_dim=vector_lenght*sentence_lenght
 
    num_epochs=mains_args["num_epochs"]
    new_data=WordRepresentation(dataset.dataset,method,sentence_lenght,vector_lenght)
    print("-----------------Word Representation Good-----------------")
    model=FcNeuralNet(input_dim=input_dim,hidden_dim=hidden_dim,num_classes=num_classes)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay= weight_decay)
    print("-----------------Creation model Good-----------------",model)
    train_data, test_data = train_test_split(new_data.final_data, test_size = 0.2, random_state = 42, shuffle = True)

    print("-----------------Split data Good-----------------")
    model,losses=train(model,train_data,optimizer,criterion,num_epochs=num_epochs)
    print("-----------------Training Good-----------------")
    print("-----------------Evaluation of Training-----------------")
    corrects=eval(model,train_data)
    print("-----------------Evaluation of Testing-----------------")
    corrects=eval(model,test_data)
    print("-----------------Evaluation Good-----------------",corrects)
    plot(losses,method)
else:
    print("Not Good")
