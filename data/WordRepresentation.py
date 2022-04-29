from collections import Counter

from pyparsing import WordStart
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import requests
from io import StringIO
import math
import gensim.downloader as api
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec


class WordRepresentation:
    def __init__(self,dataset,method="word2vec",sentence_length=20,vector_length=50):
        if method not in ["word2vec","glove"]:
            print("Give 'word2vec' or 'glove' method")
            return None
        self.dataset=dataset
        self.glove={}
        self.word2vec={}
        self.sentence_length=sentence_length
        self.vector_length=vector_length
        self.method=method
        if method=="word2vec":
            self.compute_word2vec()
        else:
            self.compute_glove()
        self.final_data=[]
        self.represent_all_sentences()
    
    #Compute Glove
    def compute_glove(self):
      wv = api.load('glove-wiki-gigaword-50')
      #self.glove["nan"]=(np.zeros(50,),0)
      #word_index=1
      for i in range(len(self.dataset)):
        document=self.dataset.iloc[i,0]
        word_list=document.split(" ")
        for word in word_list:
          if word in wv.vocab:
            self.glove[word]=wv.word_vec(word)
            # self.glove[word]=(wv.word_vec(word),word_index)
            # word_index+=1
    
    @staticmethod
    def tokenize(data): 
      word = []
      for i in range(len(data)):
        new_doc = data[i].split()
        word.append(new_doc)
      return word
    
    def compute_word2vec(self):
      token = WordRepresentation.tokenize(self.dataset['content'])
      model = Word2Vec(sentences=token, workers = 1, size = 50, min_count = 1, window = 3)
      # self.word2vec["nan"]=(np.zeros(50,),0)d
      # word_index=1
      for i in range(len(self.dataset)):
        document=self.dataset.iloc[i,0]
        word_list=document.split(" ")
        for word in word_list:
          if word in model.wv.vocab:
            self.word2vec[word]=model.wv.word_vec(word)
            # self.word2vec[word]=(model.wv.word_vec(word),word_index)
            # word_index+=1
    
    def sentence_representation(self,sentence):
        the_sentence=sentence.split(' ')
        if self.method=="word2vec":
          words_dict=self.word2vec
        else:
          words_dict=self.glove
        
        matrix=np.zeros((self.sentence_length,self.vector_length))
        i =0
        for word in the_sentence:
          if word in words_dict:
            matrix[i]=words_dict[word]
            i+=1
          if i>=self.sentence_length:
            break
        return matrix
    
    # def represent_all_sentences(self):
    #   for i in range(len(self.dataset)):
    #     document=self.dataset.iloc[i,0]
    #     y=self.dataset.iloc[i,1]
    #     x_w2v=self.sentence_representation(document,sentence_length=sentence_length,vector_length=vector_length)
    #     x_glo=self.sentence_representation(document,method="glove",sentence_length=sentence_length,vector_length=vector_length)
    #     self.w2v_final_data.append((x_w2v,y))
    #     self.glo_final_data.append((x_glo,y))
        
    def represent_all_sentences(self):
      for i in range(len(self.dataset)):
        document=self.dataset.iloc[i,0]
        y=torch.tensor([self.dataset.iloc[i,1]])
        x_w2v=self.sentence_representation(document)
        y_w2v=torch.tensor(x_w2v.reshape(1,-1))
        y_w2v=y_w2v.float()
        self.final_data.append((y_w2v,y))