from collections import Counter

from pyparsing import WordStart
import numpy as np

import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


class PreprocessData:
    def __init__(self,path,lang='english'): 
        self.dataset=pd.read_csv(path, sep=",")
        self.dataset=self.dataset[["content","score"]]
        self.stopwords=stopwords.words(lang)
        self.preprocess()
    
    @staticmethod
    def remove_punctuation(text):
        '''a function for removing punctuation'''
        # replacing the punctuations with no space, 
        # which in effect deletes the punctuation marks 
        translator = str.maketrans('', '', string.punctuation)
        # return the text stripped of punctuation marks
        return text.translate(translator)

    #A function to remove the stopwords
    def remove_stopwords(self,text):
        text = [word.lower() for word in text.split() if word.lower() not in self.stopwords]
        # joining the list of words with space separator
        return " ".join(text)

    def preprocess(self):
        self.dataset.iloc[:,0] = self.dataset.iloc[:,0].apply(self.remove_punctuation)
        self.dataset.iloc[:,0] = self.dataset.iloc[:,0].apply(self.remove_stopwords)