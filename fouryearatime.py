import pandas as pd
import numpy as np
import os
import re
import  difflib
from collections import Counter
import logging
import os
from gensim import corpora, utils
from gensim.models.wrappers.dtmmodel import DtmModel
import numpy as np
import scipy
import string

data = pd.read_csv('data_reference_doc.csv')
data = data.drop(data.index[[1921,2130]])
data = data.sort_values(by=['Date_Published_in_Print'])
data = data.loc[data['Date_Published_in_Print'] >1994]
dic_time = Counter(data['Date_Published_in_Print'])
time = list(data['Date_Published_in_Print'])
data = data.sort_values(by=['Date_Published_in_Print'])
time_seq = [67+52+73+50,51+51+79+150,204+257+189+235,91+92+94+102,95+96+94+103,100+116+122+103]
dtm_path = '/Users/yichen/dtm-master/dtm/main'
document = pd.read_csv('document.csv')
del document['Unnamed: 0']
document = document.iloc[0]        
documents = []
wrong_index = []
for i in range(2702):
    try:
        this_doc = document[i].split(" ")
        this_doc = [x for x in this_doc if x != '']
        documents.append(this_doc)
        wrong_index.append(0)
    except:
        wrong_index.append(1)
class DTMcorpus(corpora.textcorpus.TextCorpus):
    def get_texts(self):
        return self.input
    def __len__(self):
        return len(self.input)
documents = documents[34:]
data['document'] = documents
corpus = DTMcorpus(documents)
import pickle
print('stat!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
results = []
with open('result_fouryearatime.pkl','wb') as output:
    for t in range(15):
        print('finish the round',str(t+1))
        result = DtmModel(dtm_path, corpus, time_seq, num_topics=(t+2),id2word=corpus.dictionary, initialize_lda=True, model='fixed',top_chain_var=0.05)
        pickle.dump(result,output,pickle.HIGHEST_PROTOCOL)
        results.append(result)