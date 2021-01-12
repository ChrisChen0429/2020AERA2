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
time_seq = [67+52,73+50,51+51,79+150,204+257,189+235,91+92,94+102,95+96,94+103,100+116,122+103]
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
with open('result_twoyearatime.pkl','wb') as output:
    for t in range(15):
        print('finish the round',str(t+1))
        result = DtmModel(dtm_path, corpus, time_seq, num_topics=(t+2),id2word=corpus.dictionary, initialize_lda=True, model='fixed',top_chain_var=0.05)
        pickle.dump(result,output,pickle.HIGHEST_PROTOCOL)
        results.append(result)

results = []
with (open("result_twoyearatime.pkl", "rb")) as openfile:
    while True:
        try:
            results.append(pickle.load(openfile))
        except EOFError:
            break


import scipy
k=8
final_model = results[k-2]
influence_factor = []
overall_influence = []
time_seq = [67+52,73+50,51+51,79+150,204+257,189+235,91+92,94+102,95+96,94+103,100+116,122+103]
time_seq = [0] + time_seq

time_propotion = []
for i in range(len(time_seq)-1):
    number = time_seq[i+1]
    propotion = np.array([0]*k)
    for j in range(number):
        propotion = propotion + final_model.gamma_[time_seq[i] + (j+1)]
    this = propotion / propotion.sum()
    time_propotion.append(this)
for i in range(len(time_seq)-2):
    number = time_seq[i+1]
    for j in range(number):
        a = final_model.influences_time[i][j]
        b = final_model.gamma_[time_seq[i]+(j+1)]
        c = time_propotion[i]
        d = a*b*c
        d = np.amax(d)
        influence = np.mean(d)
        # b = [0] * k
        # influence = scipy.spatial.distance.euclidean(a,b)
        influence_factor.append(list(a))
        overall_influence.append(influence)
# influence_factor = influence_factor[:sum([276,534, 864,482])]
# overall_influence = overall_influence[:sum([276,534, 864, 482])]

a = []
b = []
c = []
d = []
e = []
f = []
g = []
h = []
for i in influence_factor:
    a.append(i[0])
    b.append(i[1])
    c.append(i[2])
    d.append(i[3])
    e.append(i[4])
    f.append(i[5])
    g.append(i[6])
    h.append(i[7])
        
time_stamp = ['1995-1996'] * (67+52) + ['1997-1998'] * (73+50) + ['1999-2000'] * (51+51) + ['2001-2002'] * (79+150) + ['2003-2004'] * (204+257) + ['2005-2006'] * (189+235) + ['2007-2008'] * (91+92) + ['2009-2010'] * (94+102) + ['2011-2012'] * (95+96) + ['2013-2014'] * (94+103) + ['2015-2016'] * (100+116)

reference = list(data['reference'])[:sum([67+52,73+50,51+51,79+150,204+257,189+235,91+92,94+102,95+96,94+103,100+116])]
time = list(data['Date_Published_in_Print'])[:sum([67+52,73+50,51+51,79+150,204+257,189+235,91+92,94+102,95+96,94+103,100+116])]
ID = list(data['ArticleID'])[:sum([67+52,73+50,51+51,79+150,204+257,189+235,91+92,94+102,95+96,94+103,100+116])]
time_stamp = time_stamp[:sum([67+52,73+50,51+51,79+150,204+257,189+235,91+92,94+102,95+96,94+103,100+116])]


regression_data = pd.DataFrame({'time':time,
                                'documentid':ID,
                                'reference':reference,
                                'time_stamp':time_stamp,
                                'overall_influence':overall_influence,
                                'influence_factor':influence_factor,
                                 'a':a,'b':b,'c':c,'d':d,'e':e,'f':f,
                                 'g':g,'h':h})
regression_data.to_csv('regression_data_twoyear.csv')

