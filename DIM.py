#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:57:01 2018

@author: yichen
"""

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

data = pd.read_csv('tc_record_log.csv')
data = data.loc[data['Volume']>=96]
data = data[['ArticleID','Date_Published_in_Print']]
data = data.drop_duplicates().dropna()


publish_year = []
for i in data['Date_Published_in_Print']:
    publish_year.append(i[:4])
data['Date_Published_in_Print'] = publish_year

files = os.listdir('./Data')
files_39_htm = []
for i in files:
    if (('39_' in i) or ('38_' in i)) and ('.htm' in i) and ('_g' not in i):
        files_39_htm.append(i)


articleid = list(data['ArticleID'])

final_files = []
final_articleid = []
problem = []
for i in files_39_htm:
    try:    
        this = int(i[3:-4])
        if this in articleid:
            final_files.append(i)
            final_articleid.append(this)
    except:
        problem.append(i)

no_htm = list(set(articleid) - set(final_articleid))


documents = []
indexs = []
for index,file in enumerate(final_files):
    try:
        f=open("./Data/"+file,'r')
        document= f.read()
        documents.append(document)
        indexs.append(final_articleid[index])
    except:
        pass


documents_final = [0] * data.shape[0]
for i,j in enumerate(indexs):
    where = list(data['ArticleID']).index(j)
    documents_final[where] = documents[i]

data['document'] = documents_final
data = data.loc[data['document']!=0]
data = data.drop_duplicates().dropna()

data = data.reindex()

references = []
for i in data['document']:
    reference = re.findall(r'<P style=.*>.*\(.*\).* <I>Teachers College Record',i)
    references.append(reference)
                    

for i,j in enumerate(references):
    if len(j) != 0:
        this = []
        for k in j:
            index_begin = k.index(')')
            index_end = k.index('<I>')  
            potential = k[index_begin+1:index_end].replace('.','').strip()
            translator=str.maketrans('','',string.punctuation)
            potential = potential.translate(translator).lower()            
            if len(potential) != 0 :
                this.append(potential)
            else:
                this.append(k)
        references[i] = this

data_1 = pd.read_csv('tc_record_log.csv')
title = list(data_1['Title'])
title = [str(i) for i in title]

Articleid = list(data_1['ArticleID'])

for i,a in enumerate(references):
    if len(a) != 0:
        for j,b in enumerate(a):
            references[i][j] = Articleid[title.index(difflib.get_close_matches(b, title,1,0)[0])]
                
final_references = []            
for i in references:
    final_references += i
    
dic = Counter(final_references)

reference = [0] * data.shape[0]
for key,value in dic.items():
    try:
        where =  list(data['ArticleID']).index(key)
        reference[where] = value
    except:
        pass
data['reference'] = reference

data.to_csv('data_reference_doc.csv',index=False)

dic_time = Counter(data['Date_Published_in_Print'])

time = list(data['Date_Published_in_Print'])
time_step = []

dic_time = Counter(time_step)

for i in time:
    if int(i) >= 1994 and int(i) <= 1995:
        time_step.append(1)
    elif int(i) >= 1999 and int(i) <= 2003:
        time_step.append(2)
    elif int(i) >= 2004 and int(i) <= 2008:
        time_step.append(3)
    elif int(i) >= 2009 and int(i) <= 2013:
        time_step.append(4)
    else:
        time_step.append(5)

data['time_step'] = time_step
data = data.sort_values(by=['Date_Published_in_Print'])

data.to_csv('data_reference_doc_DIM.csv',index=False)

time_seq = [276, 535, 864, 482, 545]

dtm_path = '/Users/yichen/dtm-master/dtm/'

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

data['wrong'] = wrong_index
data = data[data['wrong']==0]

data['document'] = document


time_seq = [276, 534, 864, 482, 544]

class DTMcorpus(corpora.textcorpus.TextCorpus):

    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)

corpus = DTMcorpus(documents)
model = DtmModel(dtm_path, corpus, time_seq, num_topics=2,id2word=corpus.dictionary, initialize_lda=True, model='fixed')


## document topic propotion

np.save('gamma',model.gamma_)
np.save('influence',model.influences_time)
np.save('model_result',model)





