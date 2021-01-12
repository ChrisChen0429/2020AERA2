import pandas as pd
from gensim import corpora, utils
from gensim.models.wrappers.dtmmodel import DtmModel
dtm_path = '/Users/yichen/dtm-master/dtm/main'
document = pd.read_csv('document.csv')
data = pd.read_csv('data_reference_doc_DIM.csv')

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
time_seq = [276,534, 864, 482, 544]
class DTMcorpus(corpora.textcorpus.TextCorpus):
    def get_texts(self):
        return self.input
    def __len__(self):
        return len(self.input)
corpus = DTMcorpus(documents)

import pickle
print('stat!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
topic_number = [18,20,22,24,26,28,30]
with open('result_more.pkl','wb') as output:
    for t in topic_number:
        print('finish the round',str(t+1))
        result = DtmModel(dtm_path, corpus, time_seq, num_topics=t,id2word=corpus.dictionary, initialize_lda=True, model='fixed',top_chain_var=0.05)
        pickle.dump(result,output,pickle.HIGHEST_PROTOCOL)
