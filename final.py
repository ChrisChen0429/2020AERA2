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
results = []
with open('result_new.pkl','wb') as output:
    for t in range(15):
        print('finish the round',str(t+1))
        result = DtmModel(dtm_path, corpus, time_seq, num_topics=(t+2),id2word=corpus.dictionary, initialize_lda=True, model='fixed',top_chain_var=0.05)
        pickle.dump(result,output,pickle.HIGHEST_PROTOCOL)
        results.append(result)

results = []
with (open("result_new.pkl", "rb")) as openfile:
    while True:
        try:
            results.append(pickle.load(openfile))
        except EOFError:
            break







likelihoods = []
for result in results:
    this = result.fout_liklihoods()
    a = []
    for line in open(this,'r'):
        itme = float(line.rstrip())
        a.append(itme)
    a = sum(a) / len(a)
    likelihoods.append(a)

import scipy
k=14
final_model = results[k-2]
influence_factor = []
overall_influence = []
time_seq = [276,534, 864, 482, 544]
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
        influence = sum(list(d))
        # b = [0] * k
        # influence = scipy.spatial.distance.euclidean(a,b)
        influence_factor.append(list(a))
        overall_influence.append(influence)
# influence_factor = influence_factor[:sum([276,534, 864,482])]
# overall_influence = overall_influence[:sum([276,534, 864, 482])]
reference = list(data['reference'])[:sum([276,534, 864, 482])]
time = list(data['time_step'])[:sum([276,534, 864, 482])]
time[-1] = 4

regression_data = pd.DataFrame({'time':time,
                                'reference':reference,
                                'overall_influence':overall_influence,
                                'influence_factor':influence_factor})
                                # 'a':a,'b':b,'c':c,'d':d,'e':e,'f':f,
                                # 'g':g,'h':h,'j':j})
regression_data.to_csv('regression_data.csv')

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
    j.append(i[8])





# time_influence = []
# for i in range(4):
#     time_influence.append(list(regression_data.loc[regression_data['time']==i+1]['overall_influence']))

# influence = []
# from sklearn import preprocessing
# import numpy as np 
# for j in time_influence:
#     j = np.array(j)
#     j_scaled = list(preprocessing.scale(j))
#     influence += j_scaled

# regression_data = pd.DataFrame({'time':time,
#                                 'reference':reference,
#                                 'influence':influence,
#                                 'influence_factor':influence_factor})