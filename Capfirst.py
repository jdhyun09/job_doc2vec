# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 03:30:11 2021

@author: 410-2
"""

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from konlpy.tag import Mecab


stopwords=['의','가','이','은','들','는','좀','잘',
           '걍','과','도','를','을','으로','자','에','와',
           '한','하다','-','?','(',')','[',']','<','>','/','.',',',
           '수','있','도록','하','로','됩니다','습니다','에서','왔','*']

tokenizer = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")


class Doc2VecCorpus:
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        df = pd.read_csv(self.fname)
        for i in df.index:
            #print(df.at[i,'company'])
            company_idx, text = df.at[i,'company'], df.at[i, 'info']
            temp = tokenizer.morphs(text)
            temp = [word for word in temp if not word in stopwords]
            yield TaggedDocument(
                temp, 
                tags = ['회사_%s' % company_idx])
            
doc2vec_corpus = Doc2VecCorpus('jobs_oldreal.csv')
doc2vec_model = Doc2Vec(doc2vec_corpus, dm = 1)#dm은 어떤 doc2vec모델을 쓸지

print(doc2vec_model.docvecs.most_similar('회사_매드업(주)',topn=10))


for idx, doctag in sorted(doc2vec_model.docvecs.doctags.items(), key=lambda x:x[1].offset):
    print(idx, doctag)
