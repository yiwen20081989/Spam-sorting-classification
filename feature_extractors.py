# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:19:40 2020
特征提取模块：词袋模型特征 和tfidf特征
@author: 地三仙
"""
from sklearn.feature_extraction.text import CountVectorizer

def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range) # vectorizer.vocabulary_ 词典
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
    
    
from sklearn.feature_extraction.text import TfidfVectorizer # 还可以用 TfidfTranformer在第一步基础上间接得到

def tfidf_extractor(corpus, ngram_range=(1, 1)): # norm='l2' 是l不是1什么意思  果然报错
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True,
                                 use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

    
    
