# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:05:23 2020
中文垃圾邮件分类
@author: 地三仙
"""
# 1.提取数据：get_data, remove_empty_docs, prepare_dataset
# 2.数据规整话和预处理 normalization中normalize_corpus
# 3.特征提取：feature_extractors中 bow_extractor, tfidf_extractor
# 4.训练分类器：train_predict_evaluate_model

# 疑问 垃圾邮件 重复率特别高  200条数据中只有 124条不重复数据
import numpy as np

def get_data():
    """
    获取数据
    :return:文本数据，对应的labels
    """
    # ham_data.txt:正常邮件 spam_data.txt: 垃圾邮件
    with open("./data/ham_data.txt",encoding='utf8') as ham_f, open("./data/spam_data.txt", 
encoding='utf8') as spam_f:
        ham_data = ham_f.readlines()        
        spam_data = spam_f.readlines()
        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()
        corpus = ham_data + spam_data
        labels = ham_label + spam_label
    return corpus, labels

def remove_empty_docs(corpus, labels):
    """
    :return 返回去掉非空的语料数据和对应的标签
    """
    removed_corpus = []
    reomved_labels = []
    for i in range(len(corpus)):
        if '' == corpus[i].strip():
            print("出现空的文章！")
            continue
        removed_corpus.append(corpus[i].strip())
        reomved_labels.append(labels[i])
    return removed_corpus, reomved_labels


def prepare_dataset(corpus, labels, test_portion=0.3):
    """
    构建模型训练和测试数据集,返回对应的语料和标签
    同一标签的语料集中有关系吗？ 如果有，下面的方法不可行
    """
    mid_num = int(0.5 * len(corpus))
    split_num = int(mid_num * (1 - test_portion))
    train_x = corpus[0: split_num] + corpus[mid_num: mid_num+split_num]
    train_y = labels[0: split_num] + labels[mid_num: mid_num+split_num]
    test_x = corpus[split_num: mid_num] + corpus[mid_num+split_num: len(corpus)]
    test_y = labels[split_num: mid_num] + labels[mid_num+split_num: len(corpus)]

    return train_x, test_x, train_y, test_y


def get_metrics(true_labels,predicted_labels):
    """
    定义二维数组装正负面评价预测正误的频率
    """
    metrics = np.zeros((2,2))
    for i in range(len(true_labels)):
        k = 1 if true_labels[i] == predicted_labels[i] else 0
        metrics[int(predicted_labels[i]), k] += 1
    # 计算评估指标
    # 准确率
    accuracy = (metrics[0, 1] + metrics[1, 1]) / metrics.sum()
    
    # 精准率
    precetion = metrics[1, 1] / (metrics[1, 1] + metrics[1, 0])
    # 召回率
    recall = metrics[1, 1] / (metrics[1, 1] + metrics[0, 0])
    # 得分统计
    f_measure = 2*precetion * recall / (precetion + recall)
    print("准确率：%f" % accuracy)
    print("精准率：%f" % precetion)
    print("召回率：%f" % recall)
    print("得分统计：%f" % f_measure)
    
    
    
def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels):
    # 建立模型
    classifier.fit(train_features, train_labels)
    # 用模型预测
    predictions = classifier.predict(test_features)
    # 这个函数是自定义的得到
    get_metrics(true_labels=test_labels,predicted_labels=predictions)
    return predictions


if __name__ == '__main__':
    corpus, labels = get_data()  # 获取原始数据集
    # 检测
    print("原始数据集条目：%d" % len(corpus))
    print(corpus[0] + '\n', labels[0])
    print(corpus[-1] + '\n', labels[-1])
    corpus, labels = remove_empty_docs(corpus, labels)
    # 数据集划分
    train_x, test_x, train_y, test_y = prepare_dataset(corpus, labels, test_portion=0.3)
    
    print(len(train_x))
    print(len(test_x))
    print(len(train_y))
    print(len(test_y))

    for i in range(60):
        if train_x[i] != corpus[i].strip():
            print(train_x[i])
#    dulcnt = 0
#    undulcnt = 0
#    dul_list = []
#    for i in range(len(corpus)):
#        for j in range(i+1, len(corpus),1):
#            if corpus[i] == corpus[j]:
#                print(i,j)
#                print(corpus[i][ :10])
#                dulcnt += 1
#                dul_list += [i, j]
#                break
#            if j >= len(corpus) - 1:
#                undulcnt += 1

# 进行规整化
from normalization import normalize_corpus
norm_train_corpus =  normalize_corpus(train_x)
norm_test_corpus =  normalize_corpus(test_x)


# 特征提取

from feature_extractors import bow_extractor, tfidf_extractor
import gensim
import jieba

# 词袋模型特征           
bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
bow_test_features = bow_vectorizer.transform(norm_test_corpus)

# tfidf特征
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

# tokenize docments
tokenized_train = [jieba.lcut(text) for text in norm_train_corpus]
tokenized_test = [jieba.lcut(text) for text in norm_test_corpus]

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', n_iter_no_change=100) # 参数 n_iter为旧版本参数名
lr = LogisticRegression()

# 基于词袋模型的多项式朴素贝叶斯
print("基于词袋模型的多项式朴素贝叶斯分类器")
mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                                   train_features=bow_train_features,
                                                   train_labels=train_y, 
                                                   test_features=bow_test_features, 
                                                   test_labels=test_y)

# 基于词袋模型的支持向量机
print("基于词袋模型的支持向量机")
svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                                   train_features=bow_train_features,
                                                   train_labels=train_y, 
                                                   test_features=bow_test_features, 
                                                   test_labels=test_y)

# 基于词袋模型的逻辑回归
print("基于词袋模型的逻辑回归")
lr_bow_predictions = train_predict_evaluate_model(classifier=lr,
                                                   train_features=bow_train_features,
                                                   train_labels=train_y, 
                                                   test_features=bow_test_features, 
                                                   test_labels=test_y)

# 基于tfidf的多项式朴素贝叶斯
print("基于tfidf的多项式朴素贝叶斯分类器")
mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb,
                                                   train_features=tfidf_train_features,
                                                   train_labels=train_y, 
                                                   test_features=tfidf_test_features, 
                                                   test_labels=test_y)

# 基于tfidf的支持向量机
print("基于tfidf的支持向量机")
svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm,
                                                   train_features=tfidf_train_features,
                                                   train_labels=train_y, 
                                                   test_features=tfidf_test_features, 
                                                   test_labels=test_y)
# 基于tfidf的逻辑回归
print("基于tfidf的逻辑回归")
lr_tfidf_predictions = train_predict_evaluate_model(classifier=lr,
                                                   train_features=tfidf_train_features,
                                                   train_labels=train_y, 
                                                   test_features=tfidf_test_features, 
                                                   test_labels=test_y)


