import os

import jieba
import pandas as pd
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import classification_report
import joblib

path = './data'
output_path = './gnb_weibo_output/'
if not os.path.exists(output_path):
    os.mkdir(output_path)


def get_stopwords():
    stopwords = []
    with open('./word2vec/hit_stopwords.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def get_sentences(texts, stopwords):
    # 分词处理
    sentences = []
    for text in texts:
        sentence = jieba.lcut(text, cut_all=False)
        words = []
        for word in sentence:
            if word != ' ' and word not in stopwords:
                words.append(word)
        sentences.append(' '.join(words))
    return sentences


if __name__ == '__main__':
    # 超参
    C = 0.5
    max_iter = 100
    # 数据预处理
    print('读取数据集')
    train_data = pd.read_csv(os.path.join(path, './train.csv'))
    dev_data = pd.read_csv(os.path.join(path, './dev.csv'))
    test_data = pd.read_csv(os.path.join(path, './test.csv'))
    # 加载自定义词典
    jieba.load_userdict('./word2vec/hownet_zh.txt')
    # 获取停用词表
    stopwords = get_stopwords()
    x_train, y_train = get_sentences(train_data.review.values, stopwords), train_data.label.values
    x_dev, y_dev = get_sentences(dev_data.review.values, stopwords), dev_data.label.values
    x_test, y_test = get_sentences(test_data.review.values, stopwords), test_data.label.values
    # 对分词后的文本进行特征值提取，生成对应的稀疏矩阵 其实也是一种 tokenizer
    # vectorizer = CountVectorizer()
    # FLAG='countvectorizer'
    # vectorizer = TfidfVectorizer()
    # FLAG = 'tfidfvectorizer'
    vectorizer = HashingVectorizer()
    FLAG = 'hashingvectorizer'
    '''
    fit_transform与transform的区别 
    fit_transform用于生成词典或者叫特征(vocabulary size 即feature size)以及得到稀疏矩阵;
    transform根据fit_transform生成的vocabulary得到对应的特征矩阵
    '''
    x_train = vectorizer.fit_transform(x_train).todense()
    x_dev = vectorizer.transform(x_dev).todense()
    x_test = vectorizer.transform(x_test).todense()
    # 初始化网络模型
    print('初始化网络模型')
    model = GaussianNB()
    # 拟合模型
    print('拟合模型...')
    model.fit(x_train, y_train)

    # 评估以及预测
    score = model.score(x_dev, y_dev)
    test_predict = model.predict(x_test)
    test_report = classification_report(y_test, test_predict, digits=4)
    result = 'dev_evaluate\n' + str(score) + '\ntest_predict\n' + str(test_report)
    print(result)
    with open(output_path + 'train_lr_' + FLAG + '_result.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    # 保存模型
    joblib.dump(model, output_path + 'weibo_lr_' + FLAG + '_model.pkl')
