"""
编写代码，根据weibo_senti_100k.csv数据集生成训练词向量的语料。
"""
import pandas as pd
import os
import jieba


def get_stopwords():
    stopwords = []
    with open('./hit_stopwords.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def generate():
    # 读取原始训练数据
    readers = pd.read_csv(os.path.join('../data', 'weibo_senti_100k.csv'), chunksize=1000, delimiter=',')
    reviews = []
    for reader in readers:
        # 使用extend方法逐个追加到reviews中
        reviews.extend(list(reader.review.values))
    with open('./weibo_source_corpus_zh.txt', 'w', encoding='utf-8') as f:
        for review in reviews:
            f.write(review + '\n')
    print('weibo_source_corpus_zh.txt创建成功')

    # 分词处理
    jieba.load_userdict('./hownet_zh.txt')
    # 加载停用词
    stopwords = get_stopwords()
    with open('./weibo_train_corpus_zh.txt', 'w', encoding='utf-8') as f:
        words = []
        for review in reviews:
            sentence = jieba.lcut(review, cut_all=False)
            for word in sentence:
                if word != ' ' and word not in stopwords:
                    words.append(word)
        f.write(' '.join(words))
    print('weibo_train_corpus_zh.txt创建成功')

if __name__ == '__main__':
    generate()
