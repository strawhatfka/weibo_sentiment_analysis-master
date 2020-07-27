"""
使用word2vec工具训练词向量
"""
import logging
import os
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def train():
    # 日志信息输出
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))
    dimensions = [100, 200, 300]
    # 训练模型 生成不同维度的词向量
    for dimension in dimensions:
        input_file = './weibo_train_corpus_zh.txt'
        # output1 = './weibo_zh_word2vec_' + str(dimension) + '.model'
        output2 = './weibo_zh_word2vec_format_' + str(dimension) + '.txt'
        '''
        cbow_mean=1 使用CBOW模型
        negative=5 使用negative sampling
        hs=0 不使用hierarchical softmax
        '''
        model = Word2Vec(LineSentence(input_file), size=dimension, window=5, min_count=2
                         , workers=multiprocessing.cpu_count(), iter=15)
        # model.save(output1)
        model.wv.save_word2vec_format(output2, binary=False)


if __name__ == '__main__':
    train()
