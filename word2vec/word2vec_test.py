"""
测试word2vec词向量
"""
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def test():
    print('加载词向量...')
    # 方式一 加载模型文件
    # weibo_word2vec_model = Word2Vec.load('./weibo_zh_word2vec_100.model')
    # wv = weibo_word2vec_model.wv
    # 方式二 加载词向量文件
    wv = KeyedVectors.load_word2vec_format('./weibo_zh_word2vec_format_100.txt', binary=False)
    # 获取最相似的词向量
    similar = wv.most_similar('孩子', topn=5)
    print(similar)
    # 计算相似度
    similarity = wv.similarity('小孩', '父母')
    print(similarity)
    # 获取某个词的词向量
    vector = wv.get_vector('数学家')
    print(vector)
    # 获取两个词的距离 所谓两个词的距离就是1 - self.similarity(w1, w2)
    distance = wv.distance('微博', '讨论')
    print(distance)


if __name__ == '__main__':
    test()
