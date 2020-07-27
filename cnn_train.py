"""
使用word2vec和cnn做微博评论情感分析
"""
import os
import pandas as pd
import jieba
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import classification_report
os.environ['CUDA_VISIBLE_DEVICES']='/gpu:1'
keras = tf.keras
preprocessing = keras.preprocessing
path = './data'
output_path = './cnn_weibo_output/'
if not os.path.exists(output_path):
    os.mkdir(output_path)


def get_stopwords():
    stopwords = []
    with open('./word2vec/hit_stopwords.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def get_segment_words(texts, stopwords):
    # 分词处理
    sentences = []
    for text in texts:
        sentence = jieba.lcut(text, cut_all=False)
        words = []
        for word in sentence:
            if word != ' ' and word not in stopwords:
                words.append(word)
        sentences.append(words)
    return sentences


# F1-score评价指标
def metric_F1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = 2 * precision * recall / (precision + recall)
    return F1score


#CNN-dropout-BN-全连接-dropout-全连接
def CNN_model(max_len,vocab_size, embedding_size,embedding_matrix):
    layers = keras.layers
    model = keras.Sequential()
    #model.add(layers.Embedding(vocab_size, embedding_size, input_length=max_len))  # 随机初始化词向量
    model.add(layers.Embedding(vocab_size, embedding_size, input_length=max_len,weights=[embedding_matrix]))  # 使用预训练词向量进行初始化
    model.add(layers.Conv1D(256, 5, padding='same'))
    model.add(layers.MaxPooling1D(3, 3, padding='same'))
    model.add(layers.Conv1D(128, 5, padding='same'))
    model.add(layers.MaxPooling1D(3, 3, padding='same'))
    model.add(layers.Conv1D(64, 3, padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.1))
    model.add(layers.BatchNormalization())  # (批)规范化层
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(2, activation='softmax'))
    return model                                                        


if __name__ == '__main__':
    # 超参
    max_len = 128
    # glove/word2vec
    word_vector_type='word2vec'
    embedding_size = 100
    batch_size = 32
    epochs=5
    # 数据预处理
    print('读取数据集')
    train_data = pd.read_csv(os.path.join(path, './train.csv'))
    dev_data = pd.read_csv(os.path.join(path, './dev.csv'))
    test_data = pd.read_csv(os.path.join(path, './test.csv'))
    # 加载自定义词典
    jieba.load_userdict('./word2vec/hownet_zh.txt')
    # 获取停用词表
    stopwords = get_stopwords()
    x_train, y_train = get_segment_words(train_data.review.values, stopwords), train_data.label.values
    x_dev, y_dev = get_segment_words(dev_data.review.values, stopwords), dev_data.label.values
    x_test, y_test = get_segment_words(test_data.review.values, stopwords), test_data.label.values
    sentences = []
    sentences.extend(x_train)
    sentences.extend(x_dev)
    sentences.extend(x_test)
    # 根据分词结果 生词vocab 统计词频以及对词进行编号 词频越大，编号越小
    print('根据分词样本生成vocab')
    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    vocab = tokenizer.word_index
    # 根据vocab，将数据集中的每个样本中的分词序列转化为编号序列
    print('分词序列编号化')
    x_train = tokenizer.texts_to_sequences(x_train)
    x_dev = tokenizer.texts_to_sequences(x_dev)
    x_test = tokenizer.texts_to_sequences(x_test)
    # 每条样本长度不一，将每条样本的长度设置为一个固定值 将超过固定值的部分截掉，不足的在最前面用0填充
    print('padding sequence')
    # 知道样本中的所有词在vocab中的位置信息，以及位置所对应的词向量矩阵，就可以实现Embedding
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_dev = preprocessing.sequence.pad_sequences(x_dev, maxlen=max_len)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    #加载词向量
    print('加载词向量...')
    #wv=KeyedVectors.load_word2vec_format('/home/ydwang/word_vector/news_12g_baidubaike_20g_novel_90g_embedding_64.bin',binary=True)
    if word_vector_type == 'word2vec':
        wv=KeyedVectors.load_wordsvec_format('./word2vec/weibo_zh_word2vec_format_'+str(embedding_size)+'.txt',binary=False)
    else:
        wv = KeyedVectors.load_word2vec_format('./glove/weibo_zh_glove_'+str(embedding_size)+'.txt',binary=False)
    # 基于预训练词向量，根据vocab生成嵌入矩阵 预训练的词向量中没有出现的词用0向量表示
    # 词向量的嵌入矩阵的行数为什么是len(vocab)+1呢？
    # 因为vocab词典中词的最小编号是从1开始的，为了保证vocab与嵌入矩阵的索引统一，所以做个加1操作
    print('构建嵌入矩阵')
    embedding_matrix=np.zeros((len(vocab)+1,embedding_size))
    for word,i in vocab.items():
        try:
            embedding_matrix[i]=wv[str(word)]
        except KeyError:
            continue
    # 初始化网络模型
    print('初始化网络模型')
    model = CNN_model(max_len,len(vocab)+1,embedding_size,embedding_matrix)
    metrics = keras.metrics
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print('Train...')
    y_train_one_hot_labels = keras.utils.to_categorical(y_train, num_classes=2)  # 将标签转换为one-hot编码
    y_dev_one_hot_labels = keras.utils.to_categorical(y_dev, num_classes=2)  # 将标签转换为one-hot编码
    model.fit(x_train, y_train_one_hot_labels,batch_size=batch_size,epochs=epochs,verbose=1
              ,validation_data=(x_dev, y_dev_one_hot_labels))
    # 统计测试数据集的准确率的方式一
    y_predict=model.predict(x_test,batch_size=batch_size,verbose=1)
    y_predict = np.argmax(y_predict, axis=1)  # 获得最大概率对应的标签
    report = classification_report(y_test, y_predict,digits=4)
    result=str(report)
    print(result)
    with open(output_path+'train_cnn_result_'+word_vector_type+'_'+str(embedding_size)+'.txt','w',encoding='utf-8') as f:
        f.write(result)
    # 保存网络模型
    model.save(output_path+'weibo_cnn_model_'+word_vector_type+'_'+str(embedding_size)+'.h5')
    print('模型保存成功')
