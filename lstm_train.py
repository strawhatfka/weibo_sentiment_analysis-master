'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import tensorflow as tf

keras = tf.keras
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Embedding
# from keras.layers import LSTM
# from keras.datasets import imdb
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from tokenization import FullTokenizer

path = "./data/"
output_path='./lstm_weibo_output/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

# 精确率评价指标
def metric_precision(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    return precision


# 召回率评价指标
def metric_recall(y_true, y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    recall = TP / (TP + FN)
    return recall


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


def main():
    # pd_all = pd.read_csv(os.path.join(path, "weibo_senti_100k.csv"))
    # pd_all = shuffle(pd_all)
    # x_data, y_data = pd_all.review.values, pd_all.label.values
    # x_data = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in x_data]
    # x_train, x_test, y_train, y_test = train_test_split(np.array(x_data), y_data, test_size=0.2)
    #(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    tokenizer = FullTokenizer("vocab.txt")
    print('Loading data...')
    # 读取训练数据
    train_data = pd.read_csv(os.path.join(path, "train.csv"))
    # 读取验证数据
    dev_data = pd.read_csv(os.path.join(path, "dev.csv"))
    # 读取测试数据
    test_data = pd.read_csv(os.path.join(path, "test.csv"))
    x_train, y_train = train_data.review.values, train_data.label.values
    x_dev, y_dev = dev_data.review.values, dev_data.label.values
    x_test, y_test = test_data.review.values, test_data.label.values
    # tokenize to ids
    x_train = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in x_train]
    x_dev = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in x_dev]
    x_test = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in x_test]

    max_features = 21128
    # cut texts after this number of words (among top max_features most common words)
    maxlen = 128
    batch_size = 32

    print(len(x_train), 'train sequences')
    print(len(x_dev), 'dev sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_dev = keras.preprocessing.sequence.pad_sequences(x_dev, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_dev shape:', x_dev.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(max_features, 200))
    model.add(keras.layers.LSTM(300, dropout=0.2, recurrent_dropout=0.2))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    # metrics 设置方式一 使用keras内部函数或者自定义函数名
    # model.compile(loss='binary_crossentropy',optimizer='adam'
    #               ,metrics=['accuracy',metric_precision,metric_recall,metric_F1score])
    # metrics 设置方式二 使用metrics对象中的函数实例对象 在tensorflow.keras.metrics中才有。
    metrics = keras.metrics
    model.compile(loss='binary_crossentropy',optimizer='adam'
                  ,metrics=['accuracy',metrics.Precision(),metrics.Recall()])

    print('Train...')
    model.fit(x_train, y_train,batch_size=batch_size,epochs=15,verbose=1
              ,validation_data=(x_dev, y_dev))

    # 统计测试数据集的准确率的方式一
    y_predicts=model.predict(x_test,batch_size=batch_size,verbose=1)
    #
    print('y_predicts.shape:', y_predicts.shape)
    print('y_predicts:', y_predicts)
    # 判断预测结果中每行是否大于一列，如果大于一列，每个样本的预测类别，就取概率值最大的列索引对应的类别
    if y_predicts.shape[-1] > 1:
        print('if true')
        y_predicts=y_predicts.argmax(axis=-1).tolist()
    else:
        print('if false')
        y_predicts=(y_predicts > 0.5).astype('int32').tolist()
    right_num=0
    total=len(y_test)
    for i in range(total):
        if y_predicts[i][0]==y_test[i]:
            right_num+=1
    result = 'Test accuracy:%.2f'%(right_num*100/total)
    # 统计测试数据集的准确率的方式二 该方式就是直接使用keras模型实例中的评估方法去评估测试数据集即可
    evaluate = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=1)
    result+='\n=========================================\n'+'loss,accuracy,precision,recall,f1-score:'+str(evaluate)
    # 方式三 使用scikit-learn 中的classification_report方法 计算p，r，f1
    y_predict = model.predict_classes(x_test,batch_size=batch_size,verbose=1)
    report = classification_report(y_test,y_predict,digits=4)
    result+='\n=========================================\n'+report
    print(result)
    with open(output_path+'train_lstm_result.txt','w',encoding='utf-8') as f:
        f.write(result)
    # 保存网络模型
    model.save(output_path+'weibo_lstm_model.h5')
    print('模型保存成功')
if __name__ == '__main__':
    main()
