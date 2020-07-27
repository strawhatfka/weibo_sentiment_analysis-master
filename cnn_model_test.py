import jieba
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
import os
os.environ['CUDA_VISIBLE_DEVICES']='/gpu:0'
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


keras = tf.keras
preprocessing = keras.preprocessing
print('模型加载中...')
# glove/word2vec
word_vector_type='word2vec'
embedding_size=100
model = keras.models.load_model('./cnn_weibo_output/weibo_cnn_model_'+word_vector_type+'_'+str(embedding_size)+'.h5')
print('模型加载结束...')
# 加载自定义词典
jieba.load_userdict('./word2vec/hownet_zh.txt')
# 获取停用词表
stopwords = []
with open('./word2vec/hit_stopwords.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stopwords.append(line.strip())
tokenizer = preprocessing.text.Tokenizer()
data = get_segment_words(pd.read_csv('./data/weibo_senti_100k.csv').review.values,stopwords)
tokenizer.fit_on_texts(data)

max_len = 128
batch_size = 32

def predict():
    print('请输入文本：')
    review = str(input())
    if review == 'exit':
        exit(0)
    else:
        try:
            words = get_segment_words([review], stopwords)
            print(wordis)
            review_ids = tokenizer.texts_to_sequences(words)
            review_ids = preprocessing.sequence.pad_sequences(review_ids, max_len)
            label = model.predict_classes(review_ids)
            
            if label[0] == 1:
                polarity = '积极'
            else:
                polarity = '消极'
            print('文本的情感极性为:', polarity)
            predict()
        except Exception as e:
            print(e)
            exit(0)


def evaluate(file_path):
    reviews = pd.read_csv(file_path)
    x_datas, y_datas = get_segment_words(reviews.review.values, stopwords), reviews.label.values
    x_datas=tokenizer.texts_to_sequences(x_datas)
    x_datas = preprocessing.sequence.pad_sequences(x_datas, maxlen=max_len)
    predicts = model.predict_classes(x_datas)
    report = classification_report(y_datas, predicts,digits=4)
    print(str(report))


if __name__ == '__main__':
    evaluate('./data/train.csv')
    evaluate('./data/dev.csv')
    evaluate('./data/test.csv')
    predict()
