import joblib
import jieba
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


# 加载自定义词典
jieba.load_userdict('./word2vec/hownet_zh.txt')
# 获取停用词表
stopwords = []
with open('./word2vec/hit_stopwords.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stopwords.append(line.strip())


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


# 训练时CountVectorizer用的什么vocab 测试时也必须使用原本的vocab
#vectorizer = CountVectorizer()
#FLAG='countvectorizer'
#vectorizer = TfidfVectorizer()
#FLAG='tfidfvectorizer'
vectorizer = HashingVectorizer()
FLAG='hashingvectorizer'

reviews = pd.read_csv('./data/train.csv')
vectorizer.fit_transform(get_sentences(reviews.review.values, stopwords))
print('模型加载中...')
model = joblib.load('./lr_weibo_output/'+'weibo_lr_'+FLAG+'_model.pkl')
print('模型加载结束...')

def predict():
    print('请输入文本：')
    review = str(input())
    if review == 'exit':
        exit(0)
    else:
        try:
            sentences = get_sentences([review], stopwords)
            print(sentences)
            review_ids = vectorizer.transform(sentences)
            label = model.predict(review_ids)
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
    x_datas, y_datas = get_sentences(reviews.review.values, stopwords), reviews.label.values
    x_datas = vectorizer.transform(x_datas)
    predicts = model.predict(x_datas)
    report = classification_report(y_datas, predicts, digits=4)
    print(str(report))


if __name__ == '__main__':
    evaluate('./data/train.csv')
    evaluate('./data/dev.csv')
    evaluate('./data/test.csv')
    predict()
