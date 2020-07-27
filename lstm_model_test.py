import tensorflow as tf
from tokenization import FullTokenizer
import pandas as pd
from sklearn.metrics import classification_report
keras = tf.keras
metrics = keras.metrics

print('模型加载中...')
model = keras.models.load_model('./lstm_weibo_output/weibo_lstm_model_128.h5')
print('模型加载结束...')
max_seq_len = 128
tokenizer = FullTokenizer('./vocab.txt')


def predict():
    print('请输入文本：')
    review = str(input())
    if review == 'exit':
        exit(0)
    else:
        try:
            '''
            review:哈哈，流泪了，泪
            tokenizer.tokenize(review):['哈', '哈', '，', '流', '泪', '了', '，', '泪']
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(review)):[1506, 1506, 8024, 3837, 3801, 749, 8024, 3801]
            
            总结：tokenizer.tokenize()方法其实就是对输入文本进行分词处理；tokenizer.convert_tokens_to_ids()方法将词转化为词在词典中所对应的索引。
            keras.preprocessing.sequence.pad_sequences()方法的第一个参数是一个二维list，
            sequences: List of lists, where each element is a sequence.
            如：[[1506, 1506, 8024, 3837, 3801, 749, 8024, 3801]]
            '''
            # 使输入样本等长
            review_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(review))]
            review_ids = keras.preprocessing.sequence.pad_sequences(review_ids, maxlen=max_seq_len)
            label = model.predict_classes(review_ids)
            if label[0][0]==1:
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
    x_datas, y_datas = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in reviews.review.values], reviews.label.values
    x_datas = keras.preprocessing.sequence.pad_sequences(x_datas, maxlen=max_seq_len)
    predicts = model.predict_classes(x_datas)
    report = classification_report(y_datas, predicts, digits=4)
    print(str(report))

if __name__ == '__main__':
    evaluate('./data/train.csv')
    evaluate('./data/dev.csv')
    evaluate('./data/test.csv')
    predict()
