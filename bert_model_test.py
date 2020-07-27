from bert_base.client import BertClient


def predict(bert_client):
    print('请输入文本：')
    review = str(input())
    if review == 'exit':
        exit(0)
    else:
        try:
            result = bert_client.encode([review])
            label = result[0]['pred_label'][0]
            if int(label) == 1:
                polarity = '积极'
            else:
                polarity = '消极'
            print('文本的情感极性为:', polarity)
            predict(bert_client)
        except Exception as e:
            print(e)
            exit(0)


if __name__ == '__main__':
    with BertClient(show_server_config=False, check_version=False
            , check_length=False, mode='CLASS', port=7006, port_out=7007) as bc:
        predict(bc)
