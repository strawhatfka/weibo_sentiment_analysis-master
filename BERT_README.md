# BERT模型训练 #
在运行bert\_train.sh文件之前，首先需要下载 [chinese\\_L-12\\_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) 文件到项目根目录下，并解压压缩文件。
# BERT模型部署 #
当模型训练结束之后，可以在bert\_weibo\_output目录下生成.ckpt文件。由于.ckpt模型文件比较大，每次预测都需要较长时间的运行，因此需要将.ckpt文件进行压缩转化为.pb文件，然后再以bert服务的形式去使用，具体可以参考：[bert-as-service](https://github.com/hanxiao/bert-as-service)<br>
1. 运行bert\_freeze_graph.sh，生成.pb文件。<br>
2. 运行bert-base-serving-start.sh，启动bert服务。
# BERT模型测试 #
在启动bert服务之后就可以运行bert_test.sh进行测试。