# 项目介绍 #
基于 [weibo\_senti_100k.csv](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb) 数据集，本项目分别使用朴素贝叶斯、逻辑回归、LSTM、CNN、BERT等模型进行了实验，其中涉及的词向量表示方式包括one-hot、Bag of Words、TF-IDF、Word2Vec、Glove等。对于Word2Vec和Glove词向量的构建过程，本项目也提供了相关代码。项目中模型的训练运行脚本为train.sh，如bert\_train.sh，模型的测试运行脚本为test.sh，如bert\_test.sh。此外，本项目也会给出如何将训练好的BERT模型以服务的形式进行部署，以满足商业应用中的实时性需求。针对具体模型的使用，请读者查看\*\_README.md文件。希望通过本项目的学习，读者能够对情感分析中常用的模型技术有进一步的理解。
# 环境准备 #
bert-base==0.0.7<br>
fire==0.3.1<br>
gensim==3.8.1<br>
h5py==2.10.0<br>
jieba==0.42.1<br>
numpy==1.18.1<br>
pandas==1.0.1<br>
tensorflow-gpu==1.15.3<br>
scikit-learn==0.23.1<br>
# 数据准备 #
本项目中所使用的数据集存放在data文件夹下，名为weibo\_senti\_100k.csv，运行代码train\_valid\_test_split.py生成train.csv、dev.csv、test.csv文件。拆分数据集的比例可以根据需要做相应修改，目前train\_valid\_test\_split.py对于训练集、验证集、测试集的拆分比例为8:1:1。
# 训练词向量 #
本项目中只有CNN模型程序代码涉及了预训练词向量的加载，因此其他模型的程序代码在环境准备完毕的情况下可以不考虑预训练词向量直接运行相关脚本文件即可。然而，CNN模型需要提前准备好预训练词向量，这里使用Word2Vec与Glove工具进行训练，具体训练词向量的步骤请查看word2vec和glove文件夹。
