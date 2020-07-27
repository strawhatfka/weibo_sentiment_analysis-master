export BERT_BASE_DIR="./chinese_L-12_H-768_A-12"
export OUTPUT_DIR="./bert_weibo_output"
python freeze_graph.py \
  -bert_model_dir $BERT_BASE_DIR \
  -model_dir $OUTPUT_DIR \
  -max_seq_len 128 \
  -num_labels 2
