export BERT_BASE_DIR="./chinese_L-12_H-768_A-12"
export OUTPUT_DIR="./bert_weibo_output"
bert-base-serving-start \
	-model_dir $OUTPUT_DIR \
	-bert_model_dir $BERT_BASE_DIR \
	-model_pb_dir $OUTPUT_DIR \
	-mode CLASS \
	-max_seq_len 128 \
	-port 7006 \
	-port_out 7007
