export BERT_BASE_DIR="./chinese_L-12_H-768_A-12"
export WEIBO_DIR="./data"
export OUTPUT_DIR="./bert_weibo_output"
python run_classifier.py \
  --task_name=WeiBo \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --data_dir=$WEIBO_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=8 \
  --learning_rate=1e-5 \
  --num_train_epochs=3.0 \
  --max_seq_length=128 \
  --output_dir=$OUTPUT_DIR
