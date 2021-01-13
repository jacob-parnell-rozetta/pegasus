# Mass experiment tests
echo "CNN/DM"
nohup python3 pegasus/bin/train.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:cnn_dailymail/plain_text-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/experiments/xent1/cnndm_1k/model.ckpt-6001 --model_dir=ckpt/experiments/reinforce/cnndm_lsum_test  > cnndm_lsum_0.9x_train.txt
nohup python3 pegasus/bin/evaluate.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/experiments/reinforce/cnndm_lsum_test/model.ckpt-2000 --evaluate_test > cnndm_lsum_0.9x_eval.txt

# RUN ALL
echo "AESLC"
nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/experiments/xent1/aeslc_1k/model.ckpt-5001 --model_dir=ckpt/experiments/reinforce/validation_tests/aeslc_1k_greedy_tf/0.1 > aeslc_val_greedytf_0.1x_train.txt
nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/experiments/reinforce/validation_tests/aeslc_1k_greedy_tf/0.1/model.ckpt-2000 > aeslc_val_greedytf_0.1x_eval.txt

echo "GIGAWORD"
nohup python3 pegasus/bin/train.py --params=gigaword_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:gigaword-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/experiments/xent1/gigaword_1k/model.ckpt-6001 --model_dir=ckpt/experiments/reinforce/validation_tests/gigaword_1k_greedy_tf/0.1 > gigaword_val_greedytf_0.1x_train.txt
nohup python3 pegasus/bin/evaluate.py --params=gigaword_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/experiments/reinforce/validation_tests/gigaword_1k_greedy_tf/0.1/model.ckpt-2000 > gigaword_val_greedytf_0.1x_eval.txt

echo "REDDIT-TIFU"
nohup python3 pegasus/bin/train.py --params=reddit_tifu_long_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds_transformed:reddit_tifu/long-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/experiments/xent1/reddit_1k/model.ckpt-6001 --model_dir=ckpt/experiments/reinforce/validation_tests/reddit_1k_greedy_tf/0.1 > reddit_val_greedytf_0.1x_train.txt
nohup python3 pegasus/bin/evaluate.py --params=reddit_tifu_long_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/experiments/reinforce/validation_tests/reddit_1k_greedy_tf/0.1/model.ckpt-2000 > reddit_val_greedytf_0.1x_eval.txt

echo "ARXIV"
nohup python3 pegasus/bin/train.py --params=arxiv_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:scientific_papers/arxiv-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/experiments/xent1/arxiv_1k/model.ckpt-8001 --model_dir=ckpt/experiments/reinforce/validation_tests/arxiv_1k_greedy_tf/0.1 > arxiv_val_greedytf_0.1x_train.txt
nohup python3 pegasus/bin/evaluate.py --params=arxiv_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/experiments/reinforce/validation_tests/arxiv_1k_greedy_tf/0.1/model.ckpt-2000 > arxiv_val_greedytf_0.1x_eval.txt

echo "PUBMED"
nohup python3 pegasus/bin/train.py --params=pubmed_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:scientific_papers/pubmed-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/experiments/xent1/pubmed_1k/model.ckpt-8001 --model_dir=ckpt/experiments/reinforce/validation_tests/pubmed_1k_greedy_tf/0.1 > pubmed_val_greedytf_0.1x_train.txt
nohup python3 pegasus/bin/evaluate.py --params=pubmed_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/experiments/reinforce/validation_tests/pubmed_1k_greedy_tf/0.1/model.ckpt-2000 > pubmed_val_greedytf_0.1x_eval.txt

echo "MULTI-NEWS"
nohup python3 pegasus/bin/train.py --params=multi_news_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:multi_news-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/experiments/xent1/multinews_1k/model.ckpt-7001 --model_dir=ckpt/experiments/reinforce/validation_tests/multinews_1k_greedy_tf/0.1 > multinews_val_greedytf_0.1x_train.txt
nohup python3 pegasus/bin/evaluate.py --params=multi_news_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/experiments/reinforce/validation_tests/multinews_1k_greedy_tf/0.1/model.ckpt-2000 > multinews_val_greedytf_0.1x_eval.txt

echo "BILLSUM"
nohup python3 pegasus/bin/train.py --params=billsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds_transformed:billsum-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/experiments/xent1/billsum_1k/model.ckpt-9001 --model_dir=ckpt/experiments/reinforce/validation_tests/billsum_1k_greedy_tf/0.1 > billsum_val_greedytf_0.1x_train.txt
nohup python3 pegasus/bin/evaluate.py --params=billsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/experiments/reinforce/validation_tests/billsum_1k_greedy_tf/0.1/model.ckpt-2000 > billsum_val_greedytf_0.1x_eval.txt

# long test
echo "EXTRA"
nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train,batch_size=1,learning_rate=0.0005,train_steps=500000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslctest > aeslctest.txt
