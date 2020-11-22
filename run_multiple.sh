#!/bin/bash
# AESLC
nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=100000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/experiments/xent1/aeslc_1k > aeslc_xent_1k_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/experiments/xent1/aeslc_1k/model.ckpt-16000 --evaluate_test > aeslc_XENT_eval1_1k.txt

# CNN/DM
nohup python3 pegasus/bin/train.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:cnn_dailymail/plain_text-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=100000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/experiments/xent1/cnndm_1k > cnndm_xent_1k_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_1k/model.ckpt-210000 --evaluate_test > cnndm_XENT_eval1_1k.txt

# GIGAWORD
nohup python3 pegasus/bin/train.py --params=gigaword_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:gigaword-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=100000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/experiments/xent1/gigaword_1k > gigaword_xent_1k_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=gigaword_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/gigaword_1k/model.ckpt-90000 --evaluate_test > gigaword_XENT_eval1_1k.txt

# REDDIT-TIFU
nohup python3 pegasus/bin/train.py --params=reddit_tifu_long_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds_transformed:reddit_tifu/long-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=100000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/experiments/xent1/reddit_1k > reddit_xent_1k_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=reddit_tifu_long_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/reddit_1k/model.ckpt-12000 --evaluate_test > reddit_XENT_eval1_1k.txt

# ARXIV
nohup python3 pegasus/bin/train.py --params=arxiv_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:scientific_papers/arxiv-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=100000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/experiments/xent1/arxiv_1k > arxiv_xent_1k_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=arxiv_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/arxiv_1k/model.ckpt-74000 --evaluate_test > arxiv_XENT_eval1_1k.txt

# PUBMED
nohup python3 pegasus/bin/train.py --params=pubmed_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:scientific_papers/pubmed-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=100000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/experiments/xent1/pubmed_1k > pubmed_xent_1k_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=pubmed_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/pubmed_1k/model.ckpt-100000 --evaluate_test > pubmed_XENT_eval1_1k.txt

# MULTI-NEWS
nohup python3 pegasus/bin/train.py --params=multi_news_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:multi_news-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=100000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/experiments/xent1/multinews_1k > mulitnews_xent_1k_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=multi_news_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.9 --model_dir=ckpt/pegasus_ckpt/multinews_1k/model.ckpt-80000 --evaluate_test > multinews_XENT_eval1_1k.txt

# BILLSUM
nohup python3 pegasus/bin/train.py --params=billsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds_transformed:billsum-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=100000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/experiments/xent1/billsum_1k > billsum_xent_1k_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=billsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/billsum_1k/model.ckpt-100000 --evaluate_test > billsum_XENT_eval1_1k.txt


### ERRORS w/ DATASET
# NEWSROOM
# nohup python3 pegasus/bin/train.py --params=newsroom_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:newsroom-train-take_1000,batch_size=1,learning_rate=0.0004,train_steps=104000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/newsroom_1k > newsroom_XENT_trial1_1k.txt
# nohup python3 pegasus/bin/evaluate.py --params=newsroom_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/newsroom_1k/model.ckpt-104000 --evaluate_test > newsroom_XENT_eval1_1k.txt

# BIGPATENT
# nohup python3 pegasus/bin/train.py --params=big_patent_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:big_patent/all-train-take_1000,batch_size=1,learning_rate=0.005,train_steps=300000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/bigpatent_1k > bigpatent_XENT_trial1_1k.txt
# nohup python3 pegasus/bin/evaluate.py --params=big_patent_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.7 --model_dir=ckpt/pegasus_ckpt/bigpatent_1k/model.ckpt-300000 --evaluate_test > bigpatent_XENT_eval1_1k.txt

# WIKIHOW
# nohup python3 pegasus/bin/train.py --params=wikihow_all_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:wikihow/all-train-take_1000,batch_size=1,learning_rate=0.0008,train_steps=50000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/wikihow_1k > wikihow_XENT_trial1_1k.txt
# nohup python3 pegasus/bin/evaluate.py --params=wikihow_all_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/wikihow_1k/model.ckpt-50000 --evaluate_test > wikihow_XENT_eval1_1k.txt

# XSUM
# nohup python3 pegasus/bin/train.py --params=xsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:xsum-train-take_1000,batch_size=1,learning_rate=0.0001,train_steps=105000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/xsum_1k > xsum_XENT_trial1_1k.txt
# nohup python3 pegasus/bin/evaluate.py --params=xsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/xsum_1k/model.ckpt-105000 --evaluate_test > xsum_XENT_eval1_1k.txt

# Run this - /bin/bash  /home/jsparnel/Data/pegasus/run_multiple.sh