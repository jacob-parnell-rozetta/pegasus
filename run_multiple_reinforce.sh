#!/bin/bash
# AESLC
nohup python3 pegasus/bin/train_reinforce_mixed.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1,learning_rate=0.0002,train_steps=16000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc_reinforce_mixed_6040 > reinforce_train_eval/aeslc_reinforce_mixed_6040_trial1_1k.txt
nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc_reinforce_mixed_6040/model.ckpt-16000 --evaluate_test > reinforce_train_eval/aeslc_reinforce_mixed_6040_eval1_1k.txt

# CNN/DM
nohup python3 pegasus/bin/train_reinforce_mixed.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:cnn_dailymail/plain_text-train-take_1000,batch_size=1,learning_rate=0.00005 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_reinforce_mixed_6040 > reinforce_train_eval/cnndm_reinforce_mixed_6040_trial1_1k.txt
nohup python3 pegasus/bin/evaluate.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_reinforce_mixed_6040/model.ckpt-210000 --evaluate_test > reinforce_train_eval/cnndm_reinforce_mixed_6040_eval1_1k.txt

# GIGAWORD
nohup python3 pegasus/bin/train_reinforce_mixed.py --params=gigaword_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:gigaword-train-take_1000,batch_size=1,learning_rate=0.0008,train_steps=90000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/gigaword_reinforce_mixed_6040 > reinforce_train_eval/gigaword_reinforce_mixed_6040_trial1_1k.txt
nohup python3 pegasus/bin/evaluate.py --params=gigaword_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/gigaword_reinforce_mixed_6040/model.ckpt-90000 --evaluate_test > reinforce_train_eval/gigaword_reinforce_mixed_6040_eval1_1k.txt

# REDDIT-TIFU
nohup python3 pegasus/bin/train_reinforce_mixed.py --params=reddit_tifu_long_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds_transformed:reddit_tifu/long-train-take_1000,batch_size=1,learning_rate=0.0001,train_steps=12000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/reddit_reinforce_mixed_6040 > reinforce_train_eval/reddit_reinforce_mixed_6040_trial1_1k.txt
nohup python3 pegasus/bin/evaluate.py --params=reddit_tifu_long_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/reddit_reinforce_mixed_6040/model.ckpt-12000 --evaluate_test > reinforce_train_eval/reddit_reinforce_mixed_6040_eval1_1k.txt

# ARXIV
nohup python3 pegasus/bin/train_reinforce_mixed.py --params=arxiv_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:scientific_papers/arxiv-train-take_1000,batch_size=1,learning_rate=0.0008,train_steps=74000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/arxiv_reinforce_mixed_6040 > reinforce_train_eval/arxiv_reinforce_mixed_6040_trial1_1k.txt
nohup python3 pegasus/bin/evaluate.py --params=arxiv_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/arxiv_reinforce_mixed_6040/model.ckpt-74000 --evaluate_test > reinforce_train_eval/arxiv_reinforce_mixed_6040_eval1_1k.txt

# PUBMED
nohup python3 pegasus/bin/train_reinforce_mixed.py --params=pubmed_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:scientific_papers/pubmed-train-take_1000,batch_size=1,learning_rate=0.0002,train_steps=100000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/pubmed_reinforce_mixed_6040 > reinforce_train_eval/pubmed_reinforce_mixed_6040_trial1_1k.txt
nohup python3 pegasus/bin/evaluate.py --params=pubmed_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/pubmed_reinforce_mixed_6040/model.ckpt-100000 --evaluate_test > reinforce_train_eval/pubmed_reinforce_mixed_6040_eval1_1k.txt

# MULTI-NEWS
nohup python3 pegasus/bin/train_reinforce_mixed.py --params=multi_news_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:multi_news-train-take_1000,batch_size=1,learning_rate=0.00005,train_steps=80000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/multinews_reinforce_mixed_6040 > reinforce_train_eval/multinews_reinforce_mixed_6040_trial1_1k.txt
nohup python3 pegasus/bin/evaluate.py --params=multi_news_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.9 --model_dir=ckpt/pegasus_ckpt/multinews_reinforce_mixed_6040/model.ckpt-80000 --evaluate_test > reinforce_train_eval/multinews_reinforce_mixed_6040_eval1_1k.txt

# BILLSUM
nohup python3 pegasus/bin/train_reinforce_mixed.py --params=billsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds_transformed:billsum-train-take_1000,batch_size=1,learning_rate=0.0002,train_steps=100000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/billsum_reinforce_mixed_6040 > reinforce_train_eval/billsum_reinforce_mixed_6040_trial1_1k.txt
nohup python3 pegasus/bin/evaluate.py --params=billsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/billsum_reinforce_mixed_6040/model.ckpt-100000 --evaluate_test > reinforce_train_eval/billsum_reinforce_mixed_6040_eval1_1k.txt

# WIKIHOW
# nohup python3 pegasus/bin/train_reinforce_mixed.py --params=wikihow_all_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:wikihow/all-train-take_1000,batch_size=1,learning_rate=0.0008,train_steps=50000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/wikihow_reinforce_mixed_6040 > reinforce_train_eval/wikihow_reinforce_mixed_6040_trial1_1k.txt
# nohup python3 pegasus/bin/evaluate.py --params=wikihow_all_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/wikihow_reinforce_mixed_6040/model.ckpt-50000 --evaluate_test > reinforce_train_eval/wikihow_reinforce_mixed_6040_eval1_1k.txt

# XSUM
# nohup python3 pegasus/bin/train_reinforce_mixed.py --params=xsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:xsum-train-take_1000,batch_size=1,learning_rate=0.0001,train_steps=130000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/xsum_reinforce_mixed_6040 > reinforce_train_eval/xsum_reinforce_mixed_6040_trial1_1k.txt
# nohup python3 pegasus/bin/evaluate.py --params=xsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/xsum_reinforce_mixed_6040/model.ckpt-130000 --evaluate_test > reinforce_train_eval/xsum_reinforce_mixed_6040_eval1_1k.txt

# NEWSROOM
# nohup python3 pegasus/bin/train_reinforce_mixed.py --params=newsroom_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:newsroom-train-take_1000,batch_size=1,learning_rate=0.0004,train_steps=104000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/newsroom_reinforce_mixed_6040 > reinforce_train_eval/newsroom_reinforce_mixed_6040_trial1_1k.txt
# nohup python3 pegasus/bin/evaluate.py --params=newsroom_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/newsroom_reinforce_mixed_6040/model.ckpt-104000 --evaluate_test > reinforce_train_eval/newsroom_reinforce_mixed_6040_eval1_1k.txt

# BIGPATENT
# nohup python3 pegasus/bin/train_reinforce_mixed.py --params=big_patent_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:big_patent/all-train-take_1000,batch_size=1,learning_rate=0.005,train_steps=300000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/bigpatent_reinforce_mixed_6040 > reinforce_train_eval/bigpatent_reinforce_mixed_6040_trial1_1k.txt
# nohup python3 pegasus/bin/evaluate.py --params=big_patent_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.7 --model_dir=ckpt/pegasus_ckpt/bigpatent_reinforce_mixed_6040/model.ckpt-300000 --evaluate_test > reinforce_train_eval/bigpatent_reinforce_mixed_6040_eval1_1k.txt

# Run this - /bin/bash  /home/jsparnel/Data/pegasus/run_multiple_reinforce.sh
