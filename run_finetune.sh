# XENT FT
# AESLC
# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1,learning_rate=0.0002,train_steps=16000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc_1k > aeslc_XENT_trial1_1k.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc_1k/model.ckpt-16000 --evaluate_test > aeslc_XENT_eval1_1k.txt

# ARXIV
# nohup python3 pegasus/bin/train.py --params=arxiv_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:scientific_papers/arxiv-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/arxiv_ft_1000ex > arxiv_ft_1000ex_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=arxiv_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/arxiv_ft_1000ex/model.ckpt-2000 --evaluate_test > arxiv_ft_1000ex_eval.txt

# GIGAWORD
# nohup python3 pegasus/bin/train.py --params=gigaword_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:gigaword-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/gigaword_ft_1000ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/gigaword_ft_1000ex_relax_4 > gigaword_ft_1000ex_relax_4_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=gigaword_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/gigaword_ft_1000ex_relax_4/model.ckpt-2000 --evaluate_test > gigaword_ft_1000ex_relax_4_eval.txt

# CNN/DM
# nohup python3 pegasus/bin/train.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:cnn_dailymail/plain_text-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/cnndm_ft_1000ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/cnndm_ft_1000ex_relax_4 > cnndm_ft_1000ex_relax_4_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/cnndm_ft_1000ex_relax_4/model.ckpt-2000 --evaluate_test > cnndm_ft_1000ex_relax_4_eval.txt

# REDDIT-TIFU
# nohup python3 pegasus/bin/train.py --params=reddit_tifu_long_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds_transformed:reddit_tifu/long-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/reddit_tifu_ft_1000ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/reddit_tifu_ft_1000ex_relax_4 > reddit_ft_1000ex_relax_4_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=reddit_tifu_long_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/reddit_tifu_ft_1000ex_relax_4/model.ckpt-2000 --evaluate_test > reddit_ft_1000ex_relax_4_eval.txt

# PUBMED
# nohup python3 pegasus/bin/train.py --params=pubmed_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:scientific_papers/pubmed-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/pubmed_ft_1000ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/pubmed_ft_1000ex_relax_4 > pubmed_ft_1000ex_relax_4_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=pubmed_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/pubmed_ft_1000ex_relax_4/model.ckpt-2000 --evaluate_test > pubmed_ft_1000ex_relax_4_eval.txt

# MULTI-NEWS
# nohup python3 pegasus/bin/train.py --params=multi_news_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:multi_news-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/multi_news_ft_1000ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/multi_news_ft_1000ex_relax_4 > multinews_ft_1000ex_relax_4_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=multi_news_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/multi_news_ft_1000ex_relax_4/model.ckpt-2000 --evaluate_test > multinews_ft_1000ex_relax_4_eval.txt

# BILLSUM
# nohup python3 pegasus/bin/train.py --params=billsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds_transformed:billsum-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/billsum_ft_1000ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/billsum_ft_1000ex_relax_4 > billsum_ft_1000ex_relax_4_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=billsum_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/billsum_ft_1000ex_relax_4/model.ckpt-2000 --evaluate_test > billsum_ft_1000ex_relax_4_eval.txt


###
# RL FT
# nohup python3 pegasus/bin/train.py --params=arxiv_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:scientific_papers/arxiv-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/arxiv_ft_1000ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/arxiv_ft_1000ex_relax_4 > arxiv_ft_1000ex_relax_4_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=arxiv_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/arxiv_ft_1000ex_relax_4/model.ckpt-2000 --evaluate_test > arxiv_ft_1000ex_relax_4_eval.txt

# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_100,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/aeslc_ft_100ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_100ex_relax_4 > aeslc_ft_100ex_relax_4_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_100ex_relax_4/model.ckpt-2000 --evaluate_test > aeslc_ft_100ex_relax_4_eval.txt

# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/aeslc_ft_1000ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_1000ex_relax_4 > aeslc_ft_1000ex_relax_4_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_1000ex_relax_4/model.ckpt-2000 --evaluate_test > aeslc_ft_1000ex_relax_4_eval.txt

# long test
# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train,batch_size=1,learning_rate=0.0005,train_steps=400000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslctest > aeslctest.txt