#!/bin/bash
# AESLC
# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc_baseline_ffn_1k > reinforce_baseline_ffn_trial1_1k.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc_baseline_ffn_1k/model.ckpt-32000 --evaluate_test > reinforce_baseline_ffn_eval1_1k.txt

# CNN/DM
nohup python3 pegasus/bin/train.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:cnn_dailymail/plain_text-train-take_10,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_10 > cnndm_XENT_trial1_10.txt
nohup python3 pegasus/bin/evaluate.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_10/model.ckpt-210000 --evaluate_test > cnndm_XENT_eval1_10.txt

nohup python3 pegasus/bin/train.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:cnn_dailymail/plain_text-train-take_100,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_100 > cnndm_XENT_trial1_100.txt
nohup python3 pegasus/bin/evaluate.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_100/model.ckpt-210000 --evaluate_test > cnndm_XENT_eval1_100.txt

nohup python3 pegasus/bin/train.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:cnn_dailymail/plain_text-train-take_1000,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_1k > cnndm_XENT_trial1_1k.txt
nohup python3 pegasus/bin/evaluate.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_1k/model.ckpt-210000 --evaluate_test > cnndm_XENT_eval1_1k.txt

nohup python3 pegasus/bin/train.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:cnn_dailymail/plain_text-train-take_10000,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_10k > cnndm_XENT_trial1_10k.txt
nohup python3 pegasus/bin/evaluate.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_10k/model.ckpt-210000 --evaluate_test > cnndm_XENT_eval1_10k.txt

nohup python3 pegasus/bin/train.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:cnn_dailymail/plain_text-train,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_all > cnndm_XENT_trial1_all.txt
nohup python3 pegasus/bin/evaluate.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail_all/model.ckpt-210000 --evaluate_test > cnndm_XENT_eval1_all.txt


# Run this - /bin/bash  /home/jsparnel/Data/pegasus/run_multiple.sh