#!/bin/bash
nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_10,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc_reinforce_baseline_10 > reinforce_baseline_trial1_10.txt
nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc_reinforce_baseline_10/model.ckpt-32000 --evaluate_test > reinforce_baseline_eval1_10.txt

# Run this - /bin/bash  /home/jsparnel/Data/pegasus/run_multiple.sh