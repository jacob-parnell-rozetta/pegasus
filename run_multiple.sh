#!/bin/bash
nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc_reinforce_baseline_R1 > reinforce_baseline_trial1.txt
nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc_reinforce_baseline_R1/model.ckpt-32000 --evaluate_test > reinforce_baseline_eval1.txt

# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc_mixed_10k_soft > mixed_loss_trial1_10k_soft.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc_mixed_10k_soft/model.ckpt-32000 --evaluate_test > mixed_loss_eval1_10k_soft.txt

# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc_mixed_10k_hard > mixed_loss_trial1_10k_hard.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.6 --model_dir=ckpt/pegasus_ckpt/aeslc_mixed_10k_hard/model.ckpt-32000 --evaluate_test > mixed_loss_eval1_10k_hard.txt

# Run this - /bin/bash  /home/jsparnel/Data/pegasus/run_multiple.sh