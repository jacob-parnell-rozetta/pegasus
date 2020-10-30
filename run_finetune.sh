# XENT FT
# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_10,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_10ex > aeslc_ft_10ex_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_10ex/model.ckpt-2000 --evaluate_test > aeslc_ft_10ex_eval.txt

# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_100,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_100ex > aeslc_ft_100ex_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_100ex/model.ckpt-2000 --evaluate_test > aeslc_ft_100ex_eval.txt

# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_1000ex > aeslc_ft_1000ex_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_1000ex/model.ckpt-2000 --evaluate_test > aeslc_ft_1000ex_eval.txt

###
# RL FT
# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_10,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/aeslc_ft_10ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_10ex_2 > aeslc_ft_10ex_2_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_10ex_2/model.ckpt-2000 --evaluate_test > aeslc_ft_10ex_2_eval.txt

# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_100,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/aeslc_ft_100ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_100ex_relax > aeslc_ft_100ex_relax_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_100ex_relax/model.ckpt-2000 --evaluate_test > aeslc_ft_100ex_relax_eval.txt

# nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1,learning_rate=0.0005,train_steps=2000 --train_init_checkpoint=ckpt/pegasus_ckpt/aeslc_ft_1000ex/model.ckpt-2000 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_1000ex_relax > aeslc_ft_1000ex_relax_train.txt
# nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=1 --model_dir=ckpt/pegasus_ckpt/aeslc_ft_1000ex_relax/model.ckpt-2000 --evaluate_test > aeslc_ft_1000ex_relax_eval.txt

