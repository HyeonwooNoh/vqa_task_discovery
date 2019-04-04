CUDA_VISIBLE_DEVICES=0 python pretrain/trainer.py --model_type separable --prefix expand_depth --max_train_iter 4810 --seed 234 --expand_depth False
CUDA_VISIBLE_DEVICES=0 python pretrain/trainer.py --model_type separable --prefix expand_depth --max_train_iter 4810 --seed 345 --expand_depth False
CUDA_VISIBLE_DEVICES=0 python pretrain/trainer.py --model_type separable --prefix expand_depth --max_train_iter 4810 --seed 456 --expand_depth False
