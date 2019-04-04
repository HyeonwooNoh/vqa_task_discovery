for PRETRAIN_SEED in 234 345 456; do
    for VQA_SEED in 234 345; do
        CUDA_VISIBLE_DEVICES=0 python vqa/trainer.py --model_type separable --pretrained_param_path train_dir/separable_expand_depth_bs512_lr0.001_dpFalse_seed${PRETRAIN_SEED}/model-4801 --pretrain_word_weight_dir train_dir/separable_expand_depth_bs512_lr0.001_dpFalse_seed${PRETRAIN_SEED}/word_weights_model-4801 --prefix pretrainseed${PRETRAIN_SEED} --seed ${VQA_SEED}
    done
done
