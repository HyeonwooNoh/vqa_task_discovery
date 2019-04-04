for VQA_SEED in 234 345 456; do
    CUDA_VISIBLE_DEVICES=0 python vqa/trainer.py --model_type "standard-vqa" --seed ${VQA_SEED}
done
