for VQA_SEED in 234 345 456; do
    CUDA_VISIBLE_DEVICES=0 python vqa/trainer.py --model_type "answer-embedding" --seed ${VQA_SEED}
done
