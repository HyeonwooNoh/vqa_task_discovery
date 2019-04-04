TF_RECORD_DIR=data/preprocessed/vqa_v2/qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1_with_seen_answer_in_test/tf_record_memft
for PRETRAIN_SEED in 234 345 456; do
    for VQA_SEED in 234 345; do
        CUDA_VISIBLE_DEVICES=0 python vqa/trainer.py --model_type proposed-seen-in-test --pretrained_param_path train_dir/proposed_expand_depth_bs512_lr0.001_dpFalse_seed${PRETRAIN_SEED}/model-4801 --pretrain_word_weight_dir train_dir/proposed_expand_depth_bs512_lr0.001_dpFalse_seed${PRETRAIN_SEED}/word_weights_model-4801 --prefix pretrainseed${PRETRAIN_SEED} --seed ${VQA_SEED} --tf_record_dir ${TF_RECORD_DIR} 
    done
done
