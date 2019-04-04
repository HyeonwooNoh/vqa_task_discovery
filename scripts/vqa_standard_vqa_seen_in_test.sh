TF_RECORD_DIR=data/preprocessed/vqa_v2/qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1_with_seen_answer_in_test/tf_record_memft
for VQA_SEED in 234 345 456; do
    CUDA_VISIBLE_DEVICES=0 python vqa/trainer.py --model_type "standard-vqa" --seed ${VQA_SEED} --tf_record_dir ${TF_RECORD_DIR} --prefix "seen-in-test"
done
