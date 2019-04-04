# Learning
To reproduce results in the paper, follow instruction in the below after dataset preparation.
If you did not prepare dataset, please refer to [this document](dataset.md).
## Pretraining with Visual Genome
The proposed model can be pretrained with the following script.
```bash
python pretrain/trainer.py --model_type proposed --prefix expand_depth --max_train_iter 4810 --expand_depth False
```
Note that we use multiple seed to pretrain a model to count variance of transfer result introduced by different pretrained models.
Specifically, we use three different seeds for each models.
For convenience, following script can be used to run experiments with three different seeds.
```bash
./scripts/pretrain_proposed.sh
```
In the main paper, we compare with three baselines that should be pretrained. Following are pretraining scripts for the baselines.
* WordNet only: [./scripts/pretrain_wordnet.sh](../scripts/pretrain_wordnet.sh)
* Description only: [./scripts/pretrain_description.sh](../scripts/pretrain_description.sh)
* Separable classifier: [./scripts/pretrain_separable_classifier.sh](../scripts/pretrain_separable_classifier.sh)

## Transfer to VQA
### Export word weight
Transferring task conditional visual classifier to VQA requires rearranging the answers because the answer indices for pretraining and VQA are different.
For rearrangement, use the following script to export word weight.
```bash
python pretrain/export_word_weights.py --checkpoint {$CHECKPOINT_PATH}
```
This script will export word weight from pretrained task conditional visual classifier, which is then rearranged by VQA training script.
The CHECKPOINT_PATH should be path to model checkpointed at some iteration. e.g. ```CHECKPOINT_PATH=train_dir/proposed_expand_depth_bs512_lr0.001_dpFalse_seed234/model-4801```
Following script will export word weight of proposed model with seed=234 after 4801 iterations.
```bash
python pretrain/export_word_weights.py --checkpoint train_dir/proposed_expand_depth_bs512_lr0.001_dpFalse_seed234/model-4801
```
Note that experiments in the main paper is conducted by transferring task conditional visual classifier preatrained for 4801 iterations.
In case of separable classifier model, different script should be used because parameters for this model is defined differently.
```bash
python pretrain/export_word_weights_model_separable.py --checkpoint {$CHECKPOINT_PATH}
```
### Training for VQA
Transfer for VQA uses the following script. This script can be used with various arguments to reproduce results in the main paper.
```bash
python vqa/trainer.py --model_type ${MODEL_TYPE} --pretrained_param_path ${MODEL_PATH} --pretrain_word_weight_dir ${WORD_WEIGHT_DIR} --prefix ${TRAIN_DIR_NAME_PREFIX} --seed ${VQA_SEED}
```
Here is an example script:
```bash
python vqa/trainer.py --model_type proposed --pretrained_param_path train_dir/proposed_expand_depth_bs512_lr0.001_dpFalse_seed234/model-4801 --pretrain_word_weight_dir train_dir/proposed_expand_depth_bs512_lr0.001_dpFalse_seed234/word_weights_model-4801 --prefix pretrainseed234 --seed 234
```
The VQA training scripts we used for experiment in the main paper can be found in ```./scripts``` directory.
Following is the list of scripts and the corresponding experiments.
```bash
# Model comparison for VQA with out-of-vocabulary answers
vqa_proposed.sh
vqa_separable_classifier.sh
vqa_answer_embedding.sh
vqa_standard_vqa.sh
# Data comparison for VQA with out-of-vocabulary answers
vqa_proposed.sh
vqa_description.sh
vqa_wordnet.sh
# Combining knowledge learned by VQA
vqa_proposed_seen_in_test.sh
vqa_answer_embedding_seen_in_test.sh
vqa_standard_vqa_seen_in_test.sh
```

### Evaluation for VQA
After training VQA models with scripts above, we should run the following evaluation scripts in sequence.
```bash
python vqa/eval_multiple_model.py --train_dirs ${LIST_OF_TRAIN_DIR}
python vqa/eval_collection.py --train_dirs ${LIST_OF_TRAIN_DIR}
```
```vqa/eval_multiple_model.py``` script will evaluate every checkpoints in the directories listed in ```--train_dirs``` and evaluate results on the target VQA data.
```vqa/eval_collection.py``` script will collect all evaluation results in a single directory and generate two summary files
```bash
collect_eval_test_result.pkl
collect_eval_test_result.txt
```
The ```collect_eval_test_result.pkl``` file contains all numbers required to draw a result plot and ```collect_eval_test_result.txt``` is a text version file that can be easily checked.

See [./plot_for_paper.ipynb](../plot_for_paper.ipynb) for how the evaluation results in ```collect_eval_test_result.pkl``` are plotted.

Note that for **Combining knowledge learned by VQA** experiments use different scripts for collection evaluation results.
The evaluation script is
```bash
python vqa/eval_collection_seen_in_test.py --train_dirs ${LIST_OF_TRAIN_DIR}
```
Also note that **Combining knowledge learned by VQA** experiments require correctly setting ```--qa_split_dir``` arguments for both ```vqa/eval_multiple_model.py``` script and ```vqa/eval_collection_seen_in_test.py```.
It should be ```data/preprocessed/vqa_v2/qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1_with_seen_answer_in_test``` in default dataset configuration.
