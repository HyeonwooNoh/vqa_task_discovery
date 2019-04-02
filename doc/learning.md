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
* Separable classifier: [./scripts/pretrain_separable.sh](../scripts/pretrain_separable.sh)

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
### Training for VQA
Use the following script for learning VQA model.
```bash
python vqa/trainer.py --model_type proposed --pretrained_param_path ${MODEL_PATH} --pretrain_word_weight_dir ${WORD_WEIGHT_DIR} --prefix ${TRAIN_DIR_NAME_PREFIX} --seed ${VQA_SEED}
```
Example script:
```bash
python vqa/trainer.py --model_type proposed --pretrained_param_path train_dir/proposed_expand_depth_bs512_lr0.001_dpFalse_seed234/model-4801 --pretrain_word_weight_dir train_dir/proposed_expand_depth_bs512_lr0.001_dpFalse_seed234/word_weights_model-4801 --prefix pretrainseed234 --seed 234
```
