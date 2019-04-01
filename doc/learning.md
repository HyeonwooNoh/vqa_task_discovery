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
