# Dataset preparation
This document describes required datasets and their configuration for running experiments.

## Required third-party datasets
* [VisualGenome version 1.4](http://visualgenome.org/api/v0/api_home.html)
* [GloVe word vector](https://github.com/stanfordnlp/GloVe), especially *glove.6B.300d.txt* from [glove.6B.zip](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip)
* [VQA v2.0](https://visualqa.org/download.html)
* [Bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) feature for VQA v2.0, especially 36 features per image (fixed)
* [Bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) feature for Visual Genome. We use official bottom-up-attention based feature extractor [tools/genenerate_tsv.py](https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/generate_tsv.py) to extract the features from visual genome dataset. The preprocessed bottom-up-attention feature [vfeat_bottomup_36.hdf5](http://cvlab.postech.ac.kr/~hyeonwoonoh/research/vqa_task_discovery/VisualGenome/bottomup_feature_36/vfeat_bottomup_36.hdf5) and image information file [image_info.json](http://cvlab.postech.ac.kr/~hyeonwoonoh/research/vqa_task_discovery/VisualGenome/bottomup_feature_36/image_info.json) are available for download. Note that we preprocess bottom-up-attention feature with [data/tools/visualgenome/process_bottomup36.py](data/tools/visualgenome/process_bottomup36.py) script.

## Initial path configurations
Initial data path list is in [data/init_paths.txt](../data/init_paths.txt)

## Preprocess data
### Preprocessing GloVe 
GloVe data is proprocessed to extract vocabulary list with special tokens for setence processing. For the preprocessing, use the following script in path ```data/```.
```bash
# Run the script in data/
python tools/preprocess_glove.py
```
This script will create two files:
* data/preprocessed/glove_vocab.json
* data/preprocessed/glove.6B.300d.hdf5

### Preprocessing Visual Genome
#### Smaller vocabulary based on occurrence in Visual Genome dataset
Because there are too many unnecessary vocabulary in GloVe, we reduce its size based on their occurrence in Visual Genome datasets.
The reduced vocabulary file can be downloaded in [[link](http://cvlab.postech.ac.kr/~hyeonwoonoh/research/vqa_task_discovery/new_vocab50.json)].
Place this vocabulary in ```data/preprocessed/new_vocab50.json```

* *Download link for reduced vocabulary [[link](http://cvlab.postech.ac.kr/~hyeonwoonoh/research/vqa_task_discovery/new_vocab50.json)]*

The smaller vocabulary file is constrctured with the following scripts. Note that the result might differ due to randomness in the script, but these scripts can be used to understand the procedure for reducing vocabulary.
```bash
# Run the scripts in data/
python tools/visualgenome/construct_image_split.py
python tools/visualgenome/generator_objects.py --vocab_path preprocessed/glove_vocab.json
python tools/visualgenome/generator_attributes.py --vocab_path preprocessed/glove_vocab.json
python tools/visualgenome/generator_relationships.py --vocab_path preprocessed/glove_vocab.json
python tools/visualgenome/generator_region_descriptions.py --vocab_path preprocessed/glove_vocab.json --max_description_length 10
python tools/visualgenome/construct_frequent_vocab.py --min_word_occurrence 50
```

#### Construct pretraining data

Download preprocessed data for pretraining task conditional visual classifier on visual genome dataset.

* *Download link for preprocessed pretraining data [[link](http://cvlab.postech.ac.kr/~hyeonwoonoh/research/vqa_task_discovery/preprocessed/visualgenome/memft_all_new_vocab50_obj3000_attr1000_maxlen10.tar.gz)]*

Extract the file in the following directory
```bash
data/preprocessed/visualgenome/memft_all_new_vocab50_obj3000_attr1000_maxlen10
```
The extracted directory should contain following files
```bash
vocab.pkl  # vocabulary containing all word occurring in a visual genome dataset (including descriptions)
image_split.pkl  # [train, val] split containing list of image ids from visual genome dataset 
answer_dict.pkl  # vocabulary of visual concepts, which are answers for learning task conditional visual classifier
object_list.pkl  # names of objects (visual concept) that are used for pretraining
attribute_list.pkl  # names of attributes (visual concept) that are used for pretraining
train_processed.pkl  # preprocessed data including bounding boxes, visual concepts and blanked descriptions
train_image_info.pkl  # dictionary from visual genome image id to index within training set (to look up extracted bottomup-attention features)
val_processed.pkl  # preprocessed data including bounding boxes, visual concepts and blanked descriptions
val_image_info.pkl  # dictionary from visual genome image id to index within training set (to look up extracted bottomup-attention features)
image_id2processed.pkl  # all preprocessed information for visual genome image id
```

The preprocessed dataset is constructed by running the following script.
```bash
# Run the script in root directory /
python data/tools/visualgenome/generator_memft.py
```

Our code preload all bottomup-attention features for whole dataset into ram to minimize overhead for reading feature from HDD or SDD. This approach increases the training speed significantly because reading a large features is the important bottleneck for the training speed.
To support the preloading features, we should construct new feature files consisting of used images only. The following script will construct the new files.
```bash
python data/tools/visualgenome/sample_bottomup_vfeat_for_memft.py
```
The script will create two features for training set and validation set separately in the directory (memft_all_new_vocab50_obj3000_attr1000_maxlen10).
```bash
train_vfeat.hdf5  # visual features for visual genome training set
val_vfeat.hdf5  # visual features for visual genome validation set
```

## Useful scripts for dataset preparation
We provide useful scripts for preparing datasets in [data/scripts](../data/scripts) directory.
