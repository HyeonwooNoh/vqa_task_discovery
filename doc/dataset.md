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


## Useful scripts for dataset preparation
We provide useful scripts for downloading and configuring third-party datasets in [data/scripts](../data/scripts) directory.


# Preprocess data
## Preprocessing GloVe 
GloVe data is proprocessed to extract vocabulary list with special tokens for setence processing. For the preprocessing, use the following script in path ```data/```.
```bash
# Run the script in data/
python tools/preprocess_glove.py
```
This script will create two files:
* data/preprocessed/glove_vocab.json
* data/preprocessed/glove.6B.300d.hdf5

## Preprocessing Visual Genome
### Smaller vocabulary based on occurrence in Visual Genome dataset
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

### Construct pretraining data

#### Pretraining data split
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

#### Sampled bottomup-attention features
Our code preload all bottomup-attention features for whole dataset into ram to minimize overhead for reading feature from HDD or SDD. This approach increases the training speed significantly because reading a large features is the important bottleneck for the training speed.
To support the preloading features, we should construct new feature files consisting of used images only. The following script will construct the new files.
```bash
# Run the script in root directory /
python data/tools/visualgenome/sample_bottomup_vfeat_for_memft.py
```
The script will create two features for training set and validation set separately in the directory (memft_all_new_vocab50_obj3000_attr1000_maxlen10).
```bash
train_vfeat.hdf5  # visual features for visual genome training set
val_vfeat.hdf5  # visual features for visual genome validation set
```

#### Word set extracted from WordNet 

For unsupervised task discovery, WordNet should be preprocessed to create word sets that are used for sampling task specifications. Use the following script to preprocess WordNet
```bash
# Run the script in root directory /
python data/tools/visualgenome/find_word_group.py --expand_depth=False
```
The script should create the ```wordset_dict5_depth0.pkl``` file in ```memft_all_new_vocab50_obj3000_attr1000_maxlen10``` directory.

## Preprocessing VQA

### VQA with Out-of-vocabulary Answers

VQA with out-of-vocabulary answers split can be downloaded from the [[link](http://cvlab.postech.ac.kr/~hyeonwoonoh/research/vqa_task_discovery/preprocessed/vqa_v2/qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1.tar.gz)].

* *Download link for the VQA with out-of-vocabulary answers split [[link](http://cvlab.postech.ac.kr/~hyeonwoonoh/research/vqa_task_discovery/preprocessed/vqa_v2/qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1.tar.gz)]*


Extract the file to a path
```bash
data/preprocessed/vqa_v2qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1
```
The directory should include the following files
```bash
answer_dict.pkl  # dictionary for VQA answers
vocab.pkl  # dictionary for all words appearing in VQA
attribute_list.pkl  # list of attribute words in VQA answers
object_list.pkl  # list of object words in VQA answers
merged_annotations.pkl  # VQA annotations combining both original training and validation set
obj_attrs_split.pkl  # Split of object and attribute words for training and testing.
qa_split.pkl  # Split of question answer pairs, which is constructed based on obj_attrs_split.pkl
pure_test_qid2anno.pkl  # pure test qids and their annotations for the final evaluation.
used_image_path.txt  # Path to mscoco image used for learning or evaluation
```

If you want to understand how this split is created, refer to following scripts. The out-of-vocabulary answers split is created as follows.
```bash
# Run the script in root directory /
python data/tools/vqa_v2/qa_split_objattr_answer_memft_genome.py  # Create out-of-vocabulary split
python data/tools/vqa_v2/construct_vocab_objattr_memft_genome.py  # Construct vocabulary
python data/tools/vqa_v2/make_qid2anno_trainval.py
python data/tools/vqa_v2/make_pure_test_qid2anno.py  # Construct pure test set whose answers are not exposed to training set at all
```
Note that we have separate *pure test set*, because VQA usually have 10 different answers for each questions and we need to ensure any of these answers was not exposed during training. The *pure test set* is used for the final evaluation.

### VQA with both Out-of-vocabulary Answers and Learned Answers
If you want to understand how this split is created, refer to following scripts. The out-of-vocabulary answers split is created as follows.
```bash
# Run the script in root directory /
python data/tools/vqa_v2/qa_split_objattr_answer_memft_genome_seen_answer_in_test.py   # Create split
python data/tools/vqa_v2/construct_vocab_objattr_memft_genome.py --qa_split_dir data/preprocessed/vqa_v2/qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1_with_seen_answer_in_test  # Construct vocabulary
python data/tools/vqa_v2/make_qid2anno_trainval.py  # If you have never run this script before
python data/tools/vqa_v2/make_test_qid2anno_seen_answer_in_test.py  # Construct annotations for the final evaluation
```

### Preprocessing for learning and evaluation
The training script uses tf_record for loading annotations such as image id, bounding boxes and descriptions during while running the script, and uses separate visual feature file to preload all visual feature before running the script.
The tf_record files are generated with the script
```bash
# Run the script in root directory /
python data/tools/vqa_v2/generator_tf_record_memft_genome.py
```
To preload all visual features at once, following script is used for preprocessing bottomup-attention features.
```bash
# Run the script in root directory /
python data/tools/vqa_v2/process_bottom_up_attention_36_memft_genome.py
```
