# Dataset preparation
This document describes required datasets and their configuration for running experiments.

## Required third-party datasets
* [VisualGenome version 1.4](http://visualgenome.org/api/v0/api_home.html)
* [GloVe word vector](https://github.com/stanfordnlp/GloVe), especially *glove.6B.300d.txt* from [glove.6B.zip](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip)
* [VQA v2.0](https://visualqa.org/download.html)
* [Bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) feature for VQA v2.0, especially 36 features per image (fixed)
* [Bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) feature for Visual Genome. We use official bottom-up-attention based feature extractor [tools/genenerate_tsv.py](https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/generate_tsv.py) to extract the features from visual genome dataset. The preprocessed bottom-up-attention feature [??]() and image information file [??]() are available for download. Note that we preprocess bottom-up-attention feature with [???]() script.

## Initial path configurations
- data/VisualGenome/VG\_100K/\*.jpg
- data/VisualGenome/annotations/objects.json
- data/VisualGenome/annotations/object_synsets.json
- data/VisualGenome/annotations/attributes.json
- data/VisualGenome/annotations/attribute_synsets.json
- data/VisualGenome/annotations/relationships.json
- data/VisualGenome/annotations/relationship_synsets.json
- data/VisualGenome/annotations/region\_descriptions.json
- data/VisualGenome/annotations/image\_data.json
- data/VisualGenome/bottomup_feature_36/image_info.json
- data/VisualGenome/bottomup_feature_36/vfeat_bottomup_36.hdf5
- data/GloVe/glove.6B.50d.txt
- data/GloVe/glove.6B.100d.txt
- data/GloVe/glove.6B.200d.txt
- data/GloVe/glove.6B.300d.txt
- data/VQA_v2/annotations/v2_mscoco_train2014_annotations.json
- data/VQA_v2/annotations/v2_mscoco_val2014_annotations.json
- data/VQA_v2/questions/v2_OpenEnded_mscoco_train2014_questions.json
- data/VQA_v2/questions/v2_OpenEnded_mscoco_val2014_questions.json
- data/VQA_v2/images/val2014/\*.jpg
- data/VQA_v2/images/train2014/\*.jpg
- data/VQA_v2/images/test2015/\*.jpg
- data/VQA_v2/bottom_up_attention_36/trainval/trainval_resnet101_faster_rcnn_genome_36.tsv


## Useful scripts for dataset preparation
We provide useful scripts for preparing datasets in [data/scripts](../data/scripts) directory.
