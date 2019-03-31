# Dataset preparation
This document describes required datasets and their configuration for running experiments.

## Required third-party datasets
* [VisualGenome version 1.4](http://visualgenome.org/api/v0/api_home.html)
* [GloVe word vector](https://github.com/stanfordnlp/GloVe), especially *glove.6B.300d.txt* from [glove.6B.zip](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip)
* [VQA v2.0](https://visualqa.org/download.html)
* [Bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) feature for VQA v2.0, especially 36 features per image (fixed)
* [Bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) feature for Visual Genome. We use official bottom-up-attention based feature extractor [tools/genenerate_tsv.py](https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/generate_tsv.py) to extract the features from visual genome dataset. The preprocessed bottom-up-attention feature [??]() and image information file [??]() are available for download. Note that we preprocess bottom-up-attention feature with [???]() script.

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
#### Create image split for Visual Genome
Training in visual genome requires training / validataion / testing to analyze learning dynamics.
Use the following script to create splits.
```bash
# Run the scripts in data/
python tools/visualgenome/construct_image_split.py
```

#### Reduce GloVe vocab based on occurrence in Visual Genome dataset
We create reduced vocabulary based on word occurrences in visual genome dataset.
```bash
# Run the scripts in data/
python tools/visualgenome/generator_objects.py --vocab_path preprocessed/glove_vocab.json
python tools/visualgenome/generator_attributes.py --vocab_path preprocessed/glove_vocab.json
python tools/visualgenome/generator_relationships.py --vocab_path preprocessed/glove_vocab.json
python tools/visualgenome/generator_region_descriptions.py --vocab_path preprocessed/glove_vocab.json --max_description_length 10
python tools/construct_frequent_vocab.py --min_occurrence 50
```

## Useful scripts for dataset preparation
We provide useful scripts for preparing datasets in [data/scripts](../data/scripts) directory.
