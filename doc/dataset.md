# Dataset preparation
This document describes required datasets and their configuration for running experiments.

## Required third-party datasets
* [VisualGenome version 1.4](http://visualgenome.org/api/v0/api_home.html)
* [GloVe word vector](https://github.com/stanfordnlp/GloVe), especially *glove.6B.300d.txt* from [glove.6B.zip](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip)
* [VQA v2.0](https://visualqa.org/download.html)
* [Bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) feature for VQA v2.0, especially 36 features per image (fixed)
* [Bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) feature for Visual Genome. We use official bottom-up-attention based feature extractor [tools/genenerate_tsv.py](https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/generate_tsv.py) to extract the features from visual genome dataset. The preprocessed bottom-up-attention feature [??]() and image information file [??]() are available for download. Note that we preprocess bottom-up-attention feature with [???]() script.

## Path configurations
- VisualGenome/VG\_100K/\*.jpg
- VisualGenome/annotations/objects.json
- VisualGenome/annotations/attributes.json
- VisualGenome/annotations/relationships.json
- VisualGenome/annotations/region\_descriptions.json
- VisualGenome/annotations/region\_graphs.json
- VisualGenome/annotations/scene\_graphs.json
- VisualGenome/annotations/object\_alias.txt
- VisualGenome/annotations/relationship\_alias.txt
- GloVe/glove.6B.50d.txt
- GloVe/glove.6B.100d.txt
- GloVe/glove.6B.200d.txt
- GloVe/glove.6B.300d.txt

## Useful scripts for dataset preparation
We provide useful scripts for preparing datasets in [data/scripts](../data/scripts) directory.
