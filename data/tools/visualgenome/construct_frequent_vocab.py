import argparse
import h5py
import json
import os

from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--object_dataset_path', type=str,
                    default='preprocessed/visualgenome/objects_glove_vocab_min_occ20',
                    help=' ')
parser.add_argument('--attribute_dataset_path', type=str,
                    default='preprocessed/visualgenome/attributes_glove_vocab_min_occ20',
                    help=' ')
parser.add_argument('--relationship_dataset_path', type=str,
                    default='preprocessed/visualgenome/relationships_glove_vocab_min_occ20',
                    help=' ')
parser.add_argument('--region_dataset_path', type=str,
                    default='preprocessed/visualgenome/region_descriptions_glove_vocab_max_len10',
                    help=' ')
parser.add_argument('--vocab_path', type=str,
                    default='preprocessed/glove_vocab.json', help=' ')
parser.add_argument('--min_word_occurrence', type=int, default=50, help=' ')
config = parser.parse_args()

vocab = json.load(open(config.vocab_path, 'r'))

word_count = {}

# objects
with h5py.File(os.path.join(config.object_dataset_path, 'data.hdf5'), 'r') as f:
    intseqs = f['data_info']['objects_intseq'].value
    intseq_lens = f['data_info']['objects_intseq_len'].value
    for intseq, intseq_len in zip(intseqs, intseq_lens):
        for i in intseq[:intseq_len]:
            w = vocab['vocab'][i]
            word_count[w] = word_count.get(w, 0) + 1

# attributes
with h5py.File(os.path.join(config.attribute_dataset_path, 'data.hdf5'), 'r') as f:
    intseqs = f['data_info']['attributes_intseq'].value
    intseq_lens = f['data_info']['attributes_intseq_len'].value
    for intseq, intseq_len in zip(intseqs, intseq_lens):
        for i in intseq[:intseq_len]:
            w = vocab['vocab'][i]
            word_count[w] = word_count.get(w, 0) + 1

# relationships
with h5py.File(os.path.join(config.relationship_dataset_path, 'data.hdf5'), 'r') as f:
    intseqs = f['data_info']['relationships_intseq'].value
    intseq_lens = f['data_info']['relationships_intseq_len'].value
    for intseq, intseq_len in zip(intseqs, intseq_lens):
        for i in intseq[:intseq_len]:
            w = vocab['vocab'][i]
            word_count[w] = word_count.get(w, 0) + 1

# region_descriptions
with h5py.File(os.path.join(config.region_dataset_path, 'data.hdf5'), 'r') as f:
    ids = open(os.path.join(config.region_dataset_path, 'id.txt'),
               'r').read().splitlines()
    for id_str in tqdm(ids, desc='region_description'):
        image_id, id = id_str.split()
        entry = f[image_id][id]
        for i in entry['description'].value:
            w = vocab['vocab'][i]
            word_count[w] = word_count.get(w, 0) + 1

vocab = []
for w, c in word_count.items():
    if c >= config.min_word_occurrence:
        vocab.append(w)

vocab = sorted(vocab)
if '<s>' not in vocab: vocab.append('<s>')
if '<e>' not in vocab: vocab.append('<e>')
if '<unk>' not in vocab: vocab.append('<unk>')

vocab = {'vocab': vocab, 'dict': {v: i for i, v in enumerate(vocab)}}
save_vocab_path = 'preprocessed/new_vocab{}.json'.format(
    config.min_word_occurrence)
json.dump(vocab, open(save_vocab_path, 'w'))
print('vocabulary with {} words are constructed: {}'.format(
    len(vocab['vocab']), save_vocab_path))
