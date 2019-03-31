import h5py
import json
import numpy as np
import os

from tqdm import tqdm

GLOVE_PATH = 'GloVe/glove.6B.300d.txt'

VOCAB_PATH = 'preprocessed/glove_vocab.json'
PARAM_PATH = 'preprocessed/glove.6B.300d.hdf5'

if os.path.exists(VOCAB_PATH):
    raise RuntimeError('vocab exists: {}'.format(VOCAB_PATH))
if os.path.exists(PARAM_PATH):
    raise RuntimeError('glove param exists: {}'.format(PARAM_PATH))

if not os.path.exists('preprocessed'):
    os.makedirs('preprocessed')
    print('Create directory: preprocessed')

glove = open(GLOVE_PATH, 'r').read().splitlines()

vocab = []
glove_vector = []

for g in tqdm(glove, desc='process glove'):
    g = g.split()
    vocab.append(g[0])
    glove_vector.append(np.array([float(v) for v in g[1:]], dtype=np.float32))

vocab.extend(['<s>', '<e>', '<unk>'])

glove_vector = np.vstack(glove_vector).transpose()

f = h5py.File(PARAM_PATH, 'w')
f['param'] = glove_vector
f.close()
print('save in: {}'.format(PARAM_PATH))

vocab = {'vocab': vocab, 'dict': {v: i for i, v in enumerate(vocab)}}
json.dump(vocab, open(VOCAB_PATH, 'w'))
print('save in: {}'.format(VOCAB_PATH))
