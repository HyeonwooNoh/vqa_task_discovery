import argparse
import cPickle
import copy
import json
import os
import numpy as np

from collections import defaultdict
from nltk.corpus import wordnet as wn
from textblob import Word
from tqdm import tqdm

RANDOM_STATE = np.random.RandomState(123)


def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--genome_annotation_dir', type=str,
                    default='data/VisualGenome/annotations', help=' ')
parser.add_argument('--dir_name', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
parser.add_argument('--min_num_word', type=int, default=5, help='min num word in set')
parser.add_argument('--expand_depth', type=str2bool, default=False,
                    help='whether to use depth for wordset construction')
config = parser.parse_args()

config.object_synset_path = os.path.join(
    config.genome_annotation_dir, 'object_synsets.json')
config.attribute_synset_path = os.path.join(
    config.genome_annotation_dir, 'attribute_synsets.json')
config.answer_dict_path = os.path.join(
    config.dir_name, 'answer_dict.pkl')
config.save_wordset_path = os.path.join(
    config.dir_name, 'wordset_dict{}_depth{}.pkl'.format(config.min_num_word, int(config.expand_depth)))

object_synsets = json.load(open(config.object_synset_path, 'r'))
attribute_synsets = json.load(open(config.attribute_synset_path, 'r'))

# synsets['crisp'] = 'chip.n.04'
synsets = {}
for key, val in object_synsets.items():
    synsets[key] = val
for key, val in attribute_synsets.items():
    if key not in synsets:
        synsets[key] = val

answer_dict = cPickle.load(open(config.answer_dict_path, 'rb'))


vocab_with_synset = []
for v in answer_dict['vocab']:
    v_w = Word('_'.join(v.split()))
    if len(v_w.synsets) > 0:
        synsets[v] = v_w.synsets[0].name()
    if v in synsets:
        vocab_with_synset.append(v)

hypernym_set = set()
vocab_hypernyms = {}
max_depth = defaultdict(int)
for v in tqdm(vocab_with_synset, desc='make hypernymset'):
    # Synset('chip.n.04')
    word_synset = wn.synset(synsets[v])
    # set([Synset('nutriment.n.01'), Synset('food.n.01'), Synset('physical_entity.n.01'), Synset('entity.n.01'), Synset('matter.n.03'), Synset('dish.n.02'), Synset('snack_food.n.01'), Synset('substance.n.07'), Synset('chip.n.04')])
    hypernyms = []
    hypernym2distance = {}
    for hypernym, distance in list(word_synset.hypernym_distances()):
        hypernym = hypernym.name()
        hypernym2distance[hypernym] = distance
        hypernyms.append(hypernym)
        max_depth[hypernym] = max(max_depth[hypernym], distance)
    hypernyms = set(hypernyms)
    hypernyms = hypernyms - set([word_synset.name()])
    hypernym_set = hypernym_set | hypernyms
    # [u'nutriment.n.01', u'food.n.01', u'physical_entity.n.01', u'entity.n.01', u'matter.n.03', u'dish.n.02', u'substance.n.07', u'snack_food.n.01'

    if config.expand_depth:
        vocab_hypernyms[v] = [(h, hypernym2distance[h]) for h in list(hypernyms)]
    else:
        vocab_hypernyms[v] = [h for h in list(hypernyms)]

if config.expand_depth:
    new_vocab_hypernyms = defaultdict(list)

    for v, hypernym_info in vocab_hypernyms.items():
        for hypernym, distance in hypernym_info:
            deepest_depth = max_depth[hypernym]

            for dist in range(distance, deepest_depth + 1):
                new_vocab_hypernyms[v].append(
                    "{}.{}".format(hypernym, dist))

    vocab_hypernyms = new_vocab_hypernyms
    hypernym_set = set([h for h_list in vocab_hypernyms.values() for h in h_list])

hypernym_vocab = [v for v in list(hypernym_set)]
hypernym_dict = {v: i for i, v in enumerate(hypernym_vocab)}

hypernym_wordset = {v: [] for v in hypernym_vocab}
for v, v_hyper in tqdm(vocab_hypernyms.items(), desc='find wordset'):
    for h in v_hyper:
        hypernym_wordset[h].append(v)

for k, v in hypernym_wordset.items():
    if len(v) < config.min_num_word:
        del hypernym_wordset[k]

# add wordset with all answers
hypernym_wordset['all_answers'] = answer_dict['vocab']

wordset_dict = {}
wordset_dict['vocab'] = hypernym_wordset.keys()
wordset_dict['dict'] = {v: i for i, v in enumerate(wordset_dict['vocab'])}

wordset_dict['ans2wordset'] = {answer_dict['dict'][v]: []
                               for v in answer_dict['vocab']}
wordset_dict['wordset2ans'] = {wordset_dict['dict'][k]: set()
                               for k in wordset_dict['vocab']}
for k, v_list in hypernym_wordset.items():
    k_idx = wordset_dict['dict'][k]
    for v in v_list:
        v_idx = answer_dict['dict'][v]
        wordset_dict['ans2wordset'][v_idx].append(k_idx)
        wordset_dict['wordset2ans'][k_idx].add(v_idx)

wordset_dict['ans2shuffled_wordset'] = copy.deepcopy(wordset_dict['ans2wordset'])
for ans in wordset_dict['ans2shuffled_wordset']:
    RANDOM_STATE.shuffle(wordset_dict['ans2shuffled_wordset'][ans])

cPickle.dump(wordset_dict, open(config.save_wordset_path, 'wb'))
print('wordset is saved in : {}'.format(config.save_wordset_path))
