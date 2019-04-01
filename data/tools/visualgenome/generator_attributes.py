"""
Generator for attributes

Generate attribute data for training vlmap.
"""
import argparse
import h5py
import json
import os
import numpy as np

from collections import Counter
from tqdm import tqdm

import tools

ANNO_DIR = 'VisualGenome/annotations'
ANNO_FILE = {
    'attributes': 'attributes.json',
}

IMAGE_SPLIT_FILE = 'preprocessed/visualgenome/image_split.json'
MIN_CROP_SIZE = 32

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_name', type=str, default='attributes', help=' ')
parser.add_argument('--min_occurrence', type=int, default=20, help=' ')
parser.add_argument('--vocab_path', type=str,
                    default='preprocessed/glove_vocab.json', help=' ')
args = parser.parse_args()

args.dir_name = os.path.join('preprocessed/visualgenome', args.dir_name)
args.dir_name += '_{}_min_occ{}'.format(
    args.vocab_path.replace('preprocessed/', '').replace('.json', ''),
    args.min_occurrence)

if not os.path.exists(args.dir_name):
    os.makedirs(args.dir_name)
else:
    raise ValueError('The directory {} already exists. Do not overwrite'.format(
        args.dir_name))

args.hdf5_file = os.path.join(args.dir_name, 'data.hdf5')
args.ids_file = os.path.join(args.dir_name, 'id.txt')
args.stats_file = os.path.join(args.dir_name, 'stats.txt')
args.attributes_file = os.path.join(args.dir_name, 'attributes.txt')

print('Reading annotations..')
anno = {}
anno['attributes'] = json.load(open(os.path.join(ANNO_DIR, ANNO_FILE['attributes']), 'r'))
print('Done.')

vocab = json.load(open(args.vocab_path, 'r'))
vocab_set = set(vocab['vocab'])


def clean_name(name):
    name = tools.clean_description(name)
    if len(name) > 0 and all([n in vocab_set for n in name.split()]):
        return name
    else: return ''


def check_and_add(name, name_list):
    name = tools.clean_description(name)
    if len(name) > 0 and all([n in vocab_set for n in name.split()]):
        name_list.append(name)


def name2intseq(name):
    return np.array([vocab['dict'][n] for n in name.split()], dtype=np.int32)

attributes = []
for entry in tqdm(anno['attributes'], desc='attributes'):
    for attr in entry['attributes']:
        if 'attributes' in attr:
            for attr_name in attr['attributes']:
                check_and_add(attr_name, attributes)

attribute_count = Counter(attributes)
thr_attributes_set = set([o for o in list(set(attributes))
                          if attribute_count[o] >= args.min_occurrence])
thr_attributes_set_list = list(thr_attributes_set)
thr_attributes_set_idx_dict = {o: i for i, o in
                               enumerate(thr_attributes_set_list)}

f = h5py.File(args.hdf5_file, 'w')
id_file = open(args.ids_file, 'w')

image_split = json.load(open(IMAGE_SPLIT_FILE, 'r'))
train_image_set = set(image_split['train'])
test_image_set = set(image_split['test'])
val_image_set = set(image_split['val'])

num_train_image = len(image_split['train'])
num_test_image = len(image_split['test'])
num_val_image = len(image_split['val'])

train_ids = []
test_ids = []
val_ids = []

cnt = 0
max_name_length = 0
max_num_names = 0
for entry in tqdm(anno['attributes'], desc='attributes'):
    image_id = entry['image_id']
    image_grp = f.create_group(str(image_id))
    for attr in entry['attributes']:
        names = []
        if 'attributes' in attr:
            for attr_name in attr['attributes']:
                names.append(attr_name)

        name_len = []
        names_intseq = []
        name_ids = []
        for name in names:
            name = clean_name(name)
            if name == '' or name not in thr_attributes_set:
                continue
            intseq = name2intseq(name)
            name_len.append(len(intseq))
            names_intseq.append(intseq)
            name_ids.append(thr_attributes_set_idx_dict[name])

        if len(names_intseq) == 0:
            continue

        if attr['h'] < MIN_CROP_SIZE or attr['w'] < MIN_CROP_SIZE:
            continue

        names = np.zeros([len(names_intseq), max(name_len)], dtype=np.int32)
        for i, intseq in enumerate(names_intseq):
            names[i][:len(intseq)] = intseq
        name_len = np.array(name_len, dtype=np.int32)
        name_ids = np.array(name_ids, dtype=np.int32)

        max_num_names = max(max_num_names, names.shape[0])
        max_name_length = max(max_name_length, names.shape[1])

        id = 'attributes{:08d}_imageid{}_numname{}_maxnamelen{}'.format(
            cnt, image_id, names.shape[0], names.shape[1])

        grp = image_grp.create_group(id)
        grp['image_id'] = image_id
        grp['names'] = names
        grp['name_len'] = name_len
        grp['name_ids'] = name_ids
        grp['h'], grp['w'] = attr['h'], attr['w']
        grp['y'], grp['x'] = attr['y'], attr['x']
        grp['object_id'] = attr['object_id']

        id_str = str(image_id) + ' ' + id + '\n'
        if image_id in train_image_set: train_ids.append(id_str)
        elif image_id in test_image_set: test_ids.append(id_str)
        elif image_id in val_image_set: val_ids.append(id_str)
        else: raise ValueError('Unknown image_id')
        cnt += 1

num_train = len(train_ids)
num_test = len(test_ids)
num_val = len(val_ids)
for id_str in train_ids + test_ids + val_ids:
    id_file.write(id_str)

thr_attribute_set_intseq = np.zeros([len(thr_attributes_set), max_name_length],
                                    dtype=np.int32)
thr_attribute_set_intseq_len = np.zeros([len(thr_attributes_set)], dtype=np.int32)
for i, name in enumerate(thr_attributes_set_list):
    intseq = name2intseq(name)
    thr_attribute_set_intseq[i, :len(intseq)] = intseq
    thr_attribute_set_intseq_len[i] = len(intseq)

grp = f.create_group('data_info')
grp['max_name_length'] = max_name_length
grp['max_num_names'] = max_num_names
grp['num_data'] = cnt
grp['num_train'] = num_train
grp['num_test'] = num_test
grp['num_val'] = num_val
grp['num_images'] = len(anno['attributes'])
grp['num_train_image'] = num_train_image
grp['num_test_image'] = num_test_image
grp['num_val_image'] = num_val_image
grp['num_unique_attributes'] = len(thr_attributes_set)
grp['attributes_intseq'] = thr_attribute_set_intseq
grp['attributes_intseq_len'] = thr_attribute_set_intseq_len
grp['min_occurrence'] = args.min_occurrence

id_file.close()
f.close()

stat_file = open(args.stats_file, 'w')
stat_file.write('num_data: {}\n'.format(cnt))
stat_file.write('num_train: {}\n'.format(num_train))
stat_file.write('num_test: {}\n'.format(num_test))
stat_file.write('num_val: {}\n'.format(num_val))
stat_file.write('num_images: {}\n'.format(len(anno['attributes'])))
stat_file.write('num_train_image: {}\n'.format(num_train_image))
stat_file.write('num_test_image: {}\n'.format(num_test_image))
stat_file.write('num_val_image: {}\n'.format(num_val_image))
stat_file.write('num_unique_attributes: {}\n'.format(len(thr_attributes_set)))
stat_file.write('max_num_names: {}\n'.format(max_num_names))
stat_file.write('max_name_length: {}\n'.format(max_name_length))
stat_file.write('min_occurrence: {}\n'.format(args.min_occurrence))
stat_file.close()

attributes_file = open(args.attributes_file, 'w')
for name in list(thr_attributes_set):
    attributes_file.write(name + '\n')
attributes_file.close()

print('Attribute dataset is created: {}'.format(args.dir_name))
