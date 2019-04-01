"""
Generator for region_descriptions

Generate region_description data for training vlmap.
"""
import argparse
import h5py
import json
import os
import numpy as np

from tqdm import tqdm

import tools

ANNO_DIR = 'VisualGenome/annotations'
ANNO_FILE = {
    'region_descriptions': 'region_descriptions.json',
}

IMAGE_SPLIT_FILE = 'preprocessed/visualgenome/image_split.json'
MIN_CROP_SIZE = 32

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_name', type=str, default='region_descriptions',
                    help=' ')
parser.add_argument('--vocab_path', type=str,
                    default='preprocessed/glove_vocab.json', help=' ')
parser.add_argument('--max_description_length', type=int, default=10, help=' ')
args = parser.parse_args()

args.dir_name = os.path.join('preprocessed/visualgenome', args.dir_name)
args.dir_name += '_{}'.format(
    args.vocab_path.replace('preprocessed/', '').replace('.json', ''))
if args.max_description_length > 0:
    args.dir_name += '_max_len{}'.format(args.max_description_length)

if not os.path.exists(args.dir_name):
    os.makedirs(args.dir_name)
else:
    raise ValueError('The directory {} already exists. Do not overwrite'.format(
        args.dir_name))

args.hdf5_file = os.path.join(args.dir_name, 'data.hdf5')
args.ids_file = os.path.join(args.dir_name, 'id.txt')
args.stats_file = os.path.join(args.dir_name, 'stats.txt')
args.descriptions_file = os.path.join(args.dir_name, 'descriptions.txt')

print('Reading annotations..')
anno = {}
anno['region_descriptions'] = \
    json.load(open(os.path.join(ANNO_DIR,
                                ANNO_FILE['region_descriptions']), 'r'))
print('Done.')

vocab = json.load(open(args.vocab_path, 'r'))
vocab_set = set(vocab['vocab'])


def clean_phrase(phrase):
    phrase = tools.clean_description(phrase)
    if len(phrase) > 0 and all([n in vocab_set for n in phrase.split()]):
        return phrase
    else: return ''


def phrase2intseq(phrase):
    return np.array([vocab['dict'][n] for n in phrase.split()], dtype=np.int32)

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
max_length = 0
descriptions = []
for entry in tqdm(anno['region_descriptions'], desc='region_descriptions'):
    for region in entry['regions']:
        if region['height'] < MIN_CROP_SIZE or region['width'] < MIN_CROP_SIZE:
            continue

        phrase = region['phrase']
        phrase = clean_phrase(phrase)
        if phrase == '': continue
        descriptions.append(phrase)

        phrase = np.array(phrase2intseq(phrase), dtype=np.int32)

        len_limit = args.max_description_length
        if len_limit > 0 and len(phrase) > len_limit: continue

        max_length = max(max_length, len(phrase))

        image_id = region['image_id']
        id = 'descriptions{:08d}_imageid{}_length{}'.format(
            cnt, image_id, len(phrase))

        if str(image_id) in f: image_grp = f[str(image_id)]
        else: image_grp = f.create_group(str(image_id))

        grp = image_grp.create_group(id)
        grp['image_id'] = image_id
        grp['description'] = phrase
        grp['region_id'] = region['region_id']
        grp['x'], grp['y'] = region['x'], region['y']
        grp['w'], grp['h'] = region['width'], region['height']

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

set_descriptions = list(set(descriptions))

grp = f.create_group('data_info')
grp['max_length'] = max_length
grp['num_data'] = cnt
grp['num_train'] = num_train
grp['num_test'] = num_test
grp['num_val'] = num_val
grp['num_images'] = len(anno['region_descriptions'])
grp['num_train_image'] = num_train_image
grp['num_test_image'] = num_test_image
grp['num_val_image'] = num_val_image
grp['num_unique_descriptions'] = len(set_descriptions)

id_file.close()
f.close()

stat_file = open(args.stats_file, 'w')
stat_file.write('num_data: {}\n'.format(cnt))
stat_file.write('num_train: {}\n'.format(num_train))
stat_file.write('num_test: {}\n'.format(num_test))
stat_file.write('num_val: {}\n'.format(num_val))
stat_file.write('num_images: {}\n'.format(len(anno['region_descriptions'])))
stat_file.write('num_train_image: {}\n'.format(num_train_image))
stat_file.write('num_test_image: {}\n'.format(num_test_image))
stat_file.write('num_val_image: {}\n'.format(num_val_image))
stat_file.write('num_unique_descriptions: {}\n'.format(len(set_descriptions)))
stat_file.write('max_length: {}\n'.format(max_length))
stat_file.close()

descriptions_file = open(args.descriptions_file, 'w')
for name in set_descriptions:
    descriptions_file.write(name.encode('utf-8') + '\n')
descriptions_file.close()

print('description dataset is created: {}'.format(args.dir_name))
