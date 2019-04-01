import argparse
import cPickle
import h5py
import json
import os
import numpy as np

from collections import Counter
from tqdm import tqdm

from data.tools import tools
from util import log, box_utils

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--bottomup_data_dir', type=str,
                    default='data/VisualGenome/bottomup_feature_36', help=' ')
parser.add_argument('--dir_name', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10',
                    help=' ')
config = parser.parse_args()

config.vfeat_path = os.path.join(config.bottomup_data_dir,
                                 'vfeat_bottomup_36.hdf5')
config.image_info_path = os.path.join(config.bottomup_data_dir,
                                      'image_info.json')
vfeat_h5 = h5py.File(config.vfeat_path, 'r')
data_info = vfeat_h5['data_info']
vfeat_dim = int(data_info['vfeat_dim'].value)
max_box_num = int(data_info['max_box_num'].value)

log.warn('loading image features...')
full_image_features = np.array(vfeat_h5.get('image_features'))
log.warn('loading normal boxes...')
full_normal_boxes = np.array(vfeat_h5.get('normal_boxes'))
log.warn('loading num boxes...')
full_num_boxes = np.array(vfeat_h5.get('num_boxes'))
log.warn('loading spatial features...')
full_spatial_features = np.array(vfeat_h5.get('spatial_features'))
log.warn('loading features are done')

full_image_info = json.load(open(config.image_info_path, 'r'))
full_image_id2idx = full_image_info['image_id2idx']

for split in ['train', 'val']:
    image_info = cPickle.load(open(
        os.path.join(config.dir_name, '{}_image_info.pkl'.format(split)), 'rb'))
    image_id2idx = image_info['image_id2idx']
    f = h5py.File(
        os.path.join(config.dir_name, '{}_vfeat.hdf5'.format(split)), 'w')
    f_data_info = f.create_group('data_info')
    f_data_info['vfeat_dim'] = vfeat_dim
    f_data_info['max_box_num'] = max_box_num
    f_data_info['pretrained_param_path'] = 'bottom_up_attention_36_{}'.format(
        split)
    image_features = f.create_dataset(
        'image_features', (len(image_id2idx), max_box_num, vfeat_dim), 'f')
    normal_boxes = f.create_dataset(
        'normal_boxes', (len(image_id2idx), max_box_num, 4), 'f')
    num_boxes = np.zeros([len(image_id2idx)], dtype=np.int32)
    spatial_features = f.create_dataset(
        'spatial_features', (len(image_id2idx), max_box_num, 6), 'f')

    for image_id, image_idx in tqdm(image_id2idx.items(), desc='process {}'.format(split)):
        full_image_idx = full_image_id2idx[str(image_id)]

        image_features[image_idx, :, :] = \
            full_image_features[full_image_idx, :, :]
        normal_boxes[image_idx, :, :] = \
            full_normal_boxes[full_image_idx, :, :]
        num_boxes[image_idx] = full_num_boxes[full_image_idx]
        spatial_features[image_idx, :, :] = \
            full_spatial_features[full_image_idx, :, :]

    f['num_boxes'] = num_boxes
    f.close()
