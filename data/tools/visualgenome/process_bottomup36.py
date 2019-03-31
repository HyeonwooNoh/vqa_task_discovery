import argparse
import base64
import csv
import h5py
import json
import os
import sys
import numpy as np

from tqdm import tqdm
from util import box_utils, log

BOTTOM_UP_FILE_NAMES = [
    'genome_all_resnet101_faster_rcnn_genome.tsv.0',
    'genome_all_resnet101_faster_rcnn_genome.tsv.1',
    'genome_all_resnet101_faster_rcnn_genome.tsv.2',
]
NUM_BOXES = 36
FEATURE_DIM = 2048

IMAGE_DATA_PATH = 'data/VisualGenome/annotations/image_data.json'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--bottomup_data_dir', type=str,
                    default='data/VisualGenome/bottomup_feature_36', help=' ')
config = parser.parse_args()

image_data = json.load(open(IMAGE_DATA_PATH, 'r'))

config.vfeat_path = os.path.join(config.bottomup_data_dir,
                                 'vfeat_bottomup_36.hdf5')

image_ids = [entry['image_id'] for entry in image_data]
image_id2idx = {id: i for i, id in enumerate(image_ids)}

image_info = {
    'image_id2idx': image_id2idx,
}
json.dump(image_info, open(os.path.join(config.bottomup_data_dir,
                                        'image_info.json'), 'w'))


csv.field_size_limit(sys.maxsize)
int_field = ['image_id', 'image_w', 'image_h', 'num_boxes']
np_field = ['boxes', 'features']

f = h5py.File(config.vfeat_path, 'w')

image_features = f.create_dataset(
    'image_features', (len(image_id2idx), NUM_BOXES, FEATURE_DIM), 'f')
normal_boxes = f.create_dataset(
    'normal_boxes', (len(image_id2idx), NUM_BOXES, 4), 'f')
num_boxes = np.zeros([len(image_id2idx)], dtype=np.int32)
spatial_features = f.create_dataset(
    'spatial_features', (len(image_id2idx), NUM_BOXES, 6), 'f')

for file_name in BOTTOM_UP_FILE_NAMES:
    log.warn('process: {}'.format(file_name))
    tsv_in_file = open(os.path.join(config.bottomup_data_dir, file_name), 'r+b')
    reader = csv.DictReader(tsv_in_file, delimiter='\t',
                            fieldnames=(int_field + np_field))
    for item in tqdm(reader, desc='processing reader', total=len(image_id2idx)):
        for field in int_field:
            item[field] = int(item[field])
        for field in np_field:
            item[field] = np.frombuffer(
                base64.decodestring(item[field]),
                dtype=np.float32).reshape((item['num_boxes'], -1))

        image_id = item['image_id']
        image_idx = image_id2idx[image_id]

        vfeat = item['features'].astype(np.float32)
        image_features[image_idx, :, :] = vfeat  # add to hdf5

        box_x1y1x2y2 = item['boxes'].astype(np.float32)
        image_w, image_h = float(item['image_w']), float(item['image_h'])
        normal_box = box_utils.normalize_boxes_x1y1x2y2(box_x1y1x2y2,
                                                        image_w, image_h)
        normal_boxes[image_idx, :, :] = normal_box  # add to hdf5
        num_boxes[image_idx] = item['num_boxes']  # add to hdf5

        ft_x1 = normal_box[:, 0]
        ft_y1 = normal_box[:, 1]
        ft_x2 = normal_box[:, 2]
        ft_y2 = normal_box[:, 3]
        ft_w = ft_x2 - ft_x1
        ft_h = ft_y2 - ft_y1

        spatial_features[image_idx, :, :] = np.stack(
            [ft_x1, ft_y1, ft_x2, ft_y2, ft_w, ft_h], axis=1)

f['num_boxes'] = num_boxes

data_info = f.create_group('data_info')
data_info['vfeat_dim'] = FEATURE_DIM
data_info['max_box_num'] = NUM_BOXES
data_info['pretrained_param_path'] = 'bottom_up_attention_36'

f.close()
