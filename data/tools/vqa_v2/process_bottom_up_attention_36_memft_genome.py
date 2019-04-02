import argparse
import base64
import csv
import cPickle
import h5py
import os
import sys
import numpy as np

from tqdm import tqdm

from util import box_utils, log

BOTTOM_UP_FILE_NAMES = [
    'trainval/trainval_resnet101_faster_rcnn_genome_36.tsv',
]
NUM_BOXES = 36
FEATURE_DIM = 2048

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--tf_record_memft_dir', type=str, default='data/preprocessed/vqa_v2'
    '/qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1'
    '/tf_record_memft', help=' ')
parser.add_argument('--bottom_up_dir', type=str,
                    default='data/VQA_v2/bottom_up_attention_36', help=' ')
config = parser.parse_args()

image_info_path = os.path.join(config.tf_record_memft_dir, 'image_info.pkl')
log.infov('loading image_info: {}'.format(image_info_path))
image_info = cPickle.load(open(image_info_path, 'rb'))
log.infov('done')

image_id2idx = image_info['image_id2idx']
image_path2idx = image_info['image_path2idx']
image_num2path = image_info['image_num2path']

csv.field_size_limit(sys.maxsize)
int_field = ['image_id', 'image_w', 'image_h', 'num_boxes']
np_field = ['boxes', 'features']

config.vfeat_path = os.path.join(config.tf_record_memft_dir,
                                 'vfeat_bottomup_36.hdf5')
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

    tsv_in_file = open(os.path.join(config.bottom_up_dir, file_name), 'r+b')

    reader = csv.DictReader(tsv_in_file, delimiter='\t',
                            fieldnames=(int_field + np_field))
    for item in tqdm(reader, desc='processing reader', total=len(image_id2idx)):
        for field in int_field:
            item[field] = int(item[field])
        for field in np_field:
            item[field] = np.frombuffer(
                base64.decodestring(item[field]),
                dtype=np.float32).reshape((item['num_boxes'], -1))

        image_num = item['image_id']
        image_path = image_num2path[image_num]

        image_idx = image_path2idx[image_path]

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
