import json
import numpy as np


IMAGE_INFO_FILE = 'VisualGenome/annotations/image_data.json'
IMAGE_SPLIT_FILE = 'preprocessed/visualgenome/image_split.json'

NUM_TRAIN_IMAGE = 80000
NUM_TEST_IMAGE = 18077
NUM_VAL_IMAGE = 10000

RANDOM_STATE = np.random.RandomState(123)

image_info = json.load(open(IMAGE_INFO_FILE, 'r'))

image_ids = [info['image_id'] for info in image_info]

if len(image_ids) != NUM_TRAIN_IMAGE + NUM_TEST_IMAGE + NUM_VAL_IMAGE:
    raise ValueError('num split images doesn not sum to total image id number')

RANDOM_STATE.shuffle(image_ids)

image_split = {
    'train': image_ids[:NUM_TRAIN_IMAGE],
    'test': image_ids[NUM_TRAIN_IMAGE: NUM_TRAIN_IMAGE + NUM_TEST_IMAGE],
    'val': image_ids[NUM_TRAIN_IMAGE + NUM_TEST_IMAGE:]
}
json.dump(image_split, open(IMAGE_SPLIT_FILE, 'w'))
print('Image split is constructed: {}'.format(IMAGE_SPLIT_FILE))
