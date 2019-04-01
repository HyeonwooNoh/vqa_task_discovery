import cPickle
import h5py
import os
import numpy as np
import multiprocessing
import tensorflow as tf
from collections import namedtuple, defaultdict

from util import log, get_dummy_data

NUM_CONFIG = {
    'attr_blank_fill': 5,
    'obj_blank_fill': 5,
}

CPU_COUNT = multiprocessing.cpu_count()


class Dataset(object):
    def __init__(self, config, split, name='pretrain'):
        self.config = config
        self.data_dir = data_dir = config.data_dir

        self.name = name
        self.split = split

        log.warn('loading image_info ..')
        image_info_path = os.path.join(
            data_dir, '{}_image_info.pkl'.format(split))
        image_info = cPickle.load(open(image_info_path, 'rb'))
        self._ids = image_info['image_ids']
        self.image_id2idx = image_info['image_id2idx']
        log.info('loading image_info done')

        log.warn('loading processed data ..')
        processed_path = os.path.join(
            data_dir, '{}_processed.pkl'.format(split))
        self.processed = cPickle.load(open(processed_path, 'rb'))
        log.info('loading processed done')

        log.warn('loading answer_dict ..')
        answer_dict_path = os.path.join(data_dir, 'answer_dict.pkl')
        self.answer_dict = cPickle.load(open(answer_dict_path, 'rb'))
        self.num_answers = len(self.answer_dict['vocab'])
        log.info('loading answer_dict done')

        log.warn('loading wordset_dict ..')
        ws_dict_path = os.path.join(data_dir, config.wordset_dict_name)
        self.ws_dict = cPickle.load(open(ws_dict_path, 'rb'))
        log.info('loading wordset_dict: {} done'.format(ws_dict_path))

        if self.config.debug:
            log.warn('Debug mode: proceed with dummy data')
            self.image_features, self.spatial_features, self.normal_boxes, self.num_boxes, \
                self.max_box_num, self.vfeat_dim = get_dummy_data()
        else:
            log.warn('No debug mode: proceed with real data')
            with h5py.File(os.path.join(data_dir, '{}_vfeat.hdf5'.format(split)), 'r') as f:

                self.vfeat_dim = int(f['data_info']['vfeat_dim'].value)
                self.max_box_num = int(f['data_info']['max_box_num'].value)
                log.warn('loading {} image_features ..'.format(split))
                self.image_features = np.array(f.get('image_features'))
                log.warn('loading {} normal_boxes ..'.format(split))
                self.normal_boxes = np.array(f.get('normal_boxes'))
                log.warn('loading {} num_boxes ..'.format(split))
                self.num_boxes = np.array(f.get('num_boxes'))
                log.warn('loading {} spatial_features ..'.format(split))
                self.spatial_features = np.array(f.get('spatial_features'))
                log.warn('loading {} features done ..'.format(split))

        self.wordset_choice_idx = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)))

        self.index_list_idx = {
            'obj_blank_fill': defaultdict(int),
            'attr_blank_fill': defaultdict(int),
        }

        log.info('dataset {} {} init done'.format(name, split))

    def get_config(self):
        config = namedtuple('dataset_config', [])
        config.n_attr_bf = NUM_CONFIG['attr_blank_fill']
        config.n_obj_bf = NUM_CONFIG['obj_blank_fill']
        config.vfeat_dim = self.vfeat_dim
        config.max_box_num = self.max_box_num
        return config

    def sample_wordset_and_context_idx(self, e, category, task, image_idx, idx):
        label = e[task]

        wordsets = self.ws_dict['ans2shuffled_wordset'][label]

        wordset_choice_idx = self.wordset_choice_idx[category][task][label]

        wordset = wordsets[wordset_choice_idx]

        self.wordset_choice_idx[category][task][label] += 1

        if self.wordset_choice_idx[category][task][label] >= len(wordsets):
            np.random.shuffle(
                self.ws_dict['ans2shuffled_wordset'][label])
            self.wordset_choice_idx[category][task][label] = 0

        return wordset

    def get_data(self, image_id):
        image_idx = self.image_id2idx[image_id]
        image_ft = self.image_features[image_idx]
        spatial_ft = self.spatial_features[image_idx]
        normal_box = self.normal_boxes[image_idx]
        num_box = self.num_boxes[image_idx]

        ret = {
            'image_id': np.array(image_id, dtype=np.int32),
            'image_ft': image_ft,
            'spatial_ft': spatial_ft,
            'normal_boxes': normal_box,
            'num_boxes': num_box
        }

        entry = self.processed[image_id]
        """
        object_blank_fill
        """
        idx_list = list(range(len(entry['obj_blank_fill'])))
        np.random.shuffle(idx_list)
        idx_list = idx_list[:NUM_CONFIG['obj_blank_fill']]
        num_valid_data = len(idx_list)
        while len(idx_list) < NUM_CONFIG['obj_blank_fill']:
            idx_list.append(idx_list[-1])
        maxlen = max([len(entry['obj_blank_fill'][idx]['blank'])
                      for idx in idx_list])
        weights, normal_boxes, fills, blanks_len = [], [], [], []
        wordsets = []
        blanks = np.zeros([len(idx_list), maxlen], dtype=np.int32)
        for i, idx in enumerate(idx_list):
            e = entry['obj_blank_fill'][idx]
            weight = np.zeros([self.max_box_num], dtype=np.float32)
            weight[e['p_idx']] = e['p_weight']
            weights.append(weight)
            normal_boxes.append(e['normal_box'])
            blanks_len.append(len(e['blank']))
            blanks[i, :blanks_len[i]] = e['blank']
            fills.append(e['fill'])

            wordset = self.sample_wordset_and_context_idx(
                e, 'obj', 'fill', image_idx, idx)
            wordsets.append(wordset)

        ret.update({
            'obj_blank_fill/num': np.array(num_valid_data, dtype=np.int32),
            'obj_blank_fill/weights': np.array(weights, dtype=np.float32),
            'obj_blank_fill/normal_boxes': np.array(
                normal_boxes, dtype=np.float32),
            'obj_blank_fill/fills': np.array(fills, dtype=np.int32),
            'obj_blank_fill/blanks': blanks,
            'obj_blank_fill/blanks_len': np.array(blanks_len, dtype=np.int32),
            'obj_blank_fill/wordsets': np.array(wordsets, dtype=np.int32),
        })

        """
        attribute_blank_fill
        """

        idx_list = list(range(len(entry['attr_blank_fill'])))
        np.random.shuffle(idx_list)
        idx_list = idx_list[:NUM_CONFIG['attr_blank_fill']]
        num_valid_data = len(idx_list)
        while len(idx_list) < NUM_CONFIG['attr_blank_fill']:
            idx_list.append(idx_list[-1])
        maxlen = max([len(entry['attr_blank_fill'][idx]['blank'])
                      for idx in idx_list])
        weights, normal_boxes, fills, blanks_len = [], [], [], []
        wordsets = []
        blanks = np.zeros([len(idx_list), maxlen], dtype=np.int32)
        for i, idx in enumerate(idx_list):
            e = entry['attr_blank_fill'][idx]
            weight = np.zeros([self.max_box_num], dtype=np.float32)
            weight[e['p_idx']] = e['p_weight']
            weights.append(weight)
            normal_boxes.append(e['normal_box'])
            blanks_len.append(len(e['blank']))
            blanks[i, :blanks_len[i]] = e['blank']
            fills.append(e['fill'])

            wordset = self.sample_wordset_and_context_idx(
                e, 'attr', 'fill', image_idx, idx)
            wordsets.append(wordset)

        ret.update({
            'attr_blank_fill/num': np.array(num_valid_data, dtype=np.int32),
            'attr_blank_fill/weights': np.array(weights, dtype=np.float32),
            'attr_blank_fill/normal_boxes': np.array(
                normal_boxes, dtype=np.float32),
            'attr_blank_fill/fills': np.array(fills, dtype=np.int32),
            'attr_blank_fill/blanks': blanks,
            'attr_blank_fill/blanks_len': np.array(blanks_len, dtype=np.int32),
            'attr_blank_fill/wordsets': np.array(wordsets, dtype=np.int32),
        })
        return ret

    def get_shapes(self):
        ret_shapes = {
            'image_id': (),
            'image_ft': [self.max_box_num, self.vfeat_dim],
            'spatial_ft': [self.max_box_num, 6],
            'normal_boxes': [self.max_box_num, 4],
            'num_boxes': (),
            'obj_blank_fill/num': (),
            'obj_blank_fill/weights': [NUM_CONFIG['obj_blank_fill'], self.max_box_num],
            'obj_blank_fill/normal_boxes': [NUM_CONFIG['obj_blank_fill'], 4],
            'obj_blank_fill/fills': [NUM_CONFIG['obj_blank_fill']],
            'obj_blank_fill/blanks': [NUM_CONFIG['obj_blank_fill'], None],
            'obj_blank_fill/blanks_len': [NUM_CONFIG['obj_blank_fill']],
            'obj_blank_fill/wordsets': [NUM_CONFIG['obj_blank_fill']],
            'attr_blank_fill/num': (),
            'attr_blank_fill/weights': [NUM_CONFIG['attr_blank_fill'], self.max_box_num],
            'attr_blank_fill/normal_boxes': [NUM_CONFIG['attr_blank_fill'], 4],
            'attr_blank_fill/fills': [NUM_CONFIG['attr_blank_fill']],
            'attr_blank_fill/blanks': [NUM_CONFIG['attr_blank_fill'], None],
            'attr_blank_fill/blanks_len': [NUM_CONFIG['attr_blank_fill']],
            'attr_blank_fill/wordsets': [NUM_CONFIG['attr_blank_fill']],
        }
        return ret_shapes

    def get_types(self):
        ret_type = {
            'image_id': tf.int32,
            'image_ft': tf.float32,
            'spatial_ft': tf.float32,
            'normal_boxes': tf.float32,
            'num_boxes': tf.int32,
            'obj_blank_fill/num': tf.int32,
            'obj_blank_fill/weights': tf.float32,
            'obj_blank_fill/normal_boxes': tf.float32,
            'obj_blank_fill/fills': tf.int32,
            'obj_blank_fill/blanks': tf.int32,
            'obj_blank_fill/blanks_len': tf.int32,
            'obj_blank_fill/wordsets': tf.int32,
            'attr_blank_fill/num': tf.int32,
            'attr_blank_fill/weights': tf.float32,
            'attr_blank_fill/normal_boxes': tf.float32,
            'attr_blank_fill/fills': tf.int32,
            'attr_blank_fill/blanks': tf.int32,
            'attr_blank_fill/blanks_len': tf.int32,
            'attr_blank_fill/wordsets': tf.int32,
        }
        return ret_type

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset ({} [{}], {} examples)'.format(
            self.name, self.split, len(self))


def create_ops(batch_size,
               dataset,
               is_train=True,
               scope='pretrain',
               shuffle=True):

    with tf.device('/cpu:0'), tf.name_scope(scope):
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(dataset.ids))

        if is_train and shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=3000)

        def load_fn(image_id):
            ret = dataset.get_data(image_id)
            ret_list = [ret[key] for key in sorted(ret.keys())]
            return ret_list

        def load_pyfunc(image_id):
            ret_type = dataset.get_types()
            ret_type_list = [ret_type[key] for key in sorted(ret_type.keys())]
            pyfunc_ret_list = tf.py_func(load_fn, inp=[image_id],
                                         Tout=ret_type_list, name='input_pyfunc')
            pyfunc_ret = {key: val for key, val in
                          zip(sorted(ret_type.keys()), pyfunc_ret_list)}
            shapes = dataset.get_shapes()
            for key, shape in shapes.items():
                pyfunc_ret[key].set_shape(shape)
            return pyfunc_ret

        tf_dataset = tf_dataset.map(load_pyfunc)

    tf_dataset = tf_dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=dataset.get_shapes()
    )

    tf_dataset = tf_dataset.prefetch(max(CPU_COUNT-5, 0) or None)

    if is_train:
        tf_dataset = tf_dataset.repeat(1000)  # repeat 1000 epoch

    iterator = tf_dataset.make_one_shot_iterator()
    batch_ops = iterator.get_next()

    return batch_ops
