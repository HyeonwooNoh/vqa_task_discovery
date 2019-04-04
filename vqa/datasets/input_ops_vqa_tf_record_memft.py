import h5py
import os
import tensorflow as tf


def create(batch_size,
           tf_record_dir,
           split,
           is_train=True,
           scope='vqa_tf_record',
           shuffle=True):

    tf_record_info_path = os.path.join(tf_record_dir, 'data_info.hdf5')
    with h5py.File(tf_record_info_path, 'r') as f:
        num_answers = int(f['data_info']['num_answers'].value)

    tf_record_path = os.path.join(tf_record_dir, split, '{}-*'.format(split))
    with tf.device('/cpu:0'):
        files = tf.data.Dataset.list_files(tf_record_path)

        dataset = files.apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=10, block_length=1))

        if is_train and shuffle:
            dataset = dataset.shuffle(buffer_size=3000)

        def parse_fn(example):
            example_fmt = {
                'qid': tf.FixedLenFeature((), tf.int64, -1),
                'image_id': tf.FixedLenFeature((), tf.string, ""),
                'image_idx': tf.FixedLenFeature((), tf.int64, -1),
                'q_intseq/list': tf.FixedLenSequenceFeature(
                    (), tf.int64, allow_missing=True),
                'q_intseq/len': tf.FixedLenFeature((), tf.int64),
                'answers/ids': tf.FixedLenSequenceFeature(
                    (), tf.int64, allow_missing=True),
                'answers/scores': tf.FixedLenSequenceFeature(
                    (), tf.float32, allow_missing=True),
            }
            parsed = tf.parse_single_example(example, example_fmt)

            parsed['q_intseq/list'] = tf.cast(parsed['q_intseq/list'], tf.int32)
            parsed['q_intseq/len'] = tf.cast(parsed['q_intseq/len'], tf.int32)
            parsed['answers/ids'] = tf.cast(parsed['answers/ids'], tf.int32)

            inputs = {
                'id': parsed['qid'],
                'image_id': parsed['image_id'],
                'image_idx': parsed['image_idx'],
                'q_intseq': parsed['q_intseq/list'],
                'q_intseq_len': parsed['q_intseq/len'],
                'answer_target': tf.sparse_to_dense(
                    parsed['answers/ids'], [num_answers],
                    parsed['answers/scores'], default_value=0,
                    validate_indices=False),
            }
            inputs['q_intseq'].set_shape([None])
            return inputs

        dataset = dataset.map(map_func=parse_fn)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes={
                'id': (),
                'image_id': (),
                'image_idx': (),
                'q_intseq': [None],
                'q_intseq_len': (),
                'answer_target': [num_answers],
            })
        if is_train:
            dataset = dataset.cache()  # cache to memory

        dataset = dataset.prefetch(buffer_size=10)

        if is_train:
            dataset = dataset.repeat(1000)
        iterator = dataset.make_one_shot_iterator()
        batch_ops = iterator.get_next()

        return batch_ops
