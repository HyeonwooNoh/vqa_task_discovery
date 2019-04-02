import argparse
import cPickle
import h5py
import os
import tensorflow as tf

from util import log
from misc import modules

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
parser.add_argument('--class_feat_dim', type=int, default=2048, help=' ')
parser.add_argument('--checkpoint', type=str, required=True, help='ex) ./model-1')
config = parser.parse_args()

ckpt_dir = os.path.dirname(config.checkpoint)
ckpt_name = os.path.basename(config.checkpoint)
config.save_dir = os.path.join(ckpt_dir, 'word_weights_{}'.format(ckpt_name))
if not os.path.exists(config.save_dir):
    log.warn('create directory: {}'.format(config.save_dir))
    os.makedirs(config.save_dir)
else:
    raise ValueError('Do not overwrite: {}'.format(config.save_dir))

vocab_path = os.path.join(config.data_dir, 'vocab.pkl')
vocab = cPickle.load(open(vocab_path, 'rb'))

answer_dict_path = os.path.join(config.data_dir, 'answer_dict.pkl')
answer_dict = cPickle.load(open(answer_dict_path, 'rb'))
num_answer = len(answer_dict['vocab'])

v_word_map = modules.LearnGloVe(vocab, scope='V_GloVe')
l_word_map = modules.LearnGloVe(vocab, scope='L_GloVe')
l_answer_word_map = modules.LearnAnswerGloVe(answer_dict)

with tf.variable_scope('classifier_v/fc', reuse=tf.AUTO_REUSE):
    # (float32_ref 2048x4000) [8192000, bytes: 32768000]
    v_class_weights = tf.get_variable(
        'weights', shape=[config.class_feat_dim, num_answer])
    # (float32_ref 4000) [4000, bytes: 16000]
    v_class_biases = tf.get_variable(
        'biases', shape=[num_answer])

with tf.variable_scope('classifier_l/fc', reuse=tf.AUTO_REUSE):
    # (float32_ref 2048x4000) [8192000, bytes: 32768000]
    l_class_weights = tf.get_variable(
        'weights', shape=[config.class_feat_dim, num_answer])
    # (float32_ref 4000) [4000, bytes: 16000]
    l_class_biases = tf.get_variable(
        'biases', shape=[num_answer])

session_config = tf.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(allow_growth=True),
    device_count={'GPU': 1})
sess = tf.Session(config=session_config)

all_vars = tf.global_variables()
checkpoint_loader = tf.train.Saver(var_list=all_vars, max_to_keep=1)

log.info('Checkpoint path: {}'.format(config.checkpoint))
checkpoint_loader.restore(sess, config.checkpoint)
log.info('Loaded the checkpoint')

word_values_dict = {
    'v_word': v_word_map,
    'l_word': l_word_map,
    'l_answer_word': l_answer_word_map,
    'v_class_weights': v_class_weights,
    'v_class_biases': v_class_biases,
    'l_class_weights': l_class_weights,
    'l_class_biases': l_class_biases,
}
word_values = sess.run(word_values_dict)

h5_path = os.path.join(config.save_dir, 'weights.hdf5')
f = h5py.File(h5_path, 'w')
for key, val in word_values.items():
    f[key] = val
f.close()
log.warn('weights are saved in: {}'.format(h5_path))

save_vocab_path = os.path.join(config.save_dir, 'vocab.pkl')
cPickle.dump(vocab, open(save_vocab_path, 'wb'))
log.warn('vocab is saved in: {}'.format(save_vocab_path))

save_answer_dict_path = os.path.join(config.save_dir, 'answer_dict.pkl')
cPickle.dump(answer_dict, open(save_answer_dict_path, 'wb'))
log.warn('answer_dict is saved in: {}'.format(save_answer_dict_path))

log.warn('done')
