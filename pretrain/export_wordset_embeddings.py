import argparse
import cPickle
import h5py
import os
import tensorflow as tf

from util import log
from misc import modules


def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
parser.add_argument('--class_feat_dim', type=int, default=2048, help=' ')
parser.add_argument('--expand_depth', type=str2bool, default=False, help='whether to expand wordset based on deepest depth')
parser.add_argument('--checkpoint', type=str, required=True, help='ex) ./model-1')
config = parser.parse_args()

ckpt_dir = os.path.dirname(config.checkpoint)
ckpt_name = os.path.basename(config.checkpoint)
config.save_dir = os.path.join(ckpt_dir, 'wordset_embeddings_{}'.format(ckpt_name))
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

ws_dict_path = os.path.join(config.data_dir, 'wordset_dict5_depth{}.pkl'.format(
    int(config.expand_depth)))
ws_dict = cPickle.load(open(ws_dict_path, 'rb'))
num_ws = len(ws_dict['vocab'])

wordset_map = modules.learn_embedding_map(
    ws_dict, scope='wordset_map')

L_DIM = 1024

wordset_embed = tf.tanh(wordset_map)
wordset_ft = modules.fc_layer(
    wordset_embed, L_DIM, use_bias=True, use_bn=False, use_ln=True,
    activation_fn=tf.tanh, is_training=False, scope='wordset_ft')

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
    'wordset_ft': wordset_ft,
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

save_ws_dict_path = os.path.join(config.save_dir, 'wordset_dict5_depth{}.pkl'.format(
    int(config.expand_depth)))
cPickle.dump(ws_dict, open(save_ws_dict_path, 'wb'))
log.warn('ws_dict is saved in: {}'.format(save_ws_dict_path))

save_answer_dict_path = os.path.join(config.save_dir, 'answer_dict.pkl')
cPickle.dump(answer_dict, open(save_answer_dict_path, 'wb'))
log.warn('answer_dict is saved in: {}'.format(save_answer_dict_path))

log.warn('done')
