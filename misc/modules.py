import cPickle
import json
import h5py
import math
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from util import log

GLOVE_EMBEDDING_PATH = 'data/preprocessed/glove.6B.300d.hdf5'
GLOVE_VOCAB_PATH = 'data/preprocessed/glove_vocab.json'
ENC_I_R_MEAN = 123.68
ENC_I_G_MEAN = 116.78
ENC_I_B_MEAN = 103.94


def attention_pooling(memory, score, scope='attention_pooling'):
    """
    Args:
        - memory: [bs, len, dim]
        - score: [bs, len]
    Returns:
        - pooled_memory: [bs, dim]
    """
    with tf.name_scope(scope):
        log.warning(scope)
        expanded_score = tf.expand_dims(score, axis=1)
        # expanded_score shape is [bs, 1, len]
        # memory shape is [bs, len, dim]
        pooled_memory = tf.matmul(expanded_score, memory)
        # pooled memory shape is [bs, 1, dim]
        pooled_memory = tf.squeeze(pooled_memory, axis=1)
    return pooled_memory


def attention(memory, memory_len, query, scope='attention'):
    """
    Args:
        - memory: [bs, len, dim]
        - memory_len: [bs]
        - query: [bs, dim]
    Returns:
        - score: [bs, len] (probability that sums to one)
    """
    with tf.name_scope(scope):
        log.warning(scope)
        with tf.name_scope('compute'):
            score = tf.reduce_sum(memory * tf.expand_dims(query, axis=1),
                                  axis=-1)
        with tf.name_scope('mask'):
            score_mask_value = tf.as_dtype(score.dtype).as_numpy_dtype(-np.inf)
            score_mask = tf.sequence_mask(
                memory_len, maxlen=tf.shape(score)[1])
            score_mask_values = score_mask_value * tf.ones_like(score)
            score = tf.where(score_mask, score, score_mask_values)
        with tf.name_scope('normalize'):
            score = tf.nn.softmax(score, axis=-1)
        return score


def hadamard_attention(memory, memory_len, query, use_ln=False, is_train=True,
                       scope='hadamard_attention', reuse=tf.AUTO_REUSE,
                       normalizer='softmax'):
    """
    Args:
        - memory: [bs, len, dim]
        - memory_len: [bs]
        - query: [bs, dim]
    Returns:
        - score: [bs, len] (probability that sums to one)
    """
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        with tf.variable_scope('compute', reuse=reuse):
            score_feat = memory * tf.expand_dims(query, axis=1)
            score_feat = tf.nn.dropout(score_feat, 0.8)
            score = fc_layer(score_feat, 1, use_bias=True, use_bn=False,
                             use_ln=use_ln,
                             activation_fn=None, is_training=is_train,
                             scope='score')
            score = tf.squeeze(score, axis=-1)
        with tf.name_scope('mask'):
            score_mask_value = tf.as_dtype(score.dtype).as_numpy_dtype(-np.inf)
            score_mask = tf.sequence_mask(
                memory_len, maxlen=tf.shape(score)[1])
            score_mask_values = score_mask_value * tf.ones_like(score)
            score = tf.where(score_mask, score, score_mask_values)
        with tf.name_scope('normalize'):
            if normalizer == 'softmax':
                score = tf.nn.softmax(score, axis=-1)
        return score


def encode_L_bidirection(seq, seq_len, dim=384, scope='encode_L_bi',
                         reuse=tf.AUTO_REUSE, cell_type='LSTM'):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        dim1 = int(math.ceil(dim / 2.0))
        dim2 = int(math.floor(dim / 2.0))
        log.warning(scope.name)
        if cell_type == 'LSTM':
            cell1 = rnn.BasicLSTMCell(num_units=dim1, state_is_tuple=True)
            cell2 = rnn.BasicLSTMCell(num_units=dim2, state_is_tuple=True)
        elif cell_type == 'GRU':
            cell1 = rnn.GRUCell(num_units=dim1)
            cell2 = rnn.GRUCell(num_units=dim2)
        else: raise ValueError('Unknown cell_type')
        bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell1, cell_bw=cell2, inputs=seq, sequence_length=seq_len,
            dtype=tf.float32)
        if cell_type == 'LSTM':
            raise RuntimeError('Check how LSTM works with bidirectional rnn')
        elif cell_type == 'GRU':
            output = tf.concat(bi_outputs, -1)
            output_state = tf.concat(encoder_state, -1)
        return output, output_state


def encode_L(seq, seq_len, dim=384, scope='encode_L',
             reuse=tf.AUTO_REUSE, cell_type='LSTM'):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        if cell_type == 'LSTM':
            cell = rnn.BasicLSTMCell(num_units=dim, state_is_tuple=True)
        elif cell_type == 'GRU':
            cell = rnn.GRUCell(num_units=dim)
        else: raise ValueError('Unknown cell_type')
        _, final_state = tf.nn.dynamic_rnn(
            cell=cell, dtype=tf.float32, sequence_length=seq_len,
            inputs=seq)
        if cell_type == 'LSTM':
            out = final_state.h
        elif cell_type == 'GRU':
            out = final_state
        return out


def encode_I_full(images, is_train=False, reuse=tf.AUTO_REUSE):
    """
    Pre-trained model parameter is available here:
    https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
    """
    with tf.name_scope('enc_I_preprocess'):
        channels = tf.split(axis=3, num_or_size_splits=3, value=images)
        for i, mean in enumerate([ENC_I_R_MEAN, ENC_I_G_MEAN, ENC_I_B_MEAN]):
            channels[i] -= mean
        processed_I = tf.concat(axis=3, values=channels)

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        enc_I, _ = nets.resnet_v1.resnet_v1_50(
            processed_I,
            is_training=is_train,
            global_pool=False,
            output_stride=None,
            reuse=reuse,
            scope='resnet_v1_50')
    return enc_I


def encode_I_block3(images, is_train=False, reuse=tf.AUTO_REUSE):
    """
    Pre-trained model parameter is available here:
    https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
    """
    with tf.name_scope('enc_I_preprocess'):
        channels = tf.split(axis=3, num_or_size_splits=3, value=images)
        for i, mean in enumerate([ENC_I_R_MEAN, ENC_I_G_MEAN, ENC_I_B_MEAN]):
            channels[i] -= mean
        processed_I = tf.concat(axis=3, values=channels)

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        resnet_v1_block = nets.resnet_v1.resnet_v1_block
        blocks = [
            resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
            resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
            resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
        ]
        enc_I, _ = nets.resnet_v1.resnet_v1(
            processed_I,
            blocks,
            is_training=is_train,
            global_pool=False,
            output_stride=None,
            reuse=reuse,
            scope='resnet_v1_50')
    return enc_I


def batch_box(box_idx, offset=None):
    sz = tf.shape(box_idx)
    if offset is None:
        offset = sz[1]
    offset = tf.tile(tf.expand_dims(tf.range(0, sz[0]),
                                    axis=1), [1, sz[1]]) * offset
    batch_box = box_idx + offset
    return batch_box


def roi_pool(ftmap, box, height, width, scope='roi_pool'):
    with tf.name_scope(scope):
        ft_dim = ftmap.get_shape().as_list()[3]
        box_sz = tf.shape(box)

        batch_box = tf.reshape(box, [-1, 4])
        batch_ids = tf.reshape(tf.tile(tf.expand_dims(
            tf.range(0, box_sz[0]), axis=1), [1, box_sz[1]]), [-1])

        pooled = tf.reshape(tf.image.crop_and_resize(
            ftmap, batch_box, batch_ids, [height, width]),
            [-1, box_sz[1], height, width, ft_dim])
        return pooled


def I_reduce_dim(enc_I, out_dim, scope='I_reduce_dim', is_train=False,
                 reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        enc_I = conv2d(enc_I, out_dim, 1, pad='same', use_bias=False,
                       use_bn=True, activation_fn=tf.nn.relu,
                       is_training=is_train, scope='conv2d', reuse=reuse)
        return enc_I


def I2V(enc_I, enc_dim, out_dim, scope='I2V', is_train=False, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        V_ft = conv2d(enc_I, enc_dim, 3, pad='valid', use_bias=False,
                      use_bn=True, activation_fn=tf.nn.relu,
                      is_training=is_train, scope='conv2d_1', reuse=reuse)
        V_ft = conv2d(V_ft, out_dim, 3, pad='valid', use_bias=False,
                      use_bn=True, activation_fn=tf.nn.relu,
                      is_training=is_train, scope='conv2d_1', reuse=reuse)
        V_ft = tf.squeeze(V_ft, axis=[-3, -2])
        return V_ft


def word_prediction(inputs, word_weights, activity_regularizer=None,
                    trainable=True, name=None, reuse=None):
    layer = WordPredictor(word_weights, activity_regularizer=activity_regularizer,
                          trainable=trainable, name=name,
                          dtype=inputs.dtype.base_dtype,
                          _scope=name, _reuse=reuse)
    return layer.apply(inputs)


def batch_word_classifier(inputs, word_weights, scope='batch_word_classifier',
                          reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        bias = tf.get_variable(name='bias', shape=(),
                               initializer=tf.zeros_initializer())
        logits = tf.reduce_sum(tf.expand_dims(inputs, axis=-2) * word_weights,
                               axis=-1) + bias
        return logits


class WordPredictor(tf.layers.Layer):

    def __init__(self,
                 word_weights,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(WordPredictor, self).__init__(
            trainable=trainable, name=name,
            activity_regularizer=activity_regularizer, **kwargs)

        self.word_weights = word_weights  # [num_words, dim]
        self.num_words = word_weights.get_shape().as_list()[0]
        self.word_dim = word_weights.get_shape().as_list()[1]
        if self.num_words is None:
            raise ValueError(
                'The first dimension of the weights must be defined')

    def build(self, input_shape):
        self.bias = self.add_variable('bias',
                                      shape=(),
                                      dtype=self.dtype,
                                      initializer=tf.zeros_initializer(),
                                      regularizer=None,
                                      trainable=True)
        self.built = True

    def call(self, inputs):
        # Inputs: [bs, dim]
        logits = tf.reduce_sum(tf.expand_dims(inputs, axis=1) *
                               tf.expand_dims(self.word_weights, axis=0),
                               axis=-1) + self.bias
        return logits  # [bs, num_words]

    def compute_output_shape(self, input_shape):
        # [bs, dim]
        input_shape = tf.TensorShape(input_shape).as_list()

        # [bs, num_words]
        output_shape = tf.TensorShape([input_shape[0], self.num_words])
        return output_shape


def decode_L(inputs, dim, embed_map, start_token,
             unroll_type='teacher_forcing', seq=None, seq_len=None,
             end_token=None, max_seq_len=None, output_layer=None,
             is_train=True, scope='decode_L', reuse=tf.AUTO_REUSE):

    with tf.variable_scope(scope, reuse=reuse) as scope:
        init_c = fc_layer(inputs, dim, use_bias=True, use_bn=False,
                          activation_fn=tf.nn.tanh, is_training=is_train,
                          scope='Linear_c', reuse=reuse)
        init_h = fc_layer(inputs, dim, use_bias=True, use_bn=False,
                          activation_fn=tf.nn.tanh, is_training=is_train,
                          scope='Linear_h', reuse=reuse)
        init_state = rnn.LSTMStateTuple(init_c, init_h)
        log.warning(scope.name)

        start_tokens = tf.zeros(
            [tf.shape(inputs)[0]], dtype=tf.int32) + start_token
        if unroll_type == 'teacher_forcing':
            if seq is None: raise ValueError('seq is None')
            if seq_len is None: raise ValueError('seq_len is None')
            seq_with_start = tf.concat([tf.expand_dims(start_tokens, axis=1),
                                        seq[:, :-1]], axis=1)
            helper = seq2seq.TrainingHelper(
                tf.nn.embedding_lookup(embed_map, seq_with_start), seq_len)
        elif unroll_type == 'greedy':
            if end_token is None: raise ValueError('end_token is None')
            helper = seq2seq.GreedyEmbeddingHelper(
                lambda e: tf.nn.embedding_lookup(embed_map, e),
                start_tokens, end_token)
        else:
            raise ValueError('Unknown unroll_type')

        cell = rnn.BasicLSTMCell(num_units=dim, state_is_tuple=True)
        decoder = seq2seq.BasicDecoder(cell, helper, init_state,
                                       output_layer=output_layer)
        outputs, _, pred_length = seq2seq.dynamic_decode(
            decoder, maximum_iterations=max_seq_len,
            scope='dynamic_decoder')

        output = outputs.rnn_output
        pred = outputs.sample_id

        return output, pred, pred_length


def learn_embedding_map(used_vocab, scope='learn_embedding_map', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        embed_map = tf.get_variable(
            name='learn', shape=[len(used_vocab['vocab']), 300],
            initializer=tf.random_uniform_initializer(
                minval=-0.01, maxval=0.01))
        return embed_map


def BiasVariable(shape, initializer=None, scope='Bias', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        if initializer is None:
            zeros = np.zeros(shape, dtype=np.float32)
            initializer = tf.constant_initializer(zeros)
        bias = tf.get_variable(
            name='bias', shape=shape, initializer=initializer)
        return bias


def LearnAnswerGloVe(answer_dict, scope='LearnAnswerGloVe', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        glove_vocab = json.load(open(GLOVE_VOCAB_PATH, 'r'))
        with h5py.File(GLOVE_EMBEDDING_PATH, 'r') as f:
            glove_param = np.array(f.get('param')).transpose()
        weights = np.zeros([len(answer_dict['vocab']), 300], dtype=np.float32)
        for i, answer_word in enumerate(answer_dict['vocab']):
            ws = []
            for w in answer_word.split():
                if w in glove_vocab['dict'] and \
                        glove_vocab['dict'][w] < glove_param.shape[0]:
                    ws.append(glove_param[glove_vocab['dict'][w], :])
            if len(ws) > 0:
                weights[i, :] = np.stack(ws, axis=0).mean(axis=0)
            else: pass  # initialize to zero
        init = tf.constant_initializer(weights)
        embed_map = tf.get_variable(
            name='embed_map', shape=[len(answer_dict['vocab']), 300],
            initializer=init)
        return embed_map


def WordWeightEmbed(vocab, word_weight_dir=None, weight_name='v_word',
                    scope='WordWeightEmbed', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        weights = np.zeros([len(vocab['vocab']), 300], dtype=np.float32)
        if word_weight_dir is not None:
            word_vocab_path = os.path.join(word_weight_dir, 'vocab.pkl')
            word_vocab = cPickle.load(open(word_vocab_path, 'rb'))
            word_weight_path = os.path.join(word_weight_dir, 'weights.hdf5')
            with h5py.File(word_weight_path, 'r') as f:
                word_weight = np.array(f.get(weight_name))
            num_row = word_weight.shape[0]
            for i, w in enumerate(vocab['vocab']):
                if w in word_vocab['dict'] and word_vocab['dict'][w] < num_row:
                    weights[i, :] = word_weight[word_vocab['dict'][w], :]
                else: pass
        init = tf.constant_initializer(weights)
        embed_map = tf.get_variable(
            name='embed_map', shape=[len(vocab['vocab']), 300],
            initializer=init)
        return embed_map


def LearnGloVe(
        vocab, scope='LearnGloVe',
        reuse=tf.AUTO_REUSE,
        learnable=True,
        oov_mean_initialize=False): # out of vocab
    with tf.variable_scope(scope, reuse=reuse) as scope:
        glove_vocab = json.load(open(GLOVE_VOCAB_PATH, 'r'))
        with h5py.File(GLOVE_EMBEDDING_PATH, 'r') as f:
            glove_param = np.array(f.get('param')).transpose()
        weights = np.zeros([len(vocab['vocab']), 300], dtype=np.float32)
        for i, w in enumerate(vocab['vocab']):
            if w in glove_vocab['dict'] and \
                    glove_vocab['dict'][w] < glove_param.shape[0]:
                weights[i, :] = glove_param[glove_vocab['dict'][w], :]
            elif oov_mean_initialize and w != "":
                words = w.split()
                check_all_in_glove = all(w in glove_vocab['dict'] for w in words)

                if check_all_in_glove:
                    weights[i, :] = np.mean(
                        [glove_param[glove_vocab['dict'][w], :] for w in words], 0)
                else:
                    raise Exception("Unkown words {}".format(words))
            else:
                pass  # initialize to zero

        if learnable:
            embed_map = tf.get_variable(
                name='embed_map', shape=[len(vocab['vocab']), 300],
                initializer=tf.constant_initializer(weights))
        else:
            embed_map = tf.constant(weights.T)

        return embed_map


def GloVe_vocab(vocab, scope='GloVe', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        glove_vocab = json.load(open(GLOVE_VOCAB_PATH, 'r'))
        used_vocab_idx = [glove_vocab['dict'][v] for v in vocab['vocab'][:-3]]

        with h5py.File(GLOVE_EMBEDDING_PATH, 'r') as f:
            glove_param = f['param'].value
        subset_param = np.take(glove_param, used_vocab_idx, axis=1)

        log.warning(scope.name)
        fixed = tf.constant(subset_param.transpose())
        learn = tf.get_variable(
            name='learn', shape=[3, 300],
            initializer=tf.random_uniform_initializer(
                minval=-0.01, maxval=0.01))
        embed_map = tf.concat([fixed, learn], axis=0)
        return embed_map


def GloVe(glove_path, scope='GloVe', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        with h5py.File(glove_path, 'r') as f:
            fixed = tf.constant(f['param'].value.transpose())
        learn = tf.get_variable(
            name='learn', shape=[3, 300],
            initializer=tf.random_uniform_initializer(
                minval=-0.01, maxval=0.01))
        embed_map = tf.concat([fixed, learn], axis=0)
        return embed_map


def LearnedVector(vocab, scope='LearnedVector', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        embed_map = tf.get_variable(
            name='learn', shape=[len(vocab['vocab']), 300],
            initializer=tf.random_uniform_initializer(
                minval=-0.01, maxval=0.01))
        return embed_map


def embed_transform(embed_map, enc_dim, out_dim, is_train=True,
                    scope='embed_transform', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        h = fc_layer(
            embed_map, enc_dim, use_bias=True, use_bn=False,
            activation_fn=tf.nn.relu, is_training=is_train,
            scope='fc_1', reuse=reuse)
        new_map = fc_layer(
            h, out_dim, use_bias=True, use_bn=False,
            activation_fn=None, is_training=is_train,
            scope='Linear', reuse=reuse)
        return new_map


def V2L(feat_V, enc_dim, out_dim, is_train=True, scope='V2L',
        reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        h1 = fc_layer(
            feat_V, enc_dim, use_bias=True, use_bn=False,
            activation_fn=tf.nn.tanh, is_training=is_train,
            scope='fc_1', reuse=reuse)
        h2 = fc_layer(
            h1, enc_dim, use_bias=True, use_bn=False,
            activation_fn=tf.nn.tanh, is_training=is_train,
            scope='fc_2', reuse=reuse)
        map_L = fc_layer(
            h2, out_dim, use_bias=True, use_bn=False,
            activation_fn=None, is_training=is_train,
            scope='Linear', reuse=reuse)
        return map_L, [h1, h2, map_L]


def flat_sigmoid_loss(labels=None, logits=None):
    sigmoid_ce = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return tf.reduce_mean(sigmoid_ce, axis=-1)


def L2V(feat_L, enc_dim, out_dim, is_train=True, scope='L2V',
        reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        h = fc_layer(
            feat_L, enc_dim, use_bias=True, use_bn=False,
            activation_fn=tf.nn.relu, is_training=is_train,
            scope='fc_1', reuse=reuse)
        h = fc_layer(
            h, enc_dim, use_bias=True, use_bn=False,
            activation_fn=tf.nn.relu, is_training=is_train,
            scope='fc_2', reuse=reuse)
        map_V = fc_layer(
            h, out_dim, use_bias=True, use_bn=False,
            activation_fn=None, is_training=is_train,
            scope='Linear', reuse=reuse)
        return map_V


def conv2d(input, dim, kernel_size, pad='same', use_bias=False, use_bn=False,
           activation_fn=None, is_training=True, scope='conv2d',
           reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        if use_bias:
            out = layers.conv2d(input, dim, kernel_size, padding=pad,
                                activation_fn=None, reuse=reuse,
                                trainable=is_training, scope='conv2d')
        else:
            out = layers.conv2d(input, dim, kernel_size, padding=pad,
                                activation_fn=None,
                                biases_initializer=None, reuse=reuse,
                                trainable=is_training, scope='conv2d')
        if use_bn:
            out = layers.batch_norm(out, center=True, scale=True, decay=0.9,
                                    is_training=is_training,
                                    updates_collections=None)
        if activation_fn is not None:
            out = activation_fn(out)
        return out


def AnswerExistMask(answer_dict, word_weight_dir=None):
    mask = np.zeros([len(answer_dict['vocab'])], dtype=np.float32)
    if word_weight_dir is not None:
        word_answer_dict_path = os.path.join(word_weight_dir, 'answer_dict.pkl')
        word_answer_dict = cPickle.load(open(word_answer_dict_path, 'rb'))
        for i, a in enumerate(answer_dict['vocab']):
            if a in word_answer_dict['dict']:
                mask[i] = 1.0
    else: mask = mask + 1.0
    mask = np.expand_dims(mask, axis=0)  # make batch dimension
    mask_tensor = tf.convert_to_tensor(mask)
    return mask_tensor


def WordWeightAnswer(input, answer_dict, word_weight_dir=None,
                     use_bias=False, is_training=True,
                     scope='WordWeightAnswer',
                     weight_name='class_weights',
                     bias_name='class_biases',
                     default_bias=-100.0,
                     reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        input_dim = input.get_shape().as_list()[-1]
        dim = len(answer_dict['vocab'])
        weights = np.zeros([input_dim, dim], dtype=np.float32)
        biases = np.zeros([dim], dtype=np.float32) + default_bias
        if word_weight_dir is not None:
            word_answer_dict_path = os.path.join(word_weight_dir, 'answer_dict.pkl')
            word_answer_dict = cPickle.load(open(word_answer_dict_path, 'rb'))
            word_weight_path = os.path.join(word_weight_dir, 'weights.hdf5')
            with h5py.File(word_weight_path, 'r') as f:
                answer_weight = np.array(f.get(weight_name))
                answer_bias = np.array(f.get(bias_name))

            for i, a in enumerate(answer_dict['vocab']):
                if a in word_answer_dict['dict']:
                    weights[:, i] = answer_weight[:, word_answer_dict['dict'][a]]
                    biases[i] = answer_bias[word_answer_dict['dict'][a]]
                else: pass  # initialize to zero
        if use_bias:
            out = layers.fully_connected(
                input, dim, activation_fn=None,
                weights_initializer=tf.constant_initializer(weights),
                biases_initializer=tf.constant_initializer(biases),
                reuse=reuse, trainable=is_training, scope='fc')
        else:
            out = layers.fully_connected(
                input, dim, activation_fn=None,
                weights_initializer=tf.constant_initializer(weights),
                biases_initializer=None,
                reuse=reuse, trainable=is_training, scope='fc')
        return out


def fc_layer(input, dim, use_bias=False, use_bn=False, use_ln=False, activation_fn=None,
             is_training=True, scope='fc_layer', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        log.warning(scope.name)
        if use_bias:
            out = layers.fully_connected(
                input, dim, activation_fn=None, reuse=reuse,
                trainable=is_training, scope='fc')
        else:
            out = layers.fully_connected(
                input, dim, activation_fn=None, biases_initializer=None,
                reuse=reuse, trainable=is_training, scope='fc')
        if use_bn:
            out = layers.batch_norm(out, center=True, scale=True, decay=0.9,
                                    is_training=is_training,
                                    updates_collections=None)
        if use_ln:
            out = layers.layer_norm(out)
        if activation_fn is not None:
            out = activation_fn(out)
        return out
