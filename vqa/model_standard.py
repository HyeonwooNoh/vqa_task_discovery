import cPickle
import h5py
import os
import numpy as np
import tensorflow as tf

from util import box_utils, log
from misc import modules

W_DIM = 300  # Word dimension
L_DIM = 1024  # Language dimension
V_DIM = 1024
ENC_I_PARAM_PATH = 'data/nets/resnet_v1_50.ckpt'


class Model(object):

    def __init__(self, batch, config, is_train=True, image_features=None):
        self.batch = batch
        self.config = config
        self.image_dir = config.image_dir
        self.is_train = is_train

        # word_weight_dir is only for answer accuracy visualization
        self.word_weight_dir = getattr(config, 'pretrain_word_weight_dir', None)
        if self.word_weight_dir is None:
            log.warn('word_weight_dir is None')

        self.losses = {}
        self.report = {}
        self.mid_result = {}
        self.output = {}
        self.heavy_output = {}
        self.vis_image = {}

        self.vocab = cPickle.load(open(config.vocab_path, 'rb'))
        self.answer_dict = cPickle.load(open(
            os.path.join(config.tf_record_dir, 'answer_dict.pkl'), 'rb'))
        self.num_answer = len(self.answer_dict['vocab'])
        self.num_train_answer = self.answer_dict['num_train_answer']
        self.train_answer_mask = tf.expand_dims(tf.sequence_mask(
            self.num_train_answer, maxlen=self.num_answer, dtype=tf.float32),
            axis=0)
        self.test_answer_mask = 1.0 - self.train_answer_mask
        self.obj_answer_mask = tf.expand_dims(
            tf.constant(self.answer_dict['is_object'], dtype=tf.float32),
            axis=0)
        self.attr_answer_mask = tf.expand_dims(
            tf.constant(self.answer_dict['is_attribute'], dtype=tf.float32),
            axis=0)

        self.glove_map = modules.LearnGloVe(self.vocab)
        self.answer_exist_mask = modules.AnswerExistMask(
            self.answer_dict, self.word_weight_dir)

        if image_features is None:
            log.infov('loading image features...')
            with h5py.File(config.vfeat_path, 'r') as f:
                self.features = np.array(f.get('image_features'))
                log.infov('feature done')
                self.spatials = np.array(f.get('spatial_features'))
                log.infov('spatials done')
                self.normal_boxes = np.array(f.get('normal_boxes'))
                log.infov('normal_boxes done')
                self.num_boxes = np.array(f.get('num_boxes'))
                log.infov('num_boxes done')
                self.max_box_num = int(f['data_info']['max_box_num'].value)
                self.vfeat_dim = int(f['data_info']['vfeat_dim'].value)
            log.infov('done')
        else:
            self.features = image_features['features']
            self.spatials = image_features['spatials']
            self.normal_boxes = image_features['normal_boxes']
            self.num_boxes = image_features['num_boxes']
            self.max_box_num = image_features['max_box_num']
            self.vfeat_dim = image_features['vfeat_dim']

        self.build()

    def filter_train_vars(self, trainable_vars):
        train_vars = []
        for var in trainable_vars:
            train_vars.append(var)
        return train_vars

    def filter_transfer_vars(self, all_vars):
        transfer_vars = []
        for var in all_vars:
            if var.name.split('/')[0] == 'encode_L':
                transfer_vars.append(var)
            elif var.name.split('/')[0] == 'GloVe':
                transfer_vars.append(var)
        return transfer_vars

    def get_enc_I_param_path(self):
        return ENC_I_PARAM_PATH

    def visualize_vqa_result(self,
                             image_id, normal_box, num_box,
                             att_score,
                             q_intseq, q_intseq_len,
                             answer_target, pred,
                             max_batch_num=20, line_width=2):
        def construct_visualization(b_image_id, bb_normal_box, b_num_box,
                                    bb_att_score,
                                    b_q_intseq, b_q_intseq_len,
                                    b_answer_target, b_pred):
            # b_ : batch
            # bb_ : [batch, description]
            import textwrap
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.load_default()

            def intseq2str(intseq):
                return ' '.join([self.vocab['vocab'][i] for i in intseq])

            def string2image(string, image_width=1080):
                pil_text = Image.fromarray(
                    np.zeros([15, image_width, 3], dtype=np.uint8) + 220)
                t_draw = ImageDraw.Draw(pil_text)
                for l, line in enumerate(textwrap.wrap(string, width=90)):
                    t_draw.text((2, 2 + l * 15), line, font=font,
                                fill=(10, 10, 50))
                return np.array(pil_text).astype(np.uint8)

            batch_vis_image = []
            for batch_idx in range(min(len(b_image_id), max_batch_num)):
                image_id = b_image_id[batch_idx]
                image_path = os.path.join(self.image_dir,
                                          image_id.replace('-', '/'))
                image = Image.open(image_path)
                image = np.array(image.resize([540, 540]).convert('RGB'))
                float_image = image.astype(np.float32)
                att_mask = np.zeros_like(float_image)

                b_score = bb_att_score[batch_idx]
                max_score_idx = np.argmax(b_score)
                for box_idx in range(b_num_box[batch_idx]):
                    b_normal_box = bb_normal_box[batch_idx]
                    b_box = box_utils.scale_boxes_x1y1x2y2(b_normal_box, [540, 540])
                    box = b_box[box_idx]
                    att_mask = box_utils.add_value_x1y1x2y2(
                        image=att_mask, box=box, value=b_score[box_idx])
                att_image = Image.fromarray(
                        (float_image * att_mask).astype(np.uint8))
                draw = ImageDraw.Draw(att_image)
                b_normal_box = bb_normal_box[batch_idx]
                b_box = box_utils.scale_boxes_x1y1x2y2(b_normal_box, [540, 540])
                (x1, y1, x2, y2) = b_box[max_score_idx]
                for w in range(line_width):
                    draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                                   outline=(255, 0, 0))
                att_image = np.array(att_image, dtype=np.uint8)

                vis_image = np.concatenate([image, att_image], axis=1)
                width = vis_image.shape[1]

                q_intseq = b_q_intseq[batch_idx]
                q_intseq_len = b_q_intseq_len[batch_idx]

                question_str = ' '.join(
                    [self.vocab['vocab'][t] for t in q_intseq[:q_intseq_len]])
                question_image = string2image('Q: ' + question_str,
                                              image_width=width)
                answer_target = b_answer_target[batch_idx]
                gt_idx_list = np.where(answer_target > 0)
                if len(gt_idx_list) > 0:
                    label_answer_str = ''
                    for ans_i in gt_idx_list:
                        label_answer_str += '{}({:.1f})'.format(
                            self.answer_dict['vocab'][ans_i], answer_target[ans_i])
                else:
                    label_answer_str = '<no frequent answer>'
                label_answer_image = string2image('GT: ' + label_answer_str,
                                                  image_width=width)

                pred = b_pred[batch_idx]
                pred_answer_str = self.answer_dict['vocab'][pred]
                pred_answer_image = string2image('PRED: ' + pred_answer_str,
                                                 image_width=width)
                vis_image = np.concatenate(
                    [vis_image, question_image,
                     label_answer_image, pred_answer_image], axis=0)
                batch_vis_image.append(vis_image)
            batch_vis_image = np.stack(batch_vis_image, axis=0)
            return batch_vis_image
        return tf.py_func(
            construct_visualization,
            inp=[image_id, normal_box, num_box, att_score,
                 q_intseq, q_intseq_len, answer_target, pred],
            Tout=tf.uint8)

    def build(self):
        """
        build network architecture and loss
        """

        """
        Visual features
        """
        with tf.device('/cpu:0'):
            def load_feature(image_idx):
                selected_features = np.take(self.features, image_idx, axis=0)
                return selected_features
            V_ft = tf.py_func(
                load_feature, inp=[self.batch['image_idx']], Tout=tf.float32,
                name='sample_features')
            V_ft.set_shape([None, self.max_box_num, self.vfeat_dim])
            num_V_ft = tf.gather(self.num_boxes, self.batch['image_idx'],
                                 name='gather_num_V_ft', axis=0)
            self.mid_result['num_V_ft'] = num_V_ft
            normal_boxes = tf.gather(self.normal_boxes, self.batch['image_idx'],
                                     name='gather_normal_boxes', axis=0)
            self.mid_result['normal_boxes'] = normal_boxes

        log.warning('v_linear_v')
        v_linear_v = modules.fc_layer(
            V_ft, V_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='v_linear_v')

        """
        Encode question
        """
        q_embed = tf.nn.embedding_lookup(self.glove_map, self.batch['q_intseq'])
        # [bs, L_DIM]
        q_L_ft = modules.encode_L(q_embed, self.batch['q_intseq_len'], L_DIM,
                                  cell_type='GRU')
        self.heavy_output['condition'] = q_L_ft

        # [bs, V_DIM}
        log.warning('q_linear_v')
        q_linear_v = modules.fc_layer(
            q_L_ft, V_DIM, use_bias=True, use_bn=False, use_ln=True,
            activation_fn=tf.nn.relu, is_training=self.is_train,
            scope='q_linear_v')

        """
        Perform attention
        """
        att_score = modules.hadamard_attention(v_linear_v, num_V_ft, q_linear_v,
                                               use_ln=False, is_train=self.is_train)
        self.output['att_score'] = att_score
        self.mid_result['att_score'] = att_score
        pooled_V_ft = modules.attention_pooling(V_ft, att_score)

        """
        Answer classification
        """
        # perform two layer feature encoding and predict output
        with tf.variable_scope('reasoning') as scope:
            log.warning(scope.name)
            # [bs, L_DIM]
            log.warning('pooled_linear_l')
            pooled_linear_l = modules.fc_layer(
                pooled_V_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
                activation_fn=tf.nn.relu, is_training=self.is_train,
                scope='pooled_linear_l')

            log.warning('q_linear_l')
            q_linear_l = modules.fc_layer(
                q_L_ft, L_DIM, use_bias=True, use_bn=False, use_ln=True,
                activation_fn=tf.nn.relu, is_training=self.is_train,
                scope='q_linear_l')

            joint = modules.fc_layer(
                pooled_linear_l * q_linear_l, 2048,
                use_bias=True, use_bn=False, use_ln=True,
                activation_fn=tf.nn.relu, is_training=self.is_train, scope='joint_fc')
            joint = tf.nn.dropout(joint, 0.5)

            logit = modules.fc_layer(
                joint, self.num_answer,
                use_bias=True, use_bn=False, use_ln=False,
                activation_fn=None, is_training=self.is_train, scope='classifier')
        self.output['logit'] = logit

        """
        Compute loss and accuracy
        """
        with tf.name_scope('loss'):
            answer_target = self.batch['answer_target']
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=answer_target, logits=logit)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
            pred = tf.cast(tf.argmax(logit, axis=-1), dtype=tf.int32)
            one_hot_pred = tf.one_hot(pred, depth=self.num_answer,
                                      dtype=tf.float32)
            self.output['pred'] = pred
            all_score = tf.reduce_sum(one_hot_pred * answer_target, axis=-1)
            max_train_score = tf.reduce_max(
                answer_target * self.train_answer_mask, axis=-1)
            test_obj_score = tf.reduce_sum(
                one_hot_pred * answer_target * self.test_answer_mask *
                self.obj_answer_mask, axis=-1)
            test_obj_max_score = tf.reduce_max(
                answer_target * self.test_answer_mask *
                self.obj_answer_mask, axis=-1)
            test_attr_score = tf.reduce_sum(
                one_hot_pred * answer_target * self.test_answer_mask *
                self.attr_answer_mask, axis=-1)
            test_attr_max_score = tf.reduce_max(
                answer_target * self.test_answer_mask *
                self.attr_answer_mask, axis=-1)
            self.output['test_obj_score'] = test_obj_score
            self.output['test_obj_max_score'] = test_obj_max_score
            self.output['test_attr_score'] = test_attr_score
            self.output['test_attr_max_score'] = test_attr_max_score
            self.output['all_score'] = all_score
            self.output['max_train_score'] = max_train_score

            acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target, axis=-1))
            exist_acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target * self.answer_exist_mask,
                              axis=-1))
            test_acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target * self.test_answer_mask,
                              axis=-1))
            test_obj_acc = tf.reduce_mean(test_obj_score)
            test_attr_acc = tf.reduce_mean(test_attr_score)
            train_exist_acc = tf.reduce_mean(
                tf.reduce_sum(one_hot_pred * answer_target * self.answer_exist_mask *
                              self.train_answer_mask,
                              axis=-1))
            max_exist_answer_acc = tf.reduce_mean(
                tf.reduce_max(answer_target * self.answer_exist_mask, axis=-1))
            max_train_exist_acc = tf.reduce_mean(
                tf.reduce_max(answer_target * self.answer_exist_mask *
                              self.train_answer_mask, axis=-1))
            test_obj_max_acc = tf.reduce_mean(test_obj_max_score)
            test_attr_max_acc = tf.reduce_mean(test_attr_max_score)
            test_max_answer_acc = tf.reduce_mean(
                tf.reduce_max(answer_target * self.test_answer_mask, axis=-1))
            test_max_exist_answer_acc = tf.reduce_mean(
                tf.reduce_max(answer_target * self.answer_exist_mask *
                              self.test_answer_mask, axis=-1))
            normal_test_obj_acc = tf.where(
                tf.equal(test_obj_max_acc, 0),
                test_obj_max_acc,
                test_obj_acc / test_obj_max_acc)
            normal_test_attr_acc = tf.where(
                tf.equal(test_attr_max_acc, 0),
                test_attr_max_acc,
                test_attr_acc / test_attr_max_acc)
            normal_train_exist_acc = tf.where(
                tf.equal(max_train_exist_acc, 0),
                max_train_exist_acc,
                train_exist_acc / max_train_exist_acc)
            normal_exist_acc = tf.where(
                tf.equal(max_exist_answer_acc, 0),
                max_exist_answer_acc,
                exist_acc / max_exist_answer_acc)
            normal_test_acc = tf.where(
                tf.equal(test_max_answer_acc, 0),
                test_max_answer_acc,
                test_acc / test_max_answer_acc)

            self.mid_result['pred'] = pred

            self.losses['answer'] = loss
            self.report['answer_train_loss'] = loss
            self.report['answer_report_loss'] = loss
            self.report['answer_acc'] = acc
            self.report['exist_acc'] = exist_acc
            self.report['test_acc'] = test_acc
            self.report['normal_test_acc'] = normal_test_acc
            self.report['normal_test_object_acc'] = normal_test_obj_acc
            self.report['normal_test_attribute_acc'] = normal_test_attr_acc
            self.report['normal_exist_acc'] = normal_exist_acc
            self.report['normal_train_exist_acc'] = normal_train_exist_acc
            self.report['max_exist_acc'] = max_exist_answer_acc
            self.report['test_max_acc'] = test_max_answer_acc
            self.report['test_max_exist_acc'] = test_max_exist_answer_acc

        """
        Prepare image summary
        """
        """
        with tf.name_scope('prepare_summary'):
            self.vis_image['image_attention_qa'] = self.visualize_vqa_result(
                self.batch['image_id'],
                self.mid_result['normal_boxes'], self.mid_result['num_V_ft'],
                self.mid_result['att_score'],
                self.batch['q_intseq'], self.batch['q_intseq_len'],
                self.batch['answer_target'], self.mid_result['pred'],
                max_batch_num=20, line_width=2)
        """

        self.loss = self.losses['answer']

        # scalar summary
        for key, val in self.report.items():
            tf.summary.scalar('train/{}'.format(key), val,
                              collections=['heavy_train', 'train'])
            tf.summary.scalar('val/{}'.format(key), val,
                              collections=['heavy_val', 'val'])
            tf.summary.scalar('testval/{}'.format(key), val,
                              collections=['heavy_testval', 'testval'])

        # image summary
        for key, val in self.vis_image.items():
            tf.summary.image('train-{}'.format(key), val, max_outputs=10,
                             collections=['heavy_train'])
            tf.summary.image('val-{}'.format(key), val, max_outputs=10,
                             collections=['heavy_val'])
            tf.summary.image('testval-{}'.format(key), val, max_outputs=10,
                             collections=['heavy_testval'])

        return self.loss
