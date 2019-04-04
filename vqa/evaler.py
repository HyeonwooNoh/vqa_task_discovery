import argparse
import cPickle
import h5py
import os
import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from util import log
from vqa import importer
from vqa.datasets import input_ops_vqa_tf_record_memft as input_ops_vqa


class Evaler(object):

    @staticmethod
    def get_model_class(model_type='vqa'):
        return importer.get_model_class(model_type)

    def __init__(self, config, set_checkpoint=True, image_features=None):
        self.config = config
        self.split = config.split
        self.max_iter = config.max_iter
        self.dump_heavy_output = config.dump_heavy_output

        self.vfeat_path = config.vfeat_path
        self.tf_record_dir = config.tf_record_dir

        # Input
        self.batch_size = config.batch_size
        with tf.name_scope('datasets'):
            self.target_split = tf.placeholder(tf.string)

        with tf.name_scope('datasets/batch'):
            self.batch = input_ops_vqa.create(
                self.batch_size, self.tf_record_dir, self.split,
                is_train=False, scope='{}_ops'.format(self.split), shuffle=False)

        # Model
        Model = self.get_model_class(config.model_type)
        log.infov('using model class: {}'.format(Model))
        self.model = Model(self.batch, config, is_train=False,
                           image_features=image_features)

        trainable_vars = tf.trainable_variables()
        train_vars = self.model.filter_train_vars(trainable_vars)
        log.warn('Trainable variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(trainable_vars, print_info=True)
        log.warn('Filtered train variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(train_vars, print_info=True)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1})
        self.session = tf.Session(config=session_config)

        self.checkpoint_loader = tf.train.Saver(max_to_keep=1)

        if set_checkpoint:
            self.set_eval_dir(config)
            self.load_checkpoint(config)

    def set_eval_dir(self, config):
        self.checkpoint = config.checkpoint
        self.eval_dir = config.checkpoint + '_eval_{}'.format(self.split)
        if self.dump_heavy_output:
            self.eval_dir += '_dump_heavy'
        self.eval_dir += '_{}'.format(time.strftime("%Y%m%d-%H%M%S"))

        if not os.path.exists(self.eval_dir): os.makedirs(self.eval_dir)
        log.infov("Eval Dir: %s", self.eval_dir)
        self.save_hdf5 = os.path.join(self.eval_dir, 'results.hdf5')
        self.save_pkl = os.path.join(self.eval_dir, 'results.pkl')

    def load_checkpoint(self, config):
        if config.checkpoint is not None:
            log.info('Checkpoint path: {}'.format(config.checkpoint))
            self.checkpoint_loader.restore(self.session, config.checkpoint)
            log.info('Loaded the checkpoint')
        log.warn('Evaluation initialization is done')

    def eval(self):
        log.infov('Training starts')

        vocab = self.model.vocab
        answer_dict = self.model.answer_dict

        self.save_hdf5 = os.path.join(self.eval_dir, 'results.hdf5')
        self.save_pkl = os.path.join(self.eval_dir, 'results.pkl')
        result_dict = {
            'qid2result': {}
        }
        fetch_data = {
            'id': self.batch['id'],
            'image_id': self.batch['image_id'],
            'q_intseq': self.batch['q_intseq'],
            'q_intseq_len': self.batch['q_intseq_len'],
        }
        fetch_list = [
            self.model.report,
            self.model.output,
            fetch_data,
        ]
        if self.dump_heavy_output:
            fetch_list.append(self.model.heavy_output)
        # initialize average report (put 0 to escape average over empty list)
        avg_eval_report = {key: [] for key in self.model.report.keys()}
        avg_eval_report['testonly_score'] = []
        avg_eval_report['test_attr_only_score'] = []
        avg_eval_report['test_obj_only_score'] = []

        heavy_outputs = {key: [] for key in self.model.heavy_output.keys()}
        heavy_output_idx = 0
        if self.max_iter < 0: self.max_iter = 50000
        for s in tqdm(range(self.max_iter), desc='eval'):
            try:
                fetch = self.session.run(fetch_list)
            except tf.errors.OutOfRangeError:
                log.warn('OutOfRangeError happens at {} iter'.format(s + 1))
                break

            reports, outputs, inputs = fetch[:3]
            if self.dump_heavy_output:
                heavy_output = fetch[3]

            batch_size = len(inputs['id'])
            for b in range(batch_size):
                q_intseq = inputs['q_intseq'][b]
                q_intseq_len = inputs['q_intseq_len'][b]
                question = ' '.join(
                    [vocab['vocab'][v] for v in q_intseq[:q_intseq_len]])

                id = inputs['id'][b]
                image_id = inputs['image_id'][b]
                pred = answer_dict['vocab'][outputs['pred'][b]]

                score = outputs['all_score'][b]
                max_train_score = outputs['max_train_score'][b]
                test_obj_score = outputs['test_obj_score'][b]
                test_obj_max_score = outputs['test_obj_max_score'][b]
                test_attr_score = outputs['test_attr_score'][b]
                test_attr_max_score = outputs['test_attr_max_score'][b]

                result_dict['qid2result'][id] = {
                    'image_id': image_id,
                    'pred': pred,
                    'question': question,
                    'score': score,
                    'max_train_score': max_train_score,
                    'test_obj_score': test_obj_score,
                    'test_obj_max_score': test_obj_max_score,
                    'test_attr_score': test_attr_score,
                    'test_attr_max_score': test_attr_max_score,
                }
                if self.dump_heavy_output:
                    result_dict['qid2result'][id]['heavy_output_idx'] = heavy_output_idx
                    for key in heavy_output:
                        heavy_outputs[key].append(heavy_output[key][b])
                    heavy_output_idx += 1
                if max_train_score <= 0:
                    avg_eval_report['testonly_score'].append(score)
                    if test_obj_max_score <= 0:
                        avg_eval_report['test_attr_only_score'].append(test_attr_score)
                    if test_attr_max_score <= 0:
                        avg_eval_report['test_obj_only_score'].append(test_obj_score)
                for key in reports:
                    avg_eval_report[key].append(reports[key])

        result_dict['avg_eval_report'] = {
            key: np.array(avg_eval_report[key], dtype=np.float32).mean()
            for key in avg_eval_report}
        for key in avg_eval_report:
            result_dict['avg_eval_report']['{}_num_point'.format(key)] = len(avg_eval_report[key])
        log.info('saving pickle file to: {}'.format(self.save_pkl))
        cPickle.dump(result_dict, open(self.save_pkl, 'wb'))
        log.info('done')

        if self.dump_heavy_output:
            with h5py.File(self.save_hdf5, 'w') as f:
                log.info('saving h5 file to: {}'.format(self.save_hdf5))
                for key in tqdm(heavy_outputs):
                    f[key] = np.stack(heavy_outputs[key], axis=0)
                log.info('done')

        log.info('evaluation is done')


def check_config(config):
    pass


def parse_checkpoint(config):
    config.ckpt_name = config.checkpoint.split('/')[-1]

    dirname = config.checkpoint.split('/')[-2]
    config.model_type = dirname.split('vqa_')[1].split('_d_')[0]

    qa_split_name = dirname.split('_d_')[1].split('_tf_record_memft')[0]
    config.tf_record_dir = os.path.join(
        'data/preprocessed/vqa_v2', qa_split_name, 'tf_record_memft')

    if 'vfeat_bottomup_36_my' in dirname:
        config.vfeat_name = 'vfeat_bottomup_36_my.hdf5'
    else:
        config.vfeat_name = 'vfeat_bottomup_36.hdf5'

    config.vocab_path = os.path.join(config.tf_record_dir, config.vocab_name)
    config.vfeat_path = os.path.join(config.tf_record_dir, config.vfeat_name)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # paths
    parser.add_argument('--image_dir', type=str, default='data/VQA_v2/images',
                        help=' ')
    parser.add_argument('--vocab_name', type=str, default='vocab.pkl', help=' ')
    # evaluation setting
    parser.add_argument('--max_iter', type=int, default=-1, help=' ')
    parser.add_argument('--split', type=str, default='testval', help=' ',
                        choices=['train', 'val', 'testval', 'test'])
    # hyper parameters
    parser.add_argument('--prefix', type=str, default='default', help=' ')
    parser.add_argument('--checkpoint', type=str, default=None, required=True)
    # model parameters
    parser.add_argument('--batch_size', type=int, default=512, help=' ')
    parser.add_argument('--debug', type=int, default=0, help='0: normal, 1: debug')
    parser.add_argument('--dump_heavy_output', action='store_true', default=False,
                        help=' ')
    config = parser.parse_args()
    check_config(config)
    parse_checkpoint(config)

    evaler = Evaler(config)
    evaler.eval()

if __name__ == '__main__':
    main()
