import argparse
import os
import time
import shutil
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from util import log
from vqa import importer
from vqa.datasets import input_ops_vqa_tf_record_memft as input_ops_vqa


class Trainer(object):

    @staticmethod
    def get_model_class(model_type='proposed'):
        return importer.get_model_class(model_type)

    def __init__(self, config):
        self.config = config
        self.vfeat_path = config.vfeat_path
        self.tf_record_dir = config.tf_record_dir

        self.max_train_iter = config.max_train_iter

        hyper_parameter_str = 'bs{}_lr{}'.format(
            config.batch_size, config.learning_rate)

        if config.ft_pretrain:
            hyper_parameter_str += '_ft_pretrain'

        self.train_dir = './train_dir/vqa_{}_{}_{}_seed{}'.format(
            config.model_type, config.prefix, hyper_parameter_str, config.seed)
        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        if config.pretrain_word_weight_dir is not None:
            self.pretrain_word_weight_dir = os.path.join(
                self.train_dir, config.pretrain_word_weight_dir.split('/')[-1])
            shutil.copytree(config.pretrain_word_weight_dir, self.pretrain_word_weight_dir)
            config.pretrain_word_weight_dir = self.pretrain_word_weight_dir
        else: self.pretrain_word_weight_dir = config.pretrain_word_weight_dir

        # Input
        self.batch_size = config.batch_size
        with tf.name_scope('datasets'):
            self.target_split = tf.placeholder(tf.string)

        with tf.name_scope('datasets/batch'):
            vqa_batch = {
                'train': input_ops_vqa.create(
                    self.batch_size, self.tf_record_dir, 'train',
                    is_train=True, scope='train_ops', shuffle=True),
                'val': input_ops_vqa.create(
                    self.batch_size, self.tf_record_dir, 'val',
                    is_train=True, scope='val_ops', shuffle=False),
                'testval': input_ops_vqa.create(
                    self.batch_size, self.tf_record_dir, 'testval',
                    is_train=True, scope='testval_ops', shuffle=False),
                'test': input_ops_vqa.create(
                    self.batch_size, self.tf_record_dir, 'test',
                    is_train=True, scope='test_ops', shuffle=False)
            }
            batch_opt = {
                tf.equal(self.target_split, 'train'): lambda: vqa_batch['train'],
                tf.equal(self.target_split, 'val'): lambda: vqa_batch['val'],
                tf.equal(self.target_split, 'testval'): lambda: vqa_batch['testval'],
                tf.equal(self.target_split, 'test'): lambda: vqa_batch['test'],
            }
            self.batch = tf.case(
                batch_opt, default=lambda: vqa_batch['train'], exclusive=True)

        # Model
        Model = self.get_model_class(config.model_type)
        log.infov('using model class: {}'.format(Model))
        self.model = Model(self.batch, config, is_train=True)

        # Optimizer
        self.global_step = tf.train.get_or_create_global_step(graph=None)
        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate')

        # Checkpoint and monitoring
        trainable_vars = tf.trainable_variables()
        train_vars = self.model.filter_train_vars(trainable_vars)
        log.warn('Trainable variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(trainable_vars, print_info=True)
        log.warn('Filtered train variables:')
        tf.contrib.slim.model_analyzer.analyze_vars(train_vars, print_info=True)

        self.optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            variables=train_vars,
            increment_global_step=True,
            name='optimizer')

        self.avg_report = {
            'train': {},
            'val': {},
            'testval': {},
        }
        for split in ['train', 'val', 'testval']:
            for key in self.model.report.keys():
                self.avg_report[split][key] = tf.placeholder(tf.float32)
                tf.summary.scalar('average_{}/{}'.format(split, key),
                                  self.avg_report[split][key],
                                  collections=['average_{}'.format(split)])

        self.summary_ops = {
            'train': tf.summary.merge_all(key='train'),
            'val': tf.summary.merge_all(key='val'),
            'testval': tf.summary.merge_all(key='testval'),
            'heavy_train': tf.summary.merge_all(key='heavy_train'),
            'heavy_val': tf.summary.merge_all(key='heavy_val'),
            'heavy_testval': tf.summary.merge_all(key='heavy_testval'),
            'average_train': tf.summary.merge_all(key='average_train'),
            'average_val': tf.summary.merge_all(key='average_val'),
            'average_testval': tf.summary.merge_all(key='average_testval'),
            'no_op': tf.no_op(),
        }

        all_vars = tf.global_variables()
        transfer_vars = self.model.filter_transfer_vars(all_vars)

        self.saver = tf.train.Saver(max_to_keep=100)
        self.checkpoint_loader = tf.train.Saver(max_to_keep=1)
        self.pretrain_loader = tf.train.Saver(var_list=transfer_vars, max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.train_average_iter = self.config.train_average_iter
        self.val_average_iter = self.config.val_average_iter
        self.heavy_summary_step = self.config.heavy_summary_step
        self.validation_step = self.config.validation_step
        self.checkpoint_step = self.config.checkpoint_step

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=None,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1})

        self.session = self.supervisor.prepare_or_wait_for_session(
            config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info('Checkpoint path: {}'.format(self.ckpt_path))
            self.checkpoint_loader.restore(self.session, self.ckpt_path)
            log.info('Loaded the checkpoint')

        self.pretrained_param_path = config.pretrained_param_path
        if self.pretrained_param_path is not None:
            log.warn('Filtered transfer_vars (loaded from pre-trained param):')
            tf.contrib.slim.model_analyzer.analyze_vars(transfer_vars, print_info=True)

            log.info('Pre-trained param path: {}'.format(self.pretrained_param_path))
            self.pretrain_loader.restore(self.session, self.pretrained_param_path)
            log.info('Loaded the pre-trained parameters')

    def train(self):
        log.infov('Training starts')

        # initialize average report (put 0 to escape average over empty list)
        avg_step_time = [0]
        avg_train_report = {key: [0] for key in self.avg_report['train']}

        for s in range(self.max_train_iter):
            """
            write average summary and print log
            """
            if s % self.train_average_iter == 0:
                step, avg_train_summary = self.write_average_summary(
                    avg_train_report, split='train')
                self.summary_writer.add_summary(
                    avg_train_summary, global_step=step)
                self.log_message(step, avg_train_report, avg_step_time,
                                 split='train', is_train=True)
                for key in avg_train_report: avg_train_report[key] = []
                avg_step_time = []

            """
            Periodic inference on validation set
            """
            if s % self.validation_step == 0:
                # val
                avg_val_report = {key: [] for key in self.avg_report['val']}
                avg_val_step_time = []
                for i in tqdm(range(self.val_average_iter), desc='eval val'):
                    step, summary, loss, report, step_time = self.run_val_step(
                        i == (self.val_average_iter - 1), split='val')
                    for key in avg_val_report:
                        avg_val_report[key].append(report[key])
                    avg_val_step_time.append(step_time)
                self.summary_writer.add_summary(summary, global_step=step)
                step, avg_val_summary = self.write_average_summary(
                    avg_val_report, split='val')
                self.summary_writer.add_summary(avg_val_summary, global_step=step)
                self.log_message(step, avg_val_report, avg_val_step_time,
                                 split='val', is_train=False)

                # testval
                avg_val_report = {key: [] for key in self.avg_report['testval']}
                avg_val_step_time = []
                for i in tqdm(range(self.val_average_iter), desc='eval testval'):
                    step, summary, loss, report, step_time = self.run_val_step(
                        i == (self.val_average_iter - 1), split='testval')
                    for key in avg_val_report:
                        avg_val_report[key].append(report[key])
                    avg_val_step_time.append(step_time)
                self.summary_writer.add_summary(summary, global_step=step)
                step, avg_val_summary = self.write_average_summary(
                    avg_val_report, split='testval')
                self.summary_writer.add_summary(avg_val_summary, global_step=step)
                self.log_message(step, avg_val_report, avg_val_step_time,
                                 split='testval', is_train=False)

            """
            Run TRAINING step
            """
            step, train_summary, loss, train_report, step_time = \
                self.run_train_step(s % self.heavy_summary_step == 0)
            for key in avg_train_report:
                avg_train_report[key].append(train_report[key])
            avg_step_time.append(step_time)
            if s % self.heavy_summary_step == 0:
                self.summary_writer.add_summary(train_summary, global_step=step)

            """
            Save Checkpoint
            """
            if s % self.checkpoint_step == 0:
                log.infov('Saved checkpoint at {}'.format(step))
                self.saver.save(
                    self.session, os.path.join(self.train_dir, 'model'),
                    global_step=step)

    def write_average_summary(self, avg_report, split='train'):
        feed_dict = {
            self.avg_report[split][key]:
            np.array(avg_report[key], dtype=np.float32).mean()
            for key in self.avg_report[split]}
        summary_op = self.summary_ops['average_{}'.format(split)]
        step, avg_summary = self.session.run([self.global_step, summary_op],
                                             feed_dict=feed_dict)
        return step, avg_summary

    def run_train_step(self, use_heavy_summary):
        if use_heavy_summary:
            summary_op = self.summary_ops['heavy_train']
        else: summary_op = self.summary_ops['no_op']

        _start_time = time.time()
        fetch = [self.global_step, summary_op,
                 self.model.loss, self.model.report, self.optimizer]
        fetch_values = self.session.run(fetch,
                                        feed_dict={self.target_split: 'train'})
        [step, summary, loss, report] = fetch_values[:4]
        _end_time = time.time()
        return step, summary, loss, report, (_end_time - _start_time)

    def run_val_step(self, use_heavy_summary, split):
        if use_heavy_summary:
            summary_op = self.summary_ops['heavy_{}'.format(split)]
        else: summary_op = self.summary_ops['no_op']

        _start_time = time.time()
        fetch = [self.global_step, summary_op, self.model.loss, self.model.report]
        fetch_values = self.session.run(
            fetch, feed_dict={self.target_split: split})
        [step, summary, loss, report] = fetch_values[:4]
        _end_time = time.time()
        return step, summary, loss, report, (_end_time - _start_time)

    def log_message(self, step, avg_report, avg_step_time, split='train', is_train=True):
        step_time = np.array(avg_step_time, dtype=np.float32).mean()
        if step_time == 0: step_time = 0.001
        log_str = ''
        log_str += '[{:5s} step {:4d} '.format(split, step)
        log_str += '({:.3f} sec/batch, {:.3f} instances/sec)]\n'.format(
            step_time, self.batch_size / step_time)
        for key in sorted(avg_report.keys()):
            report = np.array(avg_report[key], dtype=np.float32).mean()
            log_str += '  * {}: {:.5f}\n'.format(key, report)
        log_fn = (log.info if is_train else log.infov)
        log_fn(log_str)


def check_config(config):
    if config.checkpoint is not None and config.pretrained_param_path is not None:
        raise ValueError('Do not set both checkpoint and pretrained_param_path')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # paths
    parser.add_argument('--image_dir', type=str, default='data/VQA_v2/images',
                        help=' ')
    parser.add_argument('--tf_record_dir', type=str,
                        default='data/preprocessed/vqa_v2'
                        '/qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1'
                        '/tf_record_memft', help=' ')
    parser.add_argument('--vfeat_name', type=str,
                        default='vfeat_bottomup_36.hdf5', help=' ')
    parser.add_argument('--vocab_name', type=str, default='vocab.pkl', help=' ')
    # log
    parser.add_argument('--max_train_iter', type=int, default=6500)
    parser.add_argument('--train_average_iter', type=int, default=200)
    parser.add_argument('--val_average_iter', type=int, default=419)  # 419 for 1 epoch
    parser.add_argument('--heavy_summary_step', type=int, default=400)  # 867 for 1 epoch
    parser.add_argument('--validation_step', type=int, default=400)
    parser.add_argument('--checkpoint_step', type=int, default=400)
    # hyper parameters
    parser.add_argument('--prefix', type=str, default='default', help=' ')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pretrained_param_path', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.001, help=' ')
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    # model parameters
    parser.add_argument('--batch_size', type=int, default=512, help=' ')
    parser.add_argument('--model_type', type=str, default='proposed',
                        help=' ', choices=importer.get_model_types())
    # model specific parameters
    parser.add_argument('--pretrain_word_weight_dir', type=str, default=None,
                        help=' ')
    parser.add_argument('--ft_pretrain', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=123, help=' ')
    parser.add_argument('--debug', type=int, default=0, help='0: normal, 1: debug')

    config = parser.parse_args()
    config.vocab_path = os.path.join(config.tf_record_dir, config.vocab_name)
    config.vfeat_path = os.path.join(config.tf_record_dir, config.vfeat_name)
    check_config(config)

    # Set random seed
    tf.set_random_seed(config.seed)
    np.random.seed(config.seed)

    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
