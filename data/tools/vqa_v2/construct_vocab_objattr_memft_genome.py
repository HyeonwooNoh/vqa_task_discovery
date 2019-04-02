import argparse
import collections
import cPickle
import json
import os

from tqdm import tqdm

from util import log

GLOVE_VOCAB_PATH = 'data/preprocessed/glove_vocab.json'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--qa_split_dir', type=str, default='data/preprocessed/vqa_v2'
    '/qa_split_objattr_answer_3div4_genome_memft_check_all_answer_thres1_50000_thres2_-1',
    help=' ')
parser.add_argument('--answer_set_limit', type=int, default=3000, help=' ')
parser.add_argument('--max_answer_len', type=int, default=3, help=' ')
config = parser.parse_args()

log.info('loading merged_annotations..')
merged_anno_path = os.path.join(config.qa_split_dir, 'merged_annotations.pkl')
qid2anno = cPickle.load(open(merged_anno_path, 'rb'))
log.info('loading glove_vocab..')
glove_vocab = json.load(open(GLOVE_VOCAB_PATH, 'r'))
log.info('done')

obj_attrs_split_path = os.path.join(config.qa_split_dir, 'obj_attrs_split.pkl')
obj_attrs_split = cPickle.load(open(obj_attrs_split_path, 'rb'))

"""
Filtering qa:
    - count answer occurrences and filter rare answers
      (filtering answer candidates is for efficiency of answer classification)
    - make answer set with only frequent answers
    - make sure that every chosen answers consist of glove vocabs
    - all questions are used for vocab construction
*: When making GT, let's make N(answer) + 1 slots and if answer is not in answer
set, mark gt to N(answer) + 1 slot (only for testing data). For training data,
we will just ignore qa with rare answers.
"""

glove_vocab_set = set(glove_vocab['vocab'])

answers = list()
for anno in tqdm(qid2anno.values(), desc='count answers'):
    answers.append(' '.join(anno['a_tokens']))
answer_counts = collections.Counter(answers)

ans_in_order = list(zip(*answer_counts.most_common())[0])
ans_in_order_glove = []
for ans in ans_in_order:
    in_glove = True
    a_tokens = ans.split()
    if len(a_tokens) > config.max_answer_len:
        continue
    for t in a_tokens:
        if t not in glove_vocab_set:
            in_glove = False
            break
    if in_glove:
        ans_in_order_glove.append(ans)

freq_ans = ans_in_order_glove[:config.answer_set_limit]
freq_ans_set = set(freq_ans)

q_vocab = set()
for anno in tqdm(qid2anno.values(), desc='count q_vocab'):
    for t in anno['q_tokens']: q_vocab.add(t)
q_vocab = q_vocab & glove_vocab_set

a_vocab = set()
for ans in tqdm(freq_ans, desc='count a_vocab'):
    for t in ans.split(): a_vocab.add(t)
if len(a_vocab - glove_vocab_set) > 0:
    raise RuntimeError('Somethings wrong: a_vocab is already filtered')
qa_vocab = q_vocab | a_vocab
qa_vocab = list(qa_vocab)
qa_vocab.append('<s>')
qa_vocab.append('<e>')
qa_vocab.append('<unk>')

"""
How to store vocab:
    - ['vocab'] = ['yes', 'no', 'apple', ...]
    - ['dict'] = {'yes': 0, 'no: 1, 'apple': 2, ...}
    - in a json format
"""
save_vocab = {}
save_vocab['vocab'] = qa_vocab
save_vocab['dict'] = {v: i for i, v in enumerate(qa_vocab)}

vocab_path = os.path.join(config.qa_split_dir, 'vocab.pkl')
log.warn('save vocab: {}'.format(vocab_path))
cPickle.dump(save_vocab, open(vocab_path, 'wb'))

test_object_set = set(obj_attrs_split['test'])
train_ans_set = freq_ans_set - test_object_set
test_ans_set = freq_ans_set & test_object_set

log.info('loading object and attribute list')
obj_list_path = os.path.join(config.qa_split_dir, 'object_list.pkl')
obj_list = cPickle.load(open(obj_list_path, 'rb'))
attr_list_path = os.path.join(config.qa_split_dir, 'attribute_list.pkl')
attr_list = cPickle.load(open(attr_list_path, 'rb'))
obj_set = set(obj_list)
attr_set = set(attr_list)

answer_dict = {}
answer_dict['vocab'] = list(train_ans_set) + list(test_ans_set)
answer_dict['num_train_answer'] = len(train_ans_set)
answer_dict['dict'] = {v: i for i, v in enumerate(answer_dict['vocab'])}
answer_dict['is_object'] = [int(v in obj_set) for v in answer_dict['vocab']]
answer_dict['is_attribute'] = [int(v in attr_set) for v in answer_dict['vocab']]

answer_dict_path = os.path.join(config.qa_split_dir, 'answer_dict.pkl')
log.warn('save answer_dict: {}'.format(answer_dict_path))
cPickle.dump(answer_dict, open(answer_dict_path, 'wb'))

log.warn('done')
