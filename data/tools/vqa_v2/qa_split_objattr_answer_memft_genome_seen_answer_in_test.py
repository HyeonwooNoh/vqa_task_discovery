import argparse
import cPickle
import json
import os
import re
import numpy as np

from tqdm import tqdm
from util import log


RANDOM_STATE = np.random.RandomState(123)


QUESTION_PATHS = {
    'train': 'data/VQA_v2/questions'
    '/v2_OpenEnded_mscoco_train2014_questions.json',
    'val': 'data/VQA_v2/questions'
    '/v2_OpenEnded_mscoco_val2014_questions.json',
}
ANNOTATION_PATHS = {
    'train': 'data/VQA_v2/annotations'
    '/v2_mscoco_train2014_annotations.json',
    'val': 'data/VQA_v2/annotations'
    '/v2_mscoco_val2014_annotations.json',
}

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--genome_memft_dir', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
parser.add_argument('--occ_thres_1', type=int, default=50000,
                    help='object classes with occurrence greater or equal to '
                    'this threshold are splited into train')
parser.add_argument('--occ_thres_2', type=int, default=-1,
                    help='object classes with occurrence greater or equal to '
                    'this threshold are splited into train and test. If '
                    'occurrence is smaller, they are splited into train-reserve '
                    'and test.')
parser.add_argument('--save_split_dir', type=str,
                    default='data/preprocessed/vqa_v2'
                    '/qa_split_objattr_answer_3div4_genome_memft_check_all_answer', help=' ')
config = parser.parse_args()

config.save_split_dir += '_thres1_{}'.format(config.occ_thres_1)
config.save_split_dir += '_thres2_{}'.format(config.occ_thres_2)
config.save_split_dir += '_with_seen_answer_in_test'

if not os.path.exists(config.save_split_dir):
    log.warn('Create directory: {}'.format(config.save_split_dir))
    os.makedirs(config.save_split_dir)
else:
    raise ValueError('Do not overwrite: {}'.format(config.save_split_dir))

config.obj_list_path = os.path.join(
    config.genome_memft_dir, 'object_list.pkl')
config.attr_list_path = os.path.join(
    config.genome_memft_dir, 'attribute_list.pkl')
objects = cPickle.load(open(config.obj_list_path, 'rb'))
attrs = cPickle.load(open(config.attr_list_path, 'rb'))
obj_attrs = objects + attrs

"""
When we split obj_attrs, we consider every object classes independent. Therefore,
splits such as "right leg" and "left leg" could happen. This is because making
disjoint set based on the vocabulary is difficult as most vocabulary is
connected to each other and more than 900 vocabulary end up belongs to the same
clusters. For splits including subsets such as "leg" and "right leg", we allocate
questions to longest object name split.
"""
questions = {}
for key, path in tqdm(QUESTION_PATHS.items(), desc='Loading questions'):
    questions[key] = json.load(open(path, 'r'))
merge_questions = []
for key, entry in questions.items():
    merge_questions.extend(entry['questions'])

annotations = {}
for key, path in tqdm(ANNOTATION_PATHS.items(), desc='Loading annotations'):
    annotations[key] = json.load(open(path, 'r'))
merge_annotations = []
for key, entry in annotations.items():
    data_subtype = entry['data_subtype']
    for anno in tqdm(entry['annotations'], desc='Annotation {}'.format(key)):
        anno['image_path'] = '{}/COCO_{}_{:012d}.jpg'.format(
            data_subtype, data_subtype, anno['image_id'])
        anno['split'] = data_subtype
        merge_annotations.append(anno)

qid2anno = {a['question_id']: a for a in merge_annotations}
for q in tqdm(merge_questions, desc='merge question and annotations'):
    qid2anno[q['question_id']]['question'] = q['question']

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = {'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
              'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
         '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']


def split_with_punctuation(string):
    return re.findall(r"'s+|[\w]+|[.,!?;-]", string)


def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

for qid in tqdm(qid2anno.keys(), desc='tokenize QA'):
    anno = qid2anno[qid]
    qid2anno[qid]['q_tokens'] = split_with_punctuation(
        anno['question'].lower())
    qid2anno[qid]['a_tokens'] = split_with_punctuation(
        preprocess_answer(anno['multiple_choice_answer'].lower()))
    processed_answers = []
    for answer in anno['answers']:
        a_tokens = split_with_punctuation(
            preprocess_answer(answer['answer'].lower()))
        processed_answers.append(' '.join(a_tokens))
    qid2anno[qid]['processed_answers'] = processed_answers

"""
Test question or answer could have training object words, but it should contain
at least one test object words, which is unseen during training.
"""


def get_ngrams(tokens, n):
    ngrams = []
    for i in range(0, len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i: i+n]))
    return ngrams

occurrence = {name: 0 for name in obj_attrs}
max_name_len = max([len(name.split()) for name in obj_attrs])
qid2ngrams = {}
for qid, anno in tqdm(qid2anno.items(), desc='count object occurrence'):
    processed_answers = anno['processed_answers']
    ngrams = []
    for processed_answer in anno['processed_answers']:
        if str.isdigit(str(processed_answer)):
            continue
        ngrams.append(processed_answer)
    ngrams = list(set(ngrams))
    for ngram in ngrams:
        if ngram in occurrence: occurrence[ngram] += 1
    qid2ngrams[qid] = ngrams
"""
How we could split object classes based on occurrence in questions and answers?
1. Generalization to rare words: have rare obj_attrs in the test set.
This way of spliting classes well matches with the motivation:
    Collecting VQA training data for all rare object classes are difficult, but
    there are rich set of object annotations.
    So transfer to rare words make sense.
Why should we keep very frequent words:
    This is to run reasoning well. VQA training data with very rare words only
    teaches the biases. But rich words appears with various forms of questions
    and this help to learn real reasoning, not a bias.
* How we will split data:
    1. occurence >= threshold_1: always training set
    2. threshold_1 > occurrence >= threshold_2: random split
    3. threshold_2 > occurrence: always testing set

c.f)
Among 3000 classes
top 500 occurrence object classes have more than 500 occurrences
if we pick top 1650-th occurrece object, its occurrene is around 50.
Therefore, with threshold_1 == 500, threshold_2 == 50,
around 487 classes are always 'train'
around 1167 classes are randomly splited into 'train' and 'test'
around 3156 classes are randomly splited into 'train-reserve' and 'test'
(train-reserve could be used for training or not, based on the experiment result)
"""
log.warn('Split obj_attrs')
obj_grp1 = [name for name, occ in occurrence.items()
            if occ >= max(config.occ_thres_1, 1)]
log.warn('# obj_grp1: {}'.format(len(obj_grp1)))
obj_grp2 = [name for name, occ in occurrence.items()
            if config.occ_thres_1 > occ >= max(config.occ_thres_2, 1)]
log.warn('# obj_grp2: {}'.format(len(obj_grp2)))
obj_grp3 = [name for name, occ in occurrence.items()
            if config.occ_thres_2 > occ >= 1]
log.warn('# obj_grp3: {}'.format(len(obj_grp3)))
RANDOM_STATE.shuffle(obj_grp1)
RANDOM_STATE.shuffle(obj_grp2)
half2 = len(obj_grp2) * 3 / 4
RANDOM_STATE.shuffle(obj_grp3)
half3 = len(obj_grp3) * 3 / 4
obj_attrs_split = {
    'train': obj_grp1 + obj_grp2[:half2],
    'train-reserve': obj_grp3[:half3],
    'test': obj_grp2[half2:] + obj_grp3[half3:],
}
obj_attrs_split_set = {key: set(val) for key, val in obj_attrs_split.items()}
log.infov('train object: {}'.format(len(obj_attrs_split['train'])))
log.info('ex)')
for obj in obj_attrs_split['train'][:5]: log.info(obj)

log.infov('train-reserve object: {}'.format(len(obj_attrs_split['train-reserve'])))
log.info('ex)')
for obj in obj_attrs_split['train-reserve'][:5]: log.info(obj)

log.infov('test object: {}'.format(len(obj_attrs_split['test'])))
log.info('ex)')
for obj in obj_attrs_split['test'][:5]: log.info(obj)


def filter_qids_by_object_split(qids, split):
    filtered_qids = []
    for qid in tqdm(qids, desc='mark {} QA'.format(split)):
        ngrams = qid2ngrams[qid]
        is_target_split = False
        for ngram in ngrams:
            if ngram in obj_attrs_split_set[split]:
                is_target_split = True
                break
        if is_target_split: filtered_qids.append(qid)
    return filtered_qids

qids = qid2anno.keys()

log.warn('Mark test QA')
test_qids = filter_qids_by_object_split(qids, 'test')
left_qids = list(set(qids) - set(test_qids))
log.infov('{} question ids are marked for test obj_attrs'.format(len(test_qids)))

log.warn('Mark train-reserve QA')
train_reserve_qids = filter_qids_by_object_split(left_qids, 'train-reserve')
train_qids = list(set(left_qids) - set(train_reserve_qids))
log.infov('{} question ids are marked for train-reserve obj_attrs'.format(
    len(train_reserve_qids)))
log.infov('{} question ids are marked for train obj_attrs'.format(
    len(train_qids)))

log.warn('Shuffle qids')
RANDOM_STATE.shuffle(train_qids)
RANDOM_STATE.shuffle(train_reserve_qids)
RANDOM_STATE.shuffle(test_qids)

train_90p = len(train_qids) * 90 / 100
train_reserve_90p = len(train_reserve_qids) * 90 / 100
test_80p = len(test_qids) * 80 / 100

qid_splits = {}
log.warn('Split testval / test')
qid_splits['testval'] = test_qids[test_80p:]
qid_splits['test'] = test_qids[:test_80p]

log.warn('Split train / val')
qid_splits['val'] = train_qids[train_90p:]
qid_splits['train'] = train_qids[:train_90p]
qid_splits['val-reserve'] = train_reserve_qids[train_reserve_90p:]
qid_splits['train-reserve'] = train_reserve_qids[:train_reserve_90p]

RANDOM_STATE.shuffle(qid_splits['testval'])
RANDOM_STATE.shuffle(qid_splits['test'])
RANDOM_STATE.shuffle(qid_splits['val'])
RANDOM_STATE.shuffle(qid_splits['train'])
RANDOM_STATE.shuffle(qid_splits['val-reserve'])
RANDOM_STATE.shuffle(qid_splits['train-reserve'])

move_to_testval = len(qid_splits['testval']) / 2
move_to_test = len(qid_splits['test']) / 2

qid_splits['testval'] = qid_splits['testval'] + qid_splits['val'][-move_to_testval:]
qid_splits['val'] = qid_splits['val'][:-move_to_testval]
qid_splits['test'] = qid_splits['test'] + qid_splits['train'][-move_to_test:]
qid_splits['train'] = qid_splits['train'][:-move_to_test]


log.infov('test: {}, testval: {}'.format(
    len(qid_splits['test']), len(qid_splits['testval'])))
log.infov('train: {}, val: {}'.format(
    len(qid_splits['train']), len(qid_splits['val'])))
log.infov('train-reserve: {}, val-reserve: {}'.format(
    len(qid_splits['train-reserve']), len(qid_splits['val-reserve'])))

# used image ids
used_image_paths = []
for anno in tqdm(qid2anno.values(), desc='construct used_image_ids'):
    used_image_paths.append(anno['image_path'])
used_image_paths = list(set(used_image_paths))
log.infov('used_image_paths: {}'.format(len(used_image_paths)))

"""
What to save:
    - used image ids (for efficient feature extraction)
    - object splits (train , train-reserve, test)
    - qa splits qids (train, val, train-reserve, val-reserve, testval, test)
    - annotations: merged annotations is saved for evaluation / future usage
"""
with open(os.path.join(config.save_split_dir, 'used_image_path.txt'), 'w') as f:
    for image_path in used_image_paths:
        f.write(image_path + '\n')
cPickle.dump(obj_attrs_split, open(os.path.join(
    config.save_split_dir, 'obj_attrs_split.pkl'), 'wb'))
cPickle.dump(qid_splits, open(os.path.join(
    config.save_split_dir, 'qa_split.pkl'), 'wb'))
cPickle.dump(qid2anno, open(os.path.join(
    config.save_split_dir, 'merged_annotations.pkl'), 'wb'))
cPickle.dump(objects, open(os.path.join(
    config.save_split_dir, 'object_list.pkl'), 'wb'))
cPickle.dump(attrs, open(os.path.join(
    config.save_split_dir, 'attribute_list.pkl'), 'wb'))
log.warn('output saving is done.')
