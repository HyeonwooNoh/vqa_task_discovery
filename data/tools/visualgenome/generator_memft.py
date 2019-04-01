"""
Generator for on memory pre-training of vlmap

each entry contains bounding boxes and object, attribute, description annotations.
36 bounding boxes are extracted using "bottom_up attention".

pre-training aim to two tasks: attention / answer:
    - attention: language -> box relevant score (36 scalar)
    - answer: 36 boxes -> pooled vector -> answer

For each data, following information should be encoded:
    - non-zero scores for each bounding box
    - weight for each box for feature pooling
    - object, attribute, description annotations
"""
import argparse
import cPickle
import h5py
import json
import os
import numpy as np

from collections import Counter
from tqdm import tqdm

from data.tools import tools
from util import box_utils

RANDOM_STATE = np.random.RandomState(123)

ANNO_FILES = {
    'object': 'data/VisualGenome/annotations/objects.json',
    'attribute': 'data/VisualGenome/annotations/attributes.json',
    'caption': 'data/VisualGenome/annotations/region_descriptions.json',
}
IMAGE_DATA_PATH = 'data/VisualGenome/annotations/image_data.json'

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vocab_path', type=str,
                    default='data/preprocessed/new_vocab50.json', help=' ')
parser.add_argument('--bottomup_data_dir', type=str,
                    default='data/VisualGenome/bottomup_feature_36', help=' ')
parser.add_argument('--dir_name', type=str, default='memft_all', help=' ')
parser.add_argument('--num_object', type=int, default=3000, help=' ')
parser.add_argument('--num_attribute', type=int, default=1000, help=' ')
parser.add_argument('--max_description_length', type=int, default=10, help=' ')
config = parser.parse_args()

config.dir_name = os.path.join('data/preprocessed/visualgenome', config.dir_name)
config.dir_name += '_{}'.format(
    config.vocab_path.replace('data/preprocessed/', '').replace('.json', ''))
config.dir_name += '_obj{}'.format(config.num_object)
config.dir_name += '_attr{}'.format(config.num_attribute)
if config.max_description_length > 0:
    config.dir_name += '_maxlen{}'.format(config.max_description_length)

if not os.path.exists(config.dir_name): os.makedirs(config.dir_name)
else: raise ValueError('Do not overwrite {}'.format(config.dir_name))

config.save_vocab_path = os.path.join(config.dir_name, 'vocab.pkl')
config.save_answer_dict = os.path.join(config.dir_name, 'answer_dict.pkl')
config.save_image_split = os.path.join(config.dir_name, 'image_split.pkl')
config.save_processed = os.path.join(config.dir_name, 'image_id2processed.pkl')

image_data = json.load(open(IMAGE_DATA_PATH, 'r'))
image_id2data = {e['image_id']: e for e in image_data}

vocab = json.load(open(config.vocab_path, 'r'))
vocab_set = set(vocab['vocab'])

cPickle.dump(vocab, open(config.save_vocab_path, 'wb'))


def check_name(name):
    name = tools.clean_answer_word(name)
    passed = len(name) > 0 and all([n in vocab_set for n in name.split()])
    return passed, name


def check_and_add(name, name_list):
    name = tools.clean_answer_word(name)
    if len(name) > 0 and all([n in vocab_set for n in name.split()]):
        name_list.append(name)


def check_caption(caption):
    caption = tools.clean_description(caption)
    passed = len(caption) > 0 and all([n in vocab_set for n in caption.split()])
    if config.max_description_length > 0:
        passed = passed and (len(caption.split()) <= config.max_description_length)
    return passed, caption


def check_and_add_description(name, name_list):
    name = tools.clean_description(name)
    if len(name) > 0 and all([n in vocab_set for n in name.split()]):
        name_list.append(name)

annotations = {}
for key, anno_fn in tqdm(ANNO_FILES.items(), desc='loading anno..'):
    annotations[key] = json.load(open(anno_fn, 'r'))

config.vfeat_path = os.path.join(config.bottomup_data_dir,
                                 'vfeat_bottomup_36.hdf5')
config.image_info_path = os.path.join(config.bottomup_data_dir,
                                      'image_info.json')

vfeat_h5 = h5py.File(config.vfeat_path, 'r')

image_info = json.load(open(config.image_info_path, 'r'))
image_id2idx = image_info['image_id2idx']

"""
process objects
"""
obj_blacklist = set(['is', 'it', 'red', 'yellow', 'black', 'blue', 'green', 'pink',
                     'orange', 'purple', 'brown', 'white', 'gray', 'grey',
                     'gold', 'silver', 'tall', 'long', 'short', 'big', 'small',
                     'left', 'right', 'up', 'down', 'middle', 'of', 'on'])

color_words = set(['red', 'yellow', 'black', 'blue', 'green', 'pink',
                   'orange', 'purple', 'brown', 'white', 'gray', 'grey',
                   'gold', 'silver'])


def strip_color(name):
    tokens = name.split()
    if len(tokens) == 2:
        if tokens[0] in color_words:
            return tokens[0], tokens[1]
        else:
            return None, name
    else:
        return None, name


def strip_number(name):
    tokens = name.split()
    if len(tokens) == 2:
        if str.isdigit(str(tokens[0])):
            return tokens[0], tokens[1]
        else:
            return None, name
    else:
        return None, name

freq_obj = []
for entry in tqdm(annotations['object'], desc='process obj1'):
    for e in entry['objects']:
        is_passed, name = check_name(e['names'][0])
        if is_passed and (name not in obj_blacklist) and (not str.isdigit(str(name))):
            color_w, name = strip_color(name)
            digit_w, name = strip_number(name)
            e['processed_name'] = name
            freq_obj.append(name)
freq_obj = Counter(freq_obj)
freq_obj = dict(freq_obj.most_common()[:3000])  # use top 3000 objects
freq_obj_set = set(freq_obj.keys())

for entry in tqdm(annotations['object'], desc='process obj2'):
    for e in entry['objects']:
        if 'processed_name' in e:
            if e['processed_name'] not in freq_obj_set:
                del e['processed_name']

image_id2objects = {}
for entry in tqdm(annotations['object'], desc='process obj3'):
    image_id = entry['image_id']
    if image_id not in image_id2objects:
        image_id2objects[image_id] = []
    for e in entry['objects']:
        if 'processed_name' in e:
            image_id2objects[image_id].append(e)

"""
process attributes
"""
obj2attr_list = {}
for entry in tqdm(annotations['attribute'], desc='process attr1'):
    for e in entry['attributes']:
        if 'names' not in e or len(e['names']) != 1:
            continue
        if 'attributes' not in e:
            continue
        passed, processed_name = check_name(e['names'][0])
        if (not passed) or (processed_name not in freq_obj_set):
            continue
        processed_attributes = set()
        color_w, processed_name = strip_color(processed_name)
        if color_w is not None:
            processed_attributes.add(color_w)
        digit_w, processed_name = strip_number(processed_name)
        if digit_w is not None:
            processed_attributes.add(digit_w)
        for attr in e['attributes']:
            passed, processed_attr = check_name(attr)
            if passed and (processed_attr not in freq_obj_set):
                processed_attributes.add(processed_attr)
        processed_attributes = list(processed_attributes)
        if len(processed_attributes) == 0:
            continue
        e['processed_name'] = processed_name
        e['processed_attributes'] = processed_attributes
        if processed_name not in obj2attr_list:
            obj2attr_list[processed_name] = set()
        for attr in processed_attributes:
            obj2attr_list[processed_name].add(attr)

freq_attr = []
attr_blacklist = set(['is', 'it', 'up', 'down', 'of', 'on', 'under', 'at', 'from',
                      'a', 'an', 'in'])
for entry in tqdm(annotations['attribute'], desc='process attr2'):
    for e in entry['attributes']:
        if 'processed_name' not in e: continue
        if 'processed_attributes' not in e: continue
        for attr in e['processed_attributes']:
            if attr not in attr_blacklist:
                freq_attr.append(attr)
freq_attr = Counter(freq_attr)
freq_attr = dict(freq_attr.most_common()[:1000])  # use top 1000 attributes
freq_attr_set = set(freq_attr.keys())

obj2attr_list = {}
for entry in tqdm(annotations['attribute'], desc='process attr3'):
    for e in entry['attributes']:
        if 'processed_name' not in e: continue
        if 'processed_attributes' not in e: continue
        name = e['processed_name']
        if name not in obj2attr_list:
            obj2attr_list[name] = set()
        processed_attributes = set()
        for attr in e['processed_attributes']:
            if attr in freq_attr_set:
                obj2attr_list[name].add(attr)
                processed_attributes.add(attr)
        processed_attributes = list(processed_attributes)
        if len(processed_attributes) == 0:
            del e['processed_attributes']
        else: e['processed_attributes'] = processed_attributes

image_id2attrs = {}
for entry in tqdm(annotations['attribute'], desc='process attr4'):
    image_id = entry['image_id']
    if image_id not in image_id2attrs:
        image_id2attrs[image_id] = []
    for e in entry['attributes']:
        if 'processed_name' in e and 'processed_attributes' in e:
            image_id2attrs[image_id].append(e)

"""
process descriptions
"""
obj_attr_set = freq_obj_set | freq_attr_set
answer_dict = {}
answer_dict['vocab'] = list(obj_attr_set)
answer_dict['dict'] = {v: i for i, v in enumerate(answer_dict['vocab'])}
cPickle.dump(answer_dict, open(config.save_answer_dict, 'wb'))

obj_list = list(freq_obj_set)
cPickle.dump(obj_list,
             open(os.path.join(config.dir_name, 'object_list.pkl'), 'wb'))
attr_list = list(freq_attr_set)
cPickle.dump(attr_list,
             open(os.path.join(config.dir_name, 'attribute_list.pkl'), 'wb'))

vocab2obj = {}
for obj in freq_obj_set:
    for t in obj.split():
        if t not in vocab2obj:
            vocab2obj[t] = set()
        vocab2obj[t].add(obj)

vocab2attr = {}
for attr in freq_attr_set:
    for t in attr.split():
        if t not in vocab2attr:
            vocab2attr[t] = set()
        vocab2attr[t].add(attr)


def filter_longest(candidates):
    cand_list = list(candidates)
    longest = [True] * len(cand_list)
    for i in range(len(cand_list)):
        for j in range(len(cand_list)):
            if i != j and cand_list[j].count(cand_list[i]) > 0:
                longest[i] = False
    filtered_cand = [c for c, l in zip(cand_list, longest) if l]
    return filtered_cand

for entry in tqdm(annotations['caption'], desc='process caption1'):
    for e in entry['regions']:
        if 'phrase' not in e:
            continue
        passed, caption = check_caption(e['phrase'])
        if not passed:
            continue
        e['caption'] = caption
        obj_candidates = set()
        for t in caption.split():
            if t in vocab2obj:
                obj_candidates = obj_candidates | vocab2obj[t]

        matched_obj_candidates = set()
        for obj in obj_candidates:
            if caption.count(obj) > 0:
                matched_obj_candidates.add(obj)

        obj_cand = filter_longest(matched_obj_candidates)
        obj_blank_fill = [{
            'blank': ' {} '.format(caption).replace(
                ' {} '.format(cand), ' <unk> ')[1: -1],
            'fill': cand} for cand in obj_cand]
        e['obj_blank_fill'] = obj_blank_fill

        attr_candidates = set()
        for t in caption.split():
            if t in vocab2attr:
                attr_candidates = attr_candidates | vocab2attr[t]

        matched_attr_candidates = set()
        for attr in attr_candidates:
            if caption.count(attr) > 0:
                matched_attr_candidates.add(attr)

        attr_cand = filter_longest(matched_attr_candidates)
        attr_blank_fill = [{
            'blank': ' {} '.format(caption).replace(
                ' {} '.format(cand), ' <unk> ')[1: -1],
            'fill': cand} for cand in attr_cand]
        e['attr_blank_fill'] = attr_blank_fill

image_id2captions = {}
for entry in tqdm(annotations['caption'], desc='process caption2'):
    for e in entry['regions']:
        image_id = e['image_id']
        if image_id not in image_id2captions:
            image_id2captions[image_id] = []
        if 'caption' in e:
            image_id2captions[image_id].append(e)

num_boxes = np.array(vfeat_h5.get('num_boxes'))
normal_boxes = np.array(vfeat_h5.get('normal_boxes'))

image_id2processed = {}
for image_id, image_idx in tqdm(image_id2idx.items(), desc='merge all'):
    image_id = int(image_id)
    image_meta = image_id2data[image_id]
    image_w = image_meta['width']
    image_h = image_meta['height']

    num_box = num_boxes[image_idx]
    normal_box = normal_boxes[image_idx]
    box = box_utils.scale_boxes_x1y1x2y2(normal_box, [image_w, image_h])

    objects = image_id2objects[image_id]
    processed_objs = []
    for obj in objects:
        x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
        obj_box = box_utils.xywh_to_x1y1x2y2(
            np.expand_dims(np.array([x, y, w, h], dtype=np.float32), axis=0))
        iou = np.squeeze(box_utils.iou_matrix(box, obj_box), axis=1)
        is_inside = np.squeeze(box_utils.is_inside_matrix(box, obj_box), axis=1)
        is_inside = np.where(is_inside < 0.7, 0, is_inside)
        obj['box'] = np.squeeze(obj_box, axis=0)
        obj['normal_box'] = box_utils.normalize_box_x1y1x2y2(
            obj['box'], image_w, image_h)
        obj['iou'] = iou
        obj['is_inside'] = is_inside
        if is_inside.max() > 0.8:
            processed_objs.append(obj)
    if len(processed_objs) == 0:
        continue

    obj_name2score = {}
    for obj in processed_objs:
        name = obj['processed_name']
        if name in obj_name2score:
            obj_name2score[name] = \
                np.maximum(obj_name2score[name],
                           obj['is_inside'])
        else:
            obj_name2score[name] = obj['is_inside']
    obj_name2score = obj_name2score.items()

    attributes = image_id2attrs[image_id]
    processed_attrs = []
    for attr in attributes:
        x, y, w, h = attr['x'], attr['y'], attr['w'], attr['h']
        attr_box = box_utils.xywh_to_x1y1x2y2(
            np.expand_dims(np.array([x, y, w, h], dtype=np.float32), axis=0))
        is_inside = np.squeeze(box_utils.is_inside_matrix(box, attr_box), axis=1)
        is_inside = np.where(is_inside < 0.7, 0, is_inside)
        attr['box'] = np.squeeze(attr_box, axis=0)
        attr['normal_box'] = box_utils.normalize_box_x1y1x2y2(
            attr['box'], image_w, image_h)
        attr['is_inside'] = is_inside
        if is_inside.max() > 0.8:
            processed_attrs.append(attr)
    if len(processed_attrs) == 0:
        continue

    attr_name2score = {}
    for attr in processed_attrs:
        name = attr['processed_name']
        for att in attr['processed_attributes']:
            name_attr = ' '.join([att, name])
            if name_attr in attr_name2score:
                attr_name2score[name_attr] = \
                    np.maximum(attr_name2score[name_attr],
                               attr['is_inside'])
            else:
                attr_name2score[name_attr] = attr['is_inside']
    attr_name2score = attr_name2score.items()

    captions = image_id2captions[image_id]
    processed_caps = []
    for cap in captions:
        x, y, w, h = cap['x'], cap['y'], cap['width'], cap['height']
        cap_box = box_utils.xywh_to_x1y1x2y2(
            np.expand_dims(np.array([x, y, w, h], dtype=np.float32), axis=0))
        iou = np.squeeze(box_utils.iou_matrix(box, cap_box), axis=1)
        is_inside = np.squeeze(box_utils.is_inside_matrix(box, cap_box), axis=1)
        is_inside = np.where(is_inside < 0.7, 0, is_inside)
        cap['box'] = np.squeeze(cap_box, axis=0)
        cap['normal_box'] = box_utils.normalize_box_x1y1x2y2(
            cap['box'], image_w, image_h)
        cap['iou'] = iou
        cap['is_inside'] = is_inside
        if is_inside.max() > 0.8:
            processed_caps.append(cap)
    if len(processed_caps) == 0:
        continue

    entry = {
        'image_id': image_id,
        'image_idx': image_idx,
        'processed_objs': processed_objs,
        'obj_name2score': obj_name2score,
        'processed_attrs': processed_attrs,
        'attr_name2score': attr_name2score,
        'processed_caps': processed_caps,
    }
    image_id2processed[image_id] = entry

cPickle.dump(image_id2processed, open(config.save_processed, 'wb'))
# distribution of number of annotations
# objs: 1: 1835, 2: 5336, 3: 9385, 4: 12550, 5: 13762, 6: 13118, 7: 10525, 8:
# 8125, 9: 5646
# attrs: 1: 19533, 2: 20183, 3: 17162, 4: 12483, 5: 8139, 6: 4806, 7: 3031, 8:
# 1802, 9: 1108
# caps: 1: 251, 2: 523, 3: 949, 4: 1484, 5: 2205, 6: 2786, 7: 3451, 8: 4129, 9:
# 4756, 10: 5284, 11: 5400, 12: 5555, 13: 5726, ...

new_image_ids = list(set(image_id2processed.keys()))

RANDOM_STATE.shuffle(new_image_ids)
num_train = int(len(new_image_ids) * 0.9)
image_split = {
    'train': new_image_ids[:num_train],
    'val': new_image_ids[num_train:],
}
for split in ['train', 'val']:
    processed = {}
    for image_id in tqdm(image_split[split], desc='processing {}'.format(split)):
        entry = image_id2processed[image_id]
        object_predict = []
        for obj in entry['processed_objs']:
            normal_box = list(obj['normal_box'])
            proposal_weights = list(obj['is_inside'] / obj['is_inside'].sum())
            p_idx, p_weight = zip(
                *[(i, s) for i, s in enumerate(proposal_weights) if s > 0])
            label = answer_dict['dict'][obj['processed_name']]
            item = {
                'normal_box': normal_box,
                'p_idx': list(p_idx),
                'p_weight': list(p_weight),
                'label': label,
            }
            object_predict.append(item)
        if len(object_predict) == 0: continue

        object_attend = []
        for obj_name, att_score in entry['obj_name2score']:
            s_idx, s_value = zip(
                *[(i, s) for i, s in enumerate(att_score) if s > 0])
            w_tokens = [vocab['dict'][t] for t in obj_name.split()]
            item = {
                's_idx': list(s_idx),
                's_value': list(s_value),
                'w_tokens': w_tokens,
            }
            object_attend.append(item)
        if len(object_attend) == 0: continue

        attr_predict = []
        for attr in entry['processed_attrs']:
            normal_box = list(attr['normal_box'])
            proposal_weights = list(attr['is_inside'] / attr['is_inside'].sum())
            p_idx, p_weight = zip(
                *[(i, s) for i, s in enumerate(proposal_weights) if s > 0])
            labels = [answer_dict['dict'][t] for t in attr['processed_attributes']]
            object_label = answer_dict['dict'][attr['processed_name']]
            item = {
                'normal_box': normal_box,
                'p_idx': list(p_idx),
                'p_weight': list(p_weight),
                'labels': labels,
                'object_label': object_label,
            }
            attr_predict.append(item)
        if len(attr_predict) == 0: continue

        attr_attend = []
        for attr_name, att_score in entry['attr_name2score']:
            s_idx, s_value = zip(
                *[(i, s) for i, s in enumerate(att_score) if s > 0])
            w_tokens = [vocab['dict'][t] for t in attr_name.split()]
            item = {
                's_idx': list(s_idx),
                's_value': list(s_value),
                'w_tokens': w_tokens,
            }
            attr_attend.append(item)
        if len(attr_attend) == 0: continue

        attr_blank_fill = []
        obj_blank_fill = []
        caption_attend = []
        for cap in entry['processed_caps']:
            normal_box = cap['normal_box']
            proposal_weights = list(cap['is_inside'] / cap['is_inside'].sum())
            p_idx, p_weight = zip(
                *[(i, s) for i, s in enumerate(proposal_weights) if s > 0])
            for blank_fill in cap['attr_blank_fill']:
                blank = [vocab['dict'][t] for t in blank_fill['blank'].split()]
                fill = answer_dict['dict'][blank_fill['fill']]
                item = {
                    'normal_box': normal_box,
                    'p_idx': list(p_idx),
                    'p_weight': list(p_weight),
                    'blank': blank,
                    'fill': fill,
                }
                attr_blank_fill.append(item)

            for blank_fill in cap['obj_blank_fill']:
                blank = [vocab['dict'][t] for t in blank_fill['blank'].split()]
                fill = answer_dict['dict'][blank_fill['fill']]
                item = {
                    'normal_box': normal_box,
                    'p_idx': list(p_idx),
                    'p_weight': list(p_weight),
                    'blank': blank,
                    'fill': fill,
                }
                obj_blank_fill.append(item)

            s_idx, s_value = zip(
                *[(i, s) for i, s in enumerate(cap['is_inside']) if s > 0])
            w_tokens = [vocab['dict'][t] for t in cap['caption'].split()]
            item = {
                's_idx': list(s_idx),
                's_value': list(s_value),
                'w_tokens': w_tokens,
            }
            caption_attend.append(item)
        if len(attr_blank_fill) == 0: continue
        if len(obj_blank_fill) == 0: continue
        if len(caption_attend) == 0: continue
        processed[image_id] = {
            'object_predict': object_predict,  # [5, 9]
            'object_attend': object_attend,  # [4, 6]
            'attr_predict': attr_predict,  # [1, 4]
            'attr_attend': attr_attend,  # [1, 4]
            'attr_blank_fill': attr_blank_fill,  # [1, 23]
            'obj_blank_fill': obj_blank_fill,  # [1, 23]
            'caption_attend': caption_attend,  # [10, 17]
        }
    cPickle.dump(processed, open(
        os.path.join(config.dir_name, '{}_processed.pkl'.format(split)), 'wb'))

    image_split[split] = processed.keys()  # used_image_ids
    split_image_ids = image_split[split]
    split_image_id2idx = {id: i for i, id in enumerate(split_image_ids)}
    split_image_info = {
        'image_ids': split_image_ids,
        'image_id2idx': split_image_id2idx,
    }
    cPickle.dump(split_image_info, open(
        os.path.join(config.dir_name, '{}_image_info.pkl'.format(split)), 'wb'))
cPickle.dump(image_split, open(config.save_image_split, 'wb'))

print('done')
