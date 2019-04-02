import cPickle
import json

print('load anno_train')
anno_train = json.load(
    open('data/VQA_v2/annotations/v2_mscoco_train2014_annotations.json', 'r'))
print('load anno_val')
anno_val = json.load(
    open('data/VQA_v2/annotations/v2_mscoco_val2014_annotations.json', 'r'))

anno = anno_train['annotations'] + anno_val['annotations']

qid2anno = {a['question_id']: a for a in anno}

print('dump qid2anno_trainval')
cPickle.dump(qid2anno,
             open('data/VQA_v2/annotations/qid2anno_trainval2014.pkl', 'wb'))
print('done')
