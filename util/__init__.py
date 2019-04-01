import numpy as np

from .util import log

def get_dummy_data():
    bn = 500
    bs = 36
    dim = 2048

    features = np.zeros([bn, bs, dim])
    spatials = np.zeros([bn, bs, 6])
    normal_boxes = np.zeros([bn, bs, 4])
    num_boxes = np.ones([bn])
    max_box_num = bs
    vfeat_dim = dim

    return features, spatials, normal_boxes, num_boxes, max_box_num, vfeat_dim

# self.features, self.spatials, self.normal_boxes, self.num_boxes, self.max_box_num, self.vfeat_dim = features, spatials, normal_boxes, num_boxes, max_box_num, vfeat_dim
