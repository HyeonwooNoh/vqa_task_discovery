"""
Box utils: utility functions for box manipulation

The codes are adapted from jcjohnson/densecap/blob/master/densecap/box_utils.lua
"""
import numpy as np


def xcycwh_to_x1y1x2y2(boxes):
    """
    Convert boxes from (xc, yc, w, h) format to (x1, y1, x2, y2) format.

    Input:
        - boxes: Numpy tensor of shape (B, N, 4) or (N, 4) giving boxes
          in (xc, yc, w, h) format.

    Returns:
        - Numpy tensor of shape (B, N, 4) or (N, 4) giving boxes in (x1, y2, x2, y2)
          format; output shape will match input shape.
    """
    minibatch = True
    if boxes.ndim == 2:
        minibatch = False
        boxes = np.expand_dims(boxes, axis=0)

    xc = boxes[:, :, 0]
    yc = boxes[:, :, 1]
    w = boxes[:, :, 2]
    h = boxes[:, :, 3]

    x0 = xc - w / 2.
    x1 = xc + w / 2.
    y0 = yc - h / 2.
    y1 = yc + h / 2.

    ret = np.stack([x0, y0, x1, y1], axis=2)
    if not minibatch: ret = ret.squeeze(axis=0)
    return ret


def xywh_to_x1y1x2y2(boxes):
    """
    Convert boxes from (x, y, w, h) format to (x1, y2, x2, y2) format.

    Input:
        - boxes: Tensor of shape (B, N, 4) or (N, 4) giving boxes
        in (x, y, w, h) format.

    Returns:
        - Tensor of shape (B, N, 4) or (N, 4) giving boxes in (x1, y2, x2, y2)
          format; output shape will match input shape.
    """
    minibatch = True
    if boxes.ndim == 2:
        minibatch = False
        boxes = np.expand_dims(boxes, axis=0)

    x = boxes[:, :, 0]
    y = boxes[:, :, 1]
    w = boxes[:, :, 2]
    h = boxes[:, :, 3]

    x0 = x.copy()
    y0 = y.copy()
    x1 = x0 + w
    y1 = y0 + h

    ret = np.stack([x0, y0, x1, y1], axis=2)
    if not minibatch: ret = ret.squeeze(axis=0)
    return ret


def x1y1x2y2_to_xywh(boxes):
    """
    Convert boxes from (x1, y1, x2, y2) format to (x, y, w, y) format.

    Input:
        - boxes: Tensor of shape (B, N, 4) or (N, 4) giving boxes in
          (x1, y1, x2, y2) format.

    Returns:
        - Tensor of same shape as input giving boxes in (x, y, w, h) format.
    """
    minibatch = True
    if boxes.ndim == 2:
        minibatch = False
        boxes = np.expand_dims(boxes, axis=0)

    x0 = boxes[:, :, 0]
    y0 = boxes[:, :, 1]
    x1 = boxes[:, :, 2]
    y1 = boxes[:, :, 3]

    x = x0.copy()
    y = y0.copy()
    w = x1 - x0
    h = y1 - y0

    ret = np.stack([x, y, w, h], axis=2)
    if not minibatch: ret = ret.squeeze(axis=0)
    return ret


def x1y1x2y2_to_xcycwh(boxes):
    minibatch = True
    if boxes.ndim == 2:
        minibatch = False
        boxes = np.expand_dims(boxes, axis=0)

    x0 = boxes[:, :, 0]
    y0 = boxes[:, :, 1]
    x1 = boxes[:, :, 2]
    y1 = boxes[:, :, 3]

    xc = (x0 + x1) / 2.0
    yc = (y0 + y1) / 2.0
    w = x1 - x0
    h = y1 - y0

    ret = np.stack([xc, yc, w, h], axis=2)
    if not minibatch: ret = ret.squeeze(axis=0)
    return ret


def xcycwh_to_xywh(boxes):
    boxes_x1y1x2y2 = xcycwh_to_x1y1x2y2(boxes)
    boxes_xywh = x1y1x2y2_to_xywh(boxes_x1y1x2y2)
    return boxes_xywh


def normalize_box_x1y1x2y2(box, width, height):
    box = box.copy().astype(np.float32)
    box[0] /= float(width)
    box[1] /= float(height)
    box[2] /= float(width)
    box[3] /= float(height)
    return np.clip(box, 0, 1)


def normalize_boxes_x1y1x2y2(boxes, width, height):
    boxes = boxes.astype(np.float32)
    x1 = boxes[:, 0] / width
    y1 = boxes[:, 1] / height
    x2 = boxes[:, 2] / width
    y2 = boxes[:, 3] / height
    new_boxes = np.stack([y1, x1, y2, x2], axis=1)
    return np.clip(new_boxes, 0, 1)


def scale_box_x1y1x2y2(box, frac):
    """
    Rescale boxes to convert from one coordinate system to another.

    Inputs:
        - boxes: Tensor of shape (4) giving coordinates of a box in
          (x1, y1, x2, y2) format.
        - frac: Fraction by which to scale the boxes. For example
          if boxes assume that the input image has size 800x600 but we want to
          use them at 400x300 scale, then frac should be 0.5.
          array [frac_x, frac_y] for using separate rescale

    Returns:
        - boxes_scaled: Tensor of shape (4) giving rescaled box coordinates
          in (x1, y1, x2, y2) format.
    """
    # bb is given as Nx4 tensor of x,y,w,h
    # e.g. original width was 800 but now is 512, then frac will be 800/512 = 1.56
    if isinstance(frac, list):
        assert len(frac) == 2, 'only two dimension frac is possible for array input'
        new_box = box.copy()
        new_box[0] *= frac[0]
        new_box[1] *= frac[1]
        new_box[2] *= frac[0]
        new_box[3] *= frac[1]
    else:
        new_box = box * float(frac)
    return new_box


def scale_boxes_x1y1x2y2(boxes, frac):
    """
    Rescale boxes to convert from one coordinate system to another.

    Inputs:
        - boxes: Tensor of shape (N, 4) giving coordinates of boxes in
          (x1, y1, x2, y2) format.
        - frac: Fraction by which to scale the boxes. For example
          if boxes assume that the input image has size 800x600 but we want to
          use them at 400x300 scale, then frac should be 0.5.
          array [frac_x, frac_y] for using separate rescale

    Returns:
        - boxes_scaled: Tensor of shape (N, 4) giving rescaled box coordinates
          in (x1, y1, x2, y2) format.
    """
    # bb is given as Nx4 tensor of x,y,w,h
    # e.g. original width was 800 but now is 512, then frac will be 800/512 = 1.56
    if isinstance(frac, list):
        assert len(frac) == 2, 'only two dimension frac is possible for array input'
        new_boxes = boxes.copy()
        new_boxes[:, 0] *= frac[0]
        new_boxes[:, 1] *= frac[1]
        new_boxes[:, 2] *= frac[0]
        new_boxes[:, 3] *= frac[1]
    else:
        new_boxes = boxes * float(frac)
    return new_boxes


def scale_boxes_xywh(boxes, frac):
    """
    Rescale boxes to convert from one coordinate system to another.

    Inputs:
        - boxes: Tensor of shape (N, 4) giving coordinates of boxes in
          (x, y, w, h) format.
        - frac: Fraction by which to scale the boxes. For example
          if boxes assume that the input image has size 800x600 but we want to
          use them at 400x300 scale, then frac should be 0.5.
          array [frac_x, frac_y] for using separate rescale

    Returns:
        - boxes_scaled: Tensor of shape (N, 4) giving rescaled box coordinates
          in (x, y, w, h) format.
    """
    # bb is given as Nx4 tensor of x,y,w,h
    # e.g. original width was 800 but now is 512, then frac will be 800/512 = 1.56
    if isinstance(frac, list):
        assert len(frac) == 2, 'only two dimension frac is possible for array input'
        new_boxes = boxes.copy()
        new_boxes[:, 0] *= frac[0]
        new_boxes[:, 1] *= frac[1]
        new_boxes[:, 2] *= frac[0]
        new_boxes[:, 3] *= frac[1]
    else:
        new_boxes = boxes * float(frac)
    return new_boxes


def clip_boxes(boxes, bounds, format='x1y1x2y2'):
    """
    Clip bounding boxes to a specified region.

    Inputs:
        - boxes: Numpy tensor containing boxes, of shape (N, 4) or (N, M, 4)
        - bounds: List containing the following keys specifying the bounds:
            [x_min, y_min, x_max, , y_max]
            - x_min, x_max: Minimum and maximum values for x (inclusive)
            - y_min, y_max: Minimum and maximum values for y (inclusive)
        - format: The format of the boxes; either 'x1y1x2y2' or 'xcycwh'.

    Outputs:
        - boxes_clipped: Tensor giving coordinates of clipped boxes; has
          same shape and format as input.
        - valid: 1D byte Tensor indicating which bounding boxes are valid,
          in sense of completely out of bounds of the image.
    """
    if format == 'x1y1x2y2': boxes_clipped = boxes.copy()
    elif format == 'xcycwh': boxes_clipped = xcycwh_to_x1y1x2y2(boxes)
    elif format == 'xywh': boxes_clipped = xywh_to_x1y1x2y2(boxes)
    else: raise ValueError('Unknown box format: {}'.format(format))

    if boxes_clipped.ndim == 3:
        boxes_clipped = boxes_clipped.reshape([-1, 4])

    x_min, y_min = bounds[0], bounds[1]
    x_max, y_max = bounds[2], bounds[3]
    assert x_min < x_max, 'x_min >= x_max'
    assert y_min < y_max, 'y_min >= y_max'

    boxes_clipped[:, 0] = boxes.clipped[:, 0].clip(x_min, x_max - 1)
    boxes_clipped[:, 1] = boxes.clipped[:, 1].clip(y_min, y_max - 1)
    boxes_clipped[:, 2] = boxes.clipped[:, 2].clip(x_min + 1, x_max)
    boxes_clipped[:, 3] = boxes.clipped[:, 3].clip(y_min + 1, y_max)

    validx = boxes_clipped[:, 0] < boxes.clipped[:, 2]
    validy = boxes_clipped[:, 1] < boxes_clipped[:, 3]
    valid = np.logical_and(validx, validy)

    if format == 'xcycwh': boxes_clipped = x1y1x2y2_to_xcycwh(boxes_clipped)
    elif format == 'xywh': boxes_clipped = x1y1x2y2_to_xywh(boxes_clipped)

    boxes_clipped = boxes_clipped.reshape(boxes.shape)
    valid = valid.reshape(boxes.shape[:-1])

    # Conver to the same shape as the input
    return boxes_clipped, valid


def iou(box1, box2):
    # box format: [x0, y0, x1, y1]
    inter = np.concatenate(
        [np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])],
        axis=0)
    iw = inter[2] - inter[0]
    ih = inter[3] - inter[1]
    if iw <= 0 or ih <= 0: return 0.0
    union = np.concatenate(
        [np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])],
        axis=0)
    uw = union[2] - union[0]
    uh = union[3] - union[1]
    iou = iw * ih / (uw * uh)
    return iou


def iou_matrix_by_iter(boxes1, boxes2):
    """
    Compute pairwise NxN IOU matrix in Nx4 array of boxes in x1y1x2y2 format
    """
    n1 = boxes1.shape[0]
    n2 = boxes2.shape[0]

    M = np.zeros([n1, n2], dtype=np.float32)
    for i1 in range(n1):
        for i2 in range(n2):
            M[i1, i2] = iou(boxes1[i1], boxes2[i2])
    return M


def is_inside_matrix(boxes1, boxes2):
    """
    Compute whether percentage of boxes1 that is inside of boxes2.
    return is NXM [0, 1] matrix where boxes1: Nx4, boxes2: Mx4 in x1y1x2y2 format
    """
    # Make two NxMx4 matrices: box1 - row major, box2 - column major
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    tile_boxes1 = np.tile(np.expand_dims(boxes1, axis=1), [1, m, 1])
    tile_boxes2 = np.tile(np.expand_dims(boxes2, axis=0), [n, 1, 1])

    inter = np.concatenate([
        np.maximum(tile_boxes1[:, :, :2], tile_boxes2[:, :, :2]),
        np.minimum(tile_boxes1[:, :, 2:], tile_boxes2[:, :, 2:])], axis=2)
    iw = inter[:, :, 2] - inter[:, :, 0]
    ih = inter[:, :, 3] - inter[:, :, 1]
    iw = iw.clip(min=0)
    ih = ih.clip(min=0)
    area_i = iw * ih

    box1_w = tile_boxes1[:, :, 2] - tile_boxes1[:, :, 0]
    box1_h = tile_boxes1[:, :, 3] - tile_boxes1[:, :, 1]
    area_box1 = box1_w * box1_h
    return area_i.astype(np.float32) / area_box1.astype(np.float32)


def iou_matrix(boxes1, boxes2):
    """
    Compute pairwise NxM IOU matrix in Nx4, Mx4 array of boxes in x1y1x2y2 format
    """
    # Make two NxMx4 matrices: box1 - row major, box2 - column major
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    tile_boxes1 = np.tile(np.expand_dims(boxes1, axis=1), [1, m, 1])
    tile_boxes2 = np.tile(np.expand_dims(boxes2, axis=0), [n, 1, 1])

    inter = np.concatenate([
        np.maximum(tile_boxes1[:, :, :2], tile_boxes2[:, :, :2]),
        np.minimum(tile_boxes1[:, :, 2:], tile_boxes2[:, :, 2:])], axis=2)
    iw = inter[:, :, 2] - inter[:, :, 0]
    ih = inter[:, :, 3] - inter[:, :, 1]
    iw = iw.clip(min=0)
    ih = ih.clip(min=0)
    area_i = iw * ih

    box1_w = tile_boxes1[:, :, 2] - tile_boxes1[:, :, 0]
    box1_h = tile_boxes1[:, :, 3] - tile_boxes1[:, :, 1]
    area_box1 = box1_w * box1_h

    box2_w = tile_boxes2[:, :, 2] - tile_boxes2[:, :, 0]
    box2_h = tile_boxes2[:, :, 3] - tile_boxes2[:, :, 1]
    area_box2 = box2_w * box2_h

    area_u = (area_box1 + area_box2) - area_i
    return area_i.astype(np.float32) / area_u.astype(np.float32)


def iou_matrix_xywh(boxes1, boxes2):
    return iou_matrix(xywh_to_x1y1x2y2(boxes1), xywh_to_x1y1x2y2(boxes2))


def draw_x1y1x2y2(image, box, value):
    assert len(image.shape) == 3, 'image should have height, width, and channel'
    assert image.shape[2] == 3, 'image should have 3 channels'

    h, w, c = image.shape
    (x1, y1, x2, y2) = box
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(w - 1, x2))
    y2 = int(min(h - 1, y2))

    new_image = image.copy()
    new_image[y1: y2, x1: x2, :] = value
    return new_image


def add_value_x1y1x2y2(image, box, value):
    assert len(image.shape) == 3, 'image should have height, width, and channel'
    assert image.shape[2] == 3, 'image should have 3 channels'

    h, w, c = image.shape
    (x1, y1, x2, y2) = box
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(w - 1, x2))
    y2 = int(min(h - 1, y2))

    new_image = image.copy()
    new_image[y1: y2, x1: x2, :] += value
    return new_image
