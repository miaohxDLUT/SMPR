import mmcv
import numpy as np
import torch


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    """
    Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.

    Args:
        rois (Tensor): boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): encoded offsets with respect to each roi.
            Has shape (N, 4). Note N = num_anchors * W * H when rois is a grid
            of anchors. Offset encoding follows [1]_.
        means (list): denormalizing means for delta coordinates
        stds (list): denormalizing standard deviation for delta coordinates
        max_shape (tuple[int, int]): maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): maximum aspect ratio for boxes.

    Returns:
        Tensor: boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.2817, 0.2817, 4.7183, 4.7183],
                [0.0000, 0.6321, 7.3891, 0.3679],
                [5.8967, 2.9251, 5.5033, 3.2749]])
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(Tensor or ndarray): Shape (..., 4*k)
        img_shape(tuple): Image shape.

    Returns:
        Same type as `bboxes`: Flipped bboxes.
    """
    if isinstance(bboxes, torch.Tensor):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4] - 1
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4] - 1
        return flipped
    elif isinstance(bboxes, np.ndarray):
        return mmcv.bbox_flip(bboxes, img_shape)

def kpt_flip(kpts, img_shape):
    if isinstance(kpts, torch.Tensor):
        assert kpts.shape[-1] % 34 == 0
        kpt_flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        kpts = kpts.reshape(-1, 17, 2)
        kpts = kpts[:, kpt_flip_index]
        # flipped = kpts.clone()
        # import ipdb; ipdb.set_trace()
        # flipped[:, :, 0] = img_shape[1] - kpts[:, :, 0] - 1
        # flipped = flipped.reshape(-1, 34)
        kpts[:, :, 0] = img_shape[1] - kpts[:, :, 0] - 1
        flipped = kpts.reshape(-1, 34)
        return flipped


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * scale_factor
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape)
    return new_bboxes


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes / scale_factor
    return new_bboxes

def kpt_mapping_back(kpts, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_kpts = kpt_flip(kpts, img_shape) if flip else kpts
    new_kpts = new_kpts / scale_factor
    return new_kpts


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]

def kpt2result(bboxes, kpts, labels, num_classes):
    assert bboxes.shape[0] == kpts.shape[0]
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)], \
               [np.zeros((0, 35), dtype=np.float32) for i in range(num_classes - 1)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        kpts = kpts.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)], \
            [kpts[labels == i, :] for i in range(num_classes - 1)]

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], -1)

def offset2kpt(points, offset, max_shape=None):
    assert points.shape[0] == offset.shape[0]
    assert offset.shape[1] == 34

    x1 = points[:, 0] + offset[:, 0]
    y1 = points[:, 1] + offset[:, 1]
    x2 = points[:, 0] + offset[:, 2]
    y2 = points[:, 1] + offset[:, 3]
    x3 = points[:, 0] + offset[:, 4]
    y3 = points[:, 1] + offset[:, 5]
    x4 = points[:, 0] + offset[:, 6]
    y4 = points[:, 1] + offset[:, 7]
    x5 = points[:, 0] + offset[:, 8]
    y5 = points[:, 1] + offset[:, 9]
    x6 = points[:, 0] + offset[:, 10]
    y6 = points[:, 1] + offset[:, 11]
    x7 = points[:, 0] + offset[:, 12]
    y7 = points[:, 1] + offset[:, 13]
    x8 = points[:, 0] + offset[:, 14]
    y8 = points[:, 1] + offset[:, 15]
    x9 = points[:, 0] + offset[:, 16]
    y9 = points[:, 1] + offset[:, 17]
    x10 = points[:, 0] + offset[:, 18]
    y10 = points[:, 1] + offset[:, 19]
    x11 = points[:, 0] + offset[:, 20]
    y11 = points[:, 1] + offset[:, 21]
    x12 = points[:, 0] + offset[:, 22]
    y12 = points[:, 1] + offset[:, 23]
    x13 = points[:, 0] + offset[:, 24]
    y13 = points[:, 1] + offset[:, 25]
    x14 = points[:, 0] + offset[:, 26]
    y14 = points[:, 1] + offset[:, 27]
    x15 = points[:, 0] + offset[:, 28]
    y15 = points[:, 1] + offset[:, 29]
    x16 = points[:, 0] + offset[:, 30]
    y16 = points[:, 1] + offset[:, 31]
    x17 = points[:, 0] + offset[:, 32]
    y17 = points[:, 1] + offset[:, 33]

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
        x3 = x3.clamp(min=0, max=max_shape[1] - 1)
        y3 = y3.clamp(min=0, max=max_shape[0] - 1)
        x4 = x4.clamp(min=0, max=max_shape[1] - 1)
        y4 = y4.clamp(min=0, max=max_shape[0] - 1)
        x5 = x5.clamp(min=0, max=max_shape[1] - 1)
        y5 = y5.clamp(min=0, max=max_shape[0] - 1)
        x6 = x6.clamp(min=0, max=max_shape[1] - 1)
        y6 = y6.clamp(min=0, max=max_shape[0] - 1)
        x7 = x7.clamp(min=0, max=max_shape[1] - 1)
        y7 = y7.clamp(min=0, max=max_shape[0] - 1)
        x8 = x8.clamp(min=0, max=max_shape[1] - 1)
        y8 = y8.clamp(min=0, max=max_shape[0] - 1)
        x9 = x9.clamp(min=0, max=max_shape[1] - 1)
        y9 = y9.clamp(min=0, max=max_shape[0] - 1)
        x10 = x10.clamp(min=0, max=max_shape[1] - 1)
        y10 = y10.clamp(min=0, max=max_shape[0] - 1)
        x11 = x11.clamp(min=0, max=max_shape[1] - 1)
        y11 = y11.clamp(min=0, max=max_shape[0] - 1)
        x12 = x12.clamp(min=0, max=max_shape[1] - 1)
        y12 = y12.clamp(min=0, max=max_shape[0] - 1)
        x13 = x13.clamp(min=0, max=max_shape[1] - 1)
        y13 = y13.clamp(min=0, max=max_shape[0] - 1)
        x14 = x14.clamp(min=0, max=max_shape[1] - 1)
        y14 = y14.clamp(min=0, max=max_shape[0] - 1)
        x15 = x15.clamp(min=0, max=max_shape[1] - 1)
        y15 = y15.clamp(min=0, max=max_shape[0] - 1)
        x16 = x16.clamp(min=0, max=max_shape[1] - 1)
        y16 = y16.clamp(min=0, max=max_shape[0] - 1)
        x17 = x17.clamp(min=0, max=max_shape[1] - 1)
        y17 = y17.clamp(min=0, max=max_shape[0] - 1)
    
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, \
        x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, x13, y13,\
        x14, y14, x15, y15, x16, y16, x17, y17], -1)

def offset2kpt_new(points, offset_init, offset, max_shape=None):
    assert points.shape[0] == offset.shape[0]
    assert offset.shape[1] == 34
    
    x1 = points[:, 0] + offset[:, 0] + offset_init[:, 0]
    y1 = points[:, 1] + offset[:, 1] + offset_init[:, 1]
    x2 = points[:, 0] + offset[:, 2] + offset_init[:, 2]
    y2 = points[:, 1] + offset[:, 3] + offset_init[:, 3]
    x3 = points[:, 0] + offset[:, 4] + offset_init[:, 4]
    y3 = points[:, 1] + offset[:, 5] + offset_init[:, 5]
    x4 = points[:, 0] + offset[:, 6] + offset_init[:, 6]
    y4 = points[:, 1] + offset[:, 7] + offset_init[:, 7]
    x5 = points[:, 0] + offset[:, 8] + offset_init[:, 8]
    y5 = points[:, 1] + offset[:, 9] + offset_init[:, 9]
    x6 = points[:, 0] + offset[:, 10] + offset_init[:, 10]
    y6 = points[:, 1] + offset[:, 11] + offset_init[:, 11]
    x7 = points[:, 0] + offset[:, 12] + offset_init[:, 12]
    y7 = points[:, 1] + offset[:, 13] + offset_init[:, 13]
    x8 = points[:, 0] + offset[:, 14] + offset_init[:, 14]
    y8 = points[:, 1] + offset[:, 15] + offset_init[:, 15]
    x9 = points[:, 0] + offset[:, 16] + offset_init[:, 16]
    y9 = points[:, 1] + offset[:, 17] + offset_init[:, 17]
    x10 = points[:, 0] + offset[:, 18] + offset_init[:, 18]
    y10 = points[:, 1] + offset[:, 19] + offset_init[:, 19]
    x11 = points[:, 0] + offset[:, 20] + offset_init[:, 20]
    y11 = points[:, 1] + offset[:, 21] + offset_init[:, 21]
    x12 = points[:, 0] + offset[:, 22] + offset_init[:, 22]
    y12 = points[:, 1] + offset[:, 23] + offset_init[:, 23]
    x13 = points[:, 0] + offset[:, 24] + offset_init[:, 24]
    y13 = points[:, 1] + offset[:, 25] + offset_init[:, 25]
    x14 = points[:, 0] + offset[:, 26] + offset_init[:, 26]
    y14 = points[:, 1] + offset[:, 27] + offset_init[:, 27]
    x15 = points[:, 0] + offset[:, 28] + offset_init[:, 28]
    y15 = points[:, 1] + offset[:, 29] + offset_init[:, 29]
    x16 = points[:, 0] + offset[:, 30] + offset_init[:, 30]
    y16 = points[:, 1] + offset[:, 31] + offset_init[:, 31]
    x17 = points[:, 0] + offset[:, 32] + offset_init[:, 32]
    y17 = points[:, 1] + offset[:, 33] + offset_init[:, 33]

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
        x3 = x3.clamp(min=0, max=max_shape[1] - 1)
        y3 = y3.clamp(min=0, max=max_shape[0] - 1)
        x4 = x4.clamp(min=0, max=max_shape[1] - 1)
        y4 = y4.clamp(min=0, max=max_shape[0] - 1)
        x5 = x5.clamp(min=0, max=max_shape[1] - 1)
        y5 = y5.clamp(min=0, max=max_shape[0] - 1)
        x6 = x6.clamp(min=0, max=max_shape[1] - 1)
        y6 = y6.clamp(min=0, max=max_shape[0] - 1)
        x7 = x7.clamp(min=0, max=max_shape[1] - 1)
        y7 = y7.clamp(min=0, max=max_shape[0] - 1)
        x8 = x8.clamp(min=0, max=max_shape[1] - 1)
        y8 = y8.clamp(min=0, max=max_shape[0] - 1)
        x9 = x9.clamp(min=0, max=max_shape[1] - 1)
        y9 = y9.clamp(min=0, max=max_shape[0] - 1)
        x10 = x10.clamp(min=0, max=max_shape[1] - 1)
        y10 = y10.clamp(min=0, max=max_shape[0] - 1)
        x11 = x11.clamp(min=0, max=max_shape[1] - 1)
        y11 = y11.clamp(min=0, max=max_shape[0] - 1)
        x12 = x12.clamp(min=0, max=max_shape[1] - 1)
        y12 = y12.clamp(min=0, max=max_shape[0] - 1)
        x13 = x13.clamp(min=0, max=max_shape[1] - 1)
        y13 = y13.clamp(min=0, max=max_shape[0] - 1)
        x14 = x14.clamp(min=0, max=max_shape[1] - 1)
        y14 = y14.clamp(min=0, max=max_shape[0] - 1)
        x15 = x15.clamp(min=0, max=max_shape[1] - 1)
        y15 = y15.clamp(min=0, max=max_shape[0] - 1)
        x16 = x16.clamp(min=0, max=max_shape[1] - 1)
        y16 = y16.clamp(min=0, max=max_shape[0] - 1)
        x17 = x17.clamp(min=0, max=max_shape[1] - 1)
        y17 = y17.clamp(min=0, max=max_shape[0] - 1)
    
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, \
        x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, x13, y13,\
        x14, y14, x15, y15, x16, y16, x17, y17], -1)

def struct_offset2kpt(points, offset, max_shape=None):
    assert points.shape[0] == offset.shape[0]
    assert offset.shape[1] == 34

    x1 = points[:, 0] + offset[:, 0]
    y1 = points[:, 1] + offset[:, 1]
    x2 = x1 + offset[:, 2]
    y2 = y1 + offset[:, 3]
    x3 = x1 + offset[:, 4]
    y3 = y1 + offset[:, 5]
    x4 = x2 + offset[:, 6]
    y4 = y2 + offset[:, 7]
    x5 = x3 + offset[:, 8]
    y5 = y3 + offset[:, 9]
    x6 = points[:, 0] + offset[:, 10]
    y6 = points[:, 1] + offset[:, 11]
    x7 = points[:, 0] + offset[:, 12]
    y7 = points[:, 1] + offset[:, 13]
    x8 = x6 + offset[:, 14]
    y8 = y6 + offset[:, 15]
    x9 = x7 + offset[:, 16]
    y9 = y7 + offset[:, 17]
    x10 = x8 + offset[:, 18]
    y10 = y8 + offset[:, 19]
    x11 = x9 + offset[:, 20]
    y11 = y9 + offset[:, 21]
    x12 = points[:, 0] + offset[:, 22]
    y12 = points[:, 1] + offset[:, 23]
    x13 = points[:, 0] + offset[:, 24]
    y13 = points[:, 1] + offset[:, 25]
    x14 = x12 + offset[:, 26]
    y14 = y12 + offset[:, 27]
    x15 = x13 + offset[:, 28]
    y15 = y13 + offset[:, 29]
    x16 = x14 + offset[:, 30]
    y16 = y14 + offset[:, 31]
    x17 = x15 + offset[:, 32]
    y17 = y15 + offset[:, 33]

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
        x3 = x3.clamp(min=0, max=max_shape[1] - 1)
        y3 = y3.clamp(min=0, max=max_shape[0] - 1)
        x4 = x4.clamp(min=0, max=max_shape[1] - 1)
        y4 = y4.clamp(min=0, max=max_shape[0] - 1)
        x5 = x5.clamp(min=0, max=max_shape[1] - 1)
        y5 = y5.clamp(min=0, max=max_shape[0] - 1)
        x6 = x6.clamp(min=0, max=max_shape[1] - 1)
        y6 = y6.clamp(min=0, max=max_shape[0] - 1)
        x7 = x7.clamp(min=0, max=max_shape[1] - 1)
        y7 = y7.clamp(min=0, max=max_shape[0] - 1)
        x8 = x8.clamp(min=0, max=max_shape[1] - 1)
        y8 = y8.clamp(min=0, max=max_shape[0] - 1)
        x9 = x9.clamp(min=0, max=max_shape[1] - 1)
        y9 = y9.clamp(min=0, max=max_shape[0] - 1)
        x10 = x10.clamp(min=0, max=max_shape[1] - 1)
        y10 = y10.clamp(min=0, max=max_shape[0] - 1)
        x11 = x11.clamp(min=0, max=max_shape[1] - 1)
        y11 = y11.clamp(min=0, max=max_shape[0] - 1)
        x12 = x12.clamp(min=0, max=max_shape[1] - 1)
        y12 = y12.clamp(min=0, max=max_shape[0] - 1)
        x13 = x13.clamp(min=0, max=max_shape[1] - 1)
        y13 = y13.clamp(min=0, max=max_shape[0] - 1)
        x14 = x14.clamp(min=0, max=max_shape[1] - 1)
        y14 = y14.clamp(min=0, max=max_shape[0] - 1)
        x15 = x15.clamp(min=0, max=max_shape[1] - 1)
        y15 = y15.clamp(min=0, max=max_shape[0] - 1)
        x16 = x16.clamp(min=0, max=max_shape[1] - 1)
        y16 = y16.clamp(min=0, max=max_shape[0] - 1)
        x17 = x17.clamp(min=0, max=max_shape[1] - 1)
        y17 = y17.clamp(min=0, max=max_shape[0] - 1)
    
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, \
        x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, x13, y13,\
        x14, y14, x15, y15, x16, y16, x17, y17], -1)
