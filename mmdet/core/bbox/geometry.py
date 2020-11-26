import torch


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5238, 0.0500, 0.0041],
                [0.0323, 0.0452, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        # 左上角点
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        # 右下角点
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious

def kpt_oks(kpts1, kpts2, gt_masks_areas, mode='oks', is_aligned=False, eps=1e-6):
    """
    Args:
        kpts1 (Tensor): shape (m, 34) in <x1, y1, x2, y2, ...> format.
        kpts2 (Tensor): shape (n, 34) in <x1, y1, x2, y2, ...> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
    """
    # if kpts1.shape[0] != gt_masks_areas.shape[0]:
    #     print('kpts1.shape', kpts1.shape)
    #     print('kpts1', kpts1)
    #     print('gt_masks_areas.shape', gt_masks_areas.shape)
    #     print('gt_masks_areas', gt_masks_areas)

    kpt_oks_sigmas = torch.Tensor([.26, .25, .25, .35, .35, .79, .79, .72,
                                   .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    variances = (kpt_oks_sigmas * 2) ** 2
    rows = kpts1.size(0)
    cols = kpts2.size(0)
    if rows * cols == 0:
        return kpts1.new(rows, 1) if is_aligned else kpts1.new(rows, cols)

    kpts2 = kpts2.reshape(-1, 17, 2)
    oks_list = []

    variances = variances.to(kpts1.device)

    oks_all = torch.zeros(rows, cols, device=kpts1.device)

    for i in range(rows):
        # import ipdb; ipdb.set_trace()
        squared_distance = (kpts1[i, None, :, 0] - kpts2[:, :, 0]) ** 2 + \
            (kpts1[i, None, :, 1] - kpts2[:, :, 1]) ** 2 
        vis_flag = (kpts1[i, :, 2] > 0).int()
        vis_ind = vis_flag.nonzero()[:, 0]
        num_vis_kpt = vis_ind.shape[0]

        area = gt_masks_areas[i]
        # x = kpts1[i, :, 0][vis_ind]
        # y = kpts1[i, :, 1][vis_ind]
        # x1 = x.min()
        # y1 = y.min()
        # x2 = x.max()
        # y2 = y.max()
        # w = (x2 - x1)
        # h = (y2 - y1)
        # w = (x2 - x1).clamp(min=0)
        # h = (y2 - y1).clamp(min=0)
        # area = w * h
        # eps = area.new_tensor([eps])
        # area = torch.max(area, eps)

        # squared_distance0 = squared_distance / area / variances / 2
        squared_distance0 = squared_distance / (area * variances * 2)
        squared_distance0 = squared_distance0[:, vis_ind]
        squared_distance1 = torch.exp(-squared_distance0).sum(dim=1)
        oks = squared_distance1 / num_vis_kpt
        oks_all[i] = oks
    return oks_all


def kpt_oks_no_mask(kpts1, kpts2, mode='oks', is_aligned=False, eps=1e-6):
    """
    Args:
        kpts1 (Tensor): shape (m, 34) in <x1, y1, x2, y2, ...> format.
        kpts2 (Tensor): shape (n, 34) in <x1, y1, x2, y2, ...> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
    """

    kpt_oks_sigmas = torch.Tensor([.26, .25, .25, .35, .35, .79, .79, .72,
                                   .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    variances = (kpt_oks_sigmas * 2) ** 2
    rows = kpts1.size(0)
    cols = kpts2.size(0)
    if rows * cols == 0:
        return kpts1.new(rows, 1) if is_aligned else kpts1.new(rows, cols)

    kpts2 = kpts2.reshape(-1, 17, 2)
    oks_list = []

    variances = variances.to(kpts1.device)

    oks_all = torch.zeros(rows, cols, device=kpts1.device)

    for i in range(rows):
        # import ipdb; ipdb.set_trace()
        squared_distance = (kpts1[i, None, :, 0] - kpts2[:, :, 0]) ** 2 + \
            (kpts1[i, None, :, 1] - kpts2[:, :, 1]) ** 2 
        vis_flag = (kpts1[i, :, 2] > 0).int()
        vis_ind = vis_flag.nonzero()[:, 0]
        num_vis_kpt = vis_ind.shape[0]

        # area = gt_masks_areas[i]
        x = kpts1[i, :, 0][vis_ind]
        y = kpts1[i, :, 1][vis_ind]
        x1 = x.min()
        y1 = y.min()
        x2 = x.max()
        y2 = y.max()
        w = (x2 - x1)
        h = (y2 - y1)
        # w = (x2 - x1).clamp(min=0)
        # h = (y2 - y1).clamp(min=0)
        area = w * h
        # eps = area.new_tensor([eps])
        # area = torch.max(area, eps)
        
        squared_distance0 = squared_distance / (area * variances * 2)
        # squared_distance0 = squared_distance / area / variances / 2
        squared_distance0 = squared_distance0[:, vis_ind]
        squared_distance1 = torch.exp(-squared_distance0).sum(dim=1)
        oks = squared_distance1 / num_vis_kpt
        oks_all[i] = oks
    return oks_all


def _compute_oks(joints_gt, joints_pred, image_size=(160, 160)):
    """Compute a object keypoint similarity for one example.
    Args:
        joints_gt: a numpy array of shape (num_joints, 3).
        joints_pred: a numpy array of shape (num_joints, 3).
        image_size: a tuple, (height, width).
    Returns:
        oks: float.
    """

    num_joints = joints_gt.shape[0]

    x_gt = joints_gt[:, 0]
    y_gt = joints_gt[:, 1]
    # visibility of ground-truth joints
    v_gt = joints_gt[:, 2]

    x_pred = joints_pred[:, 0]
    y_pred = joints_pred[:, 1]

    area = image_size[0] * image_size[1]

    squared_distance = (x_gt - x_pred) ** 2 + (y_gt - y_pred) ** 2

    squared_distance /= (area * variances * 2)

    oks = 0
    count = 0

    for i in range(num_joints):
        if v_gt[i] > 0:
            oks += np.exp(-squared_distance[i], dtype=np.float32)
            count += 1

    if count == 0:
        return -1
    else:
        oks /= count
        return oks