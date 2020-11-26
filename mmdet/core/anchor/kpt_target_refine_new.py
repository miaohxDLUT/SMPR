import torch

from ..bbox import PseudoSampler, assign_and_sample, build_assigner
from ..utils import multi_apply


def kpt_target_refine_new(proposals_list,
                 gt_kpts_list,
                 gt_masks_areas_list,
                 img_metas,
                 cfg,
                 gt_labels_list=None,
                 label_channels=1):
    """
    Compute corresponding GT box and classification targets for proposals.

    Args:
        points_list (list[list]): Multi level points of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        cfg (dict): train sample configs.

    Returns:
        tuple
    """
    
    num_imgs = len(img_metas)
    assert len(proposals_list) == num_imgs

    # points number of multi levels
    # num_level_proposals = [10000, 2500, 625, 169, 49]
    num_level_proposals = [points.size(0) for points in proposals_list[0]]

    # concat all level points and flags to a single tensor
    for i in range(num_imgs):
        proposals_list[i] = torch.cat(proposals_list[i])

    # compute targets for each image
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]

    (all_labels, all_label_weights, all_bbox_gt, all_proposals,
     all_proposal_weights, pos_inds_list, neg_inds_list, all_max_overlaps) = multi_apply(
         kpt_target_single,
         proposals_list,
         gt_kpts_list,
         gt_masks_areas_list,
         gt_labels_list,
         img_metas,
         cfg=cfg,
         label_channels=label_channels)
    
    # no valid points
    if any([labels is None for labels in all_labels]):
        return None
    
    # sampled points of all images
    # num_total_pos = 7
    # pos_inds_list[0] = [    6, 10133]
    # pos_inds_list[1] = [ 8841,  8941, 13060, 13260, 13317]
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    
    # len(all_labels) = 2
    # all_labels[0].shape = [13343]
    # all_labels[0] = [0, 0, 0,  ..., 0, 0, 0]
    # (all_labels[0] > 0).nonzero().shape = [2, 1]

    # len(labels_list) = 5
    # len(labels_list[0]) = 2
    # labels_list[0][0].shape = 10000
    # labels_list[4][0]
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels_list = images_to_levels(all_labels, num_level_proposals)


    # label_weights_list[4][0] = 
    #       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.,
    #       0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    #       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    label_weights_list = images_to_levels(all_label_weights,
                                          num_level_proposals)

    # bbox_gt_list[4][0].shape = [49, 34]
    # len(all_bbox_gt) = 2
    # all_bbox_gt[0].shape = [13343, 34]
    kpt_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)

    # proposals_list[4][0].shape = [49, 34]
    proposals_list = images_to_levels(all_proposals, num_level_proposals)

    # proposal_weights_list[4][0].shape = [49, 34]
    proposal_weights_list = images_to_levels(all_proposal_weights,
                                             num_level_proposals)

    # return (labels_list, label_weights_list, kpt_gt_list, proposals_list,
    #         proposal_weights_list, num_total_pos, num_total_neg)

    max_overlaps_list = images_to_levels(all_max_overlaps, num_level_proposals)


    # len(labels_list) = 5
    # len(labels_list[0]) = 80000 (8张图像，8倍下采样下，每张图像上10000个采样点)
    return labels_list, label_weights_list, kpt_gt_list, proposals_list, num_total_pos, max_overlaps_list
    # return labels_list, kpt_gt_list, proposals_list, num_total_pos, max_overlaps_list

def images_to_levels(target, num_level_grids):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    # len(target) = 2
    # target[0].shape = [13343]

    # target.shape = [2, 13343]
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_grids:
        end = start + n
        target_list = []
        for i in range(target.shape[0]):
            target_list.append(target[i, start:end])
        level_targets.append(torch.cat(target_list, 0))
        start = end
    return level_targets

def kpt_target_single(proposals,
                      gt_kpts,
                      gt_masks_areas,
                      gt_labels,
                      img_meta,
                      cfg,
                      label_channels=1):
    # assign gt and sample proposals
    # import ipdb; ipdb.set_trace()
    kpt_assigner = build_assigner(cfg.assigner)
    assign_result = kpt_assigner.assign(proposals, gt_kpts, gt_masks_areas,
                                        gt_labels, img_meta)
    bbox_sampler = PseudoSampler()
    sampling_result = bbox_sampler.sample(assign_result, proposals, gt_kpts)
    max_overlaps = sampling_result.max_overlaps
    
    # num_valid_proposals = 13343
    num_valid_proposals = proposals.shape[0]
    kpt_gt = proposals.new_zeros([num_valid_proposals, 34])
    
    pos_proposals = torch.zeros_like(proposals)
    proposals_weights = proposals.new_zeros([num_valid_proposals, 34])
    
    labels = proposals.new_zeros(num_valid_proposals, dtype=torch.long)
    label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

    # pos_inds.shape = [88]
    # pos_inds = [ 4823,  4921,  4922,  4923,  4949,  5019,  5020,  5021,  5022,  5023,
    #              ......
    #              12805, 12807, 12830, 12831, 12832, 12833, 12856, 13168]
    pos_inds = sampling_result.pos_inds
    # neg_inds.shape = [10281]
    neg_inds = sampling_result.neg_inds

    if len(pos_inds) > 0:
        # 对gt进行采样，只选proposals中大于positive阈值的gt
        # pos_gt_bboxes.shape = [88, 34]
        pos_gt_bboxes = sampling_result.pos_gt_bboxes
        kpt_gt[pos_inds, :] = pos_gt_bboxes
        pos_proposals[pos_inds, :] = proposals[pos_inds, :]
        proposals_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            # gt_labels = [1]
            # sampling_result.pos_assigned_gt_inds.shape = 88
            # sampling_result.pos_assigned_gt_inds = [0, 0, ..., 0]
            # gt_labels[sampling_result.pos_assigned_gt_inds] = [1, 1, ..., 1]
            # labels.shape = [13343], 其中有88个值是1，其余全为0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of proposals
    # if unmap_outputs:
    #     num_total_proposals = flat_proposals.size(0)
    #     labels = unmap(labels, num_total_proposals, inside_flags)
    #     label_weights = unmap(label_weights, num_total_proposals, inside_flags)
    #     bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
    #     pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
    #     proposals_weights = unmap(proposals_weights, num_total_proposals,
    #                               inside_flags)
    return (labels, label_weights, kpt_gt, pos_proposals, proposals_weights,
            pos_inds, neg_inds, max_overlaps)


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
