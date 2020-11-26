import torch

from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


class PseudoSampler(BaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, kpts, gt_kpts, **kwargs):
        # positive的oks对应的index
        # torch.unique()表示gt的index只出现一次
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        # negative的oks对应的index
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = kpts.new_zeros(kpts.shape[0], dtype=torch.uint8)
        # gt_kpts = gt_kpts.reshape(-1, 51)
        # kpt_index = []
        # for i in range(17):
        #     kpt_index.append(3 * i)
        #     kpt_index.append(3 * i + 1)
        # gt_kpts = gt_kpts[:, kpt_index]
        gt_kpts = gt_kpts.reshape(-1, 17, 3)
        gt_kpts = gt_kpts[:, :, :2].reshape(-1, 34)
        
        # max_overlaps.shape = [13343]
        max_overlaps = assign_result.max_overlaps
        # kpts.shape = [13343, 34]
        sampling_result = SamplingResult(pos_inds, neg_inds, kpts, gt_kpts,
                                         assign_result, gt_flags, max_overlaps)
        return sampling_result
