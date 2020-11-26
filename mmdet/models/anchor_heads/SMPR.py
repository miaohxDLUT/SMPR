import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import normal_init

from mmdet.core import kpt_target_refine
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms, offset2kpt, multiclass_nms_kpt
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
from mmdet.ops import DeformConv

INF = 1e8


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            34, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption_reg = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.conv_adaption_cls = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption_reg, std=0.01)
        normal_init(self.conv_adaption_cls, std=0.01)

    def forward(self, reg_feat, cls_feat, pred_init):
        offset = self.conv_offset(pred_init.detach())
        reg_feat_refine = self.relu(self.conv_adaption_reg(reg_feat, offset))
        cls_feat_refine = self.relu(self.conv_adaption_cls(cls_feat, offset))
        return reg_feat_refine, cls_feat_refine


@HEADS.register_module
class SMPR(nn.Module):

    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                center_sampling=True,
                center_sample_radius=1.5,
                loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                loss_heatmap=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=4.0),
                loss_kpt_init=dict(type='KptL1Loss', loss_weight=0.05),
                loss_kpt_refine=dict(type='KptL1Loss', loss_weight=0.1),
                loss_rescore=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(SMPR, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_kpt_init = build_loss(loss_kpt_init)
        self.loss_kpt_refine = build_loss(loss_kpt_refine)
        self.loss_rescore = build_loss(loss_rescore)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.hm_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        for i in range(2):
            chn = self.in_channels if i == 0 else 128
            self.hm_convs.append(
                ConvModule(
                    chn,
                    128,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            
        self.hm_out = nn.Conv2d(128, 17, 3, padding=1)
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 34, 3, padding=1)
        self.fcos_rescore = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales_1 = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.scales_2 = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.feature_adaption = FeatureAdaption(
            256,
            256,
            kernel_size=3,
            deformable_groups=4)
        self.fcos_refine_out = nn.Conv2d(self.feat_channels, 34, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_refine_out, std=0.01)
        normal_init(self.fcos_rescore, std=0.01)
        self.feature_adaption.init_weights()
        for m in self.hm_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.hm_out, std=0.01, bias=bias_cls)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales_1, self.scales_2, self.fpn_strides)

    def forward_single(self, x, scale_1, scale_2, fpn_stride):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        # cls_score = self.fcos_cls(cls_feat)
        # centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        kpt_pred_init = scale_1(self.fcos_reg(reg_feat))
        kpt_refine_feat, cls_refine_feat = self.feature_adaption(reg_feat, cls_feat, kpt_pred_init)
        cls_score = self.fcos_cls(cls_refine_feat)
        # cls_score = self.fcos_cls(cls_feat)
        rescore = self.fcos_rescore(kpt_refine_feat)

        kpt_pred_refine = scale_2(self.fcos_refine_out(kpt_refine_feat))
        kpt_pred_refine = kpt_pred_init.detach() + kpt_pred_refine

        # hm_pred only stride == 8
        if fpn_stride == 8:
            hm_feat = x
            for hm_layer in self.hm_convs:
                hm_feat = hm_layer(hm_feat)
            hm_pred = self.hm_out(hm_feat)
        else:
            hm_pred = None

        if not self.training:
            kpt_pred_refine  = kpt_pred_refine * fpn_stride
            kpt_pred_init = kpt_pred_init * fpn_stride
            return cls_score, kpt_pred_init, kpt_pred_refine, rescore

        return cls_score, kpt_pred_init, kpt_pred_refine, hm_pred, rescore

    @force_fp32(apply_to=('cls_scores', 'kpt_preds_init', 'kpt_preds_refine', 'rescores'))
    def loss(self,
             cls_scores,
             kpt_preds_init,
             kpt_preds_refine,
             hm_preds,
             rescores,
             gt_bboxes,
             gt_kpts,
             gt_labels,
             gt_masks_areas,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(kpt_preds_init) == len(kpt_preds_refine) == len(rescores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        high, width = hm_preds[0].shape[2:]
        all_level_points = self.get_points(featmap_sizes, kpt_preds_init[0].dtype,
                                           kpt_preds_init[0].device)
        labels, bbox_targets, kpt_targets, \
            kpt_vis_flag, hm_targets = self.fcos_target(all_level_points, 
                                            gt_bboxes, gt_kpts,
                                            gt_labels, high, width)
        
        # generate kpt proposals
        num_imgs = cls_scores[0].size(0)
        kpt_list = []
        for i in range(num_imgs):
            kpt = []
            for lvl in range(len(kpt_preds_refine)):
                kpt_pred_offset_init = kpt_preds_init[lvl][i].detach()
                kpt_pred_offset_init = kpt_pred_offset_init.permute(1, 2, 0).reshape(-1, 34)
                kpt_pred_offset_init = kpt_pred_offset_init * self.fpn_strides[lvl]
                kpt.append(offset2kpt(all_level_points[lvl], kpt_pred_offset_init))
            kpt_list.append(kpt)
        
        # generate kpt refine targets
        labels_refine, kpt_gt_refine, proposals_list, \
                num_total_pos, max_overlaps_list = kpt_target_refine(kpt_list, \
                                                  gt_kpts, gt_masks_areas, img_metas, cfg, gt_labels)

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_hm_preds = hm_preds[0].permute(0, 2, 3, 1).reshape(-1, 17)

        flatten_kpt_preds_init = [
            kpt_pred_init.permute(0, 2, 3, 1).reshape(-1, 34)
            for kpt_pred_init in kpt_preds_init
        ]
        flatten_kpt_preds_refine = [
            kpt_pred_refine.permute(0, 2, 3, 1).reshape(-1, 34)
            for kpt_pred_refine in kpt_preds_refine
        ]
        flatten_rescores = [
            rescore.permute(0, 2, 3, 1).reshape(-1)
            for rescore in rescores
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_kpt_preds_init = torch.cat(flatten_kpt_preds_init)
        flatten_kpt_preds_refine = torch.cat(flatten_kpt_preds_refine)
        flatten_rescores = torch.cat(flatten_rescores)
        
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_kpt_targets = torch.cat(kpt_targets)
        flatten_kpt_vis_flag = torch.cat(kpt_vis_flag)
        flatten_hm_targets = torch.cat(hm_targets)

        flatten_labels_refine = torch.cat(labels_refine)
        flatten_kpt_gt_refine = torch.cat(kpt_gt_refine)
        flatten_max_overlaps = torch.cat(max_overlaps_list)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        flatten_labels_refine = flatten_labels * flatten_labels_refine
        pos_inds = flatten_labels_refine.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        pos_inds_init = flatten_labels.nonzero().reshape(-1)
        num_pos_init = len(pos_inds_init)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels_refine,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        # print('loss_cls', loss_cls)

        num_kpts = len(flatten_hm_targets.nonzero().reshape(-1))
        loss_heatmap = self.loss_heatmap(
            flatten_hm_preds,
            flatten_hm_targets,
            avg_factor=num_kpts + num_imgs)
        # print('loss_heatmap', loss_heatmap)
        
        pos_kpt_preds_init = flatten_kpt_preds_init[pos_inds_init]
        pos_kpt_preds_refine = flatten_kpt_preds_refine[pos_inds]
        pos_rescores = flatten_rescores[pos_inds]

        if num_pos_init > 0:
            # print('num_pos_init', num_pos_init)
            
            pos_max_overlaps_init = flatten_max_overlaps[pos_inds_init]

            pos_points_init = flatten_points[pos_inds_init]
            
            pos_kpt_targets_init = flatten_kpt_targets[pos_inds_init]
            pos_kpt_vis_flag_init = flatten_kpt_vis_flag[pos_inds_init]

            pos_decoded_kpt_preds_init = offset2kpt(pos_points_init, pos_kpt_preds_init)
            pos_decoded_kpt_targets_init = offset2kpt(pos_points_init, pos_kpt_targets_init)

            loss_kpt_init = self.loss_kpt_init(
                pos_decoded_kpt_preds_init,
                pos_decoded_kpt_targets_init,
                pos_kpt_vis_flag_init,
                weight=pos_max_overlaps_init,
                avg_factor=pos_max_overlaps_init.sum()
            )
            # print('loss_kpt_init', loss_kpt_init)

            if num_pos > 0:
                # print('num_pos', num_pos)
                pos_max_overlaps = flatten_max_overlaps[pos_inds]

                pos_points = flatten_points[pos_inds]
                pos_kpt_targets = flatten_kpt_targets[pos_inds]
                pos_kpt_vis_flag = flatten_kpt_vis_flag[pos_inds]
                pos_decoded_kpt_preds_refine = offset2kpt(pos_points, pos_kpt_preds_refine)
                pos_decoded_kpt_preds_refine = offset2kpt(pos_points, pos_kpt_preds_refine)
                pos_decoded_kpt_targets = offset2kpt(pos_points, pos_kpt_targets)

                loss_kpt_refine = self.loss_kpt_refine(
                    pos_decoded_kpt_preds_refine,
                    pos_decoded_kpt_targets,
                    pos_kpt_vis_flag, 
                    weight=pos_max_overlaps,
                    avg_factor=pos_max_overlaps.sum()
                )
                # print('loss_kpt_refine', loss_kpt_refine)

                loss_rescore = self.loss_rescore(pos_rescores, pos_max_overlaps)
                # print('**************************')
            else:
                loss_kpt_refine = pos_kpt_preds_refine.sum()

                pos_max_overlaps_init = flatten_max_overlaps[pos_inds_init]
                pos_rescores_init = flatten_rescores[pos_inds_init]
                loss_rescore = self.loss_rescore(pos_rescores_init, pos_max_overlaps_init)
                
                # print('--------------------------')
        else:
            loss_kpt_init = pos_kpt_preds_init.sum()
            loss_kpt_refine = pos_kpt_preds_refine.sum()
            loss_rescore = pos_rescores.sum()

        return dict(
            loss_cls=loss_cls,
            loss_heatmap=loss_heatmap,
            loss_kpt_init=loss_kpt_init,
            loss_kpt_refine=loss_kpt_refine,
            loss_rescore=loss_rescore)

    @force_fp32(apply_to=('cls_scores', 'kpt_preds', 'rescores'))
    def get_bboxes(self,
                   cls_scores,
                   kpt_preds_init,
                   kpt_preds_refine,
                   rescores,
                   img_metas,
                   cfg,
                   rescale=None,
                   nms=True):
        assert len(cls_scores) == len(kpt_preds_refine) == len(kpt_preds_init)

        num_levels = len(cls_scores)
        kpt_preds = kpt_preds_refine

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, kpt_preds[0].dtype,
                                      kpt_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            kpt_pred_list = [
                kpt_preds[i][img_id].detach() for i in range(num_levels)
            ]
            rescore_pred_list = [
                rescores[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, kpt_pred_list,
                                                rescore_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale, nms)
            result_list.append(det_bboxes)
            
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          kpt_preds,
                          rescores,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          nms=True):
        assert len(cls_scores) == len(kpt_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_kpts = []
        mlvl_rescore = []
        for cls_score, kpt_pred, rescore, points in zip(
                cls_scores, kpt_preds, rescores, mlvl_points):
            assert cls_score.size()[-2:] == kpt_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            rescore = rescore.permute(1, 2, 0).reshape(-1).sigmoid()
            kpt_pred = kpt_pred.permute(1, 2, 0).reshape(-1, 34)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * rescore[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                kpt_pred = kpt_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                rescore = rescore[topk_inds]
            kpts = offset2kpt(points, kpt_pred, max_shape=img_shape)
            bboxes = self.get_pesudo_bbox(kpts)
            mlvl_bboxes.append(bboxes)
            mlvl_kpts.append(kpts)
            mlvl_scores.append(scores)
            mlvl_rescore.append(rescore)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_kpts = torch.cat(mlvl_kpts)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_kpts /= mlvl_kpts.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)

        mlvl_rescore = torch.cat(mlvl_rescore)

        if nms:
            det_bboxes, det_kpts, det_labels = multiclass_nms_kpt(
                mlvl_bboxes,
                mlvl_kpts,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_rescore)
            return det_bboxes, det_kpts, det_labels
        else:
            return mlvl_bboxes, mlvl_kpts, mlvl_scores, mlvl_rescore

    def get_pesudo_bbox(self, kpts):
        assert kpts.shape[1] == 34
        kpts = kpts.reshape(-1, 17, 2)
        kpts_x = kpts[:, :, 0]
        kpts_y = kpts[:, :, 1]
        x1 = kpts_x.min(dim=1, keepdim=True)[0]
        y1 = kpts_y.min(dim=1, keepdim=True)[0]
        x2 = kpts_x.max(dim=1, keepdim=True)[0]
        y2 = kpts_y.max(dim=1, keepdim=True)[0]
        bboxes = torch.cat([x1, y1, x2, y2], dim=1)
        return bboxes

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_kpts_list, gt_labels_list, high, width):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        # the shape of hm_feat
        # get labels and bbox_targets of each image

        labels_list, bbox_targets_list, kpt_targets_list, \
            kpt_vis_flag_list, hm_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_kpts_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points,
            h=high,
            w=width)
        

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        kpt_targets_list =[
            kpt_targets.split(num_points, 0)
            for kpt_targets in kpt_targets_list
        ]
        kpt_vis_flag_list = [
            kpt_vis_flag.split(num_points, 0)
            for kpt_vis_flag in kpt_vis_flag_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_kpt_targets = []
        concat_lvl_kpt_vis_flag = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            # concat_lvl_bbox_targets.append(
            #     torch.cat(
            #         [bbox_targets[i] for bbox_targets in bbox_targets_list]))
            per_lvl_bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            concat_lvl_bbox_targets.append(
                per_lvl_bbox_targets / self.fpn_strides[i]
            )
            per_lvl_kpt_targets = torch.cat([kpt_targets[i] for kpt_targets in kpt_targets_list])
            concat_lvl_kpt_targets.append(
                per_lvl_kpt_targets / self.fpn_strides[i])
            concat_lvl_kpt_vis_flag.append(
                torch.cat(
                    [kpt_vis_flag[i] for kpt_vis_flag in kpt_vis_flag_list]))

        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_kpt_targets, concat_lvl_kpt_vis_flag, hm_targets_list

    def fcos_target_single(self, gt_bboxes, gt_kpts, gt_labels, points, regress_ranges,
                           num_points_per_lvl, h, w):
        assert gt_bboxes.shape[0] == gt_kpts.shape[0]
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_kpts.new_zeros((num_points, 34)), \
                   gt_kpts.new_zeros((num_points, 34)), \
                   gt_labels.new_zeros(h * w)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # generate hm target
        hm_targets = gt_labels.new_zeros((h, w))
        for i in range(num_gts):
            for j in range(17):
                if gt_kpts[i, j, 2] > 0:
                    x = torch.round((gt_kpts[i, j, 0] - 4) / 8).int()
                    y = torch.round((gt_kpts[i, j, 1] - 4) / 8).int()
                    hm_targets[y, x] = j + 1
        hm_targets = hm_targets.reshape(-1)

        # prepar kpt coordinate and vis flag
        new_kpts = gt_kpts.new_zeros(num_gts, 34)
        kpt_vis_flag = gt_kpts.new_zeros(num_gts, 34)
        for i in range(num_gts):
            for j in range(17):
                new_kpts[i, 2 * j] = gt_kpts[i, j, 0]
                new_kpts[i, 2 * j + 1] = gt_kpts[i, j, 1]
                kpt_vis_flag[i, 2 * j] = 1 if gt_kpts[i, j, 2] >= 1 else 0
                kpt_vis_flag[i, 2 * j + 1] = 1 if gt_kpts[i, j, 2] >= 1 else 0
    
        gt_kpts = new_kpts
        gt_kpts = gt_kpts[None].expand(num_points, num_gts, 34)
        kpt_vis_flag = kpt_vis_flag[None].expand(num_points, num_gts, 34)
        x1 = gt_kpts[..., 0] - xs
        y1 = gt_kpts[..., 1] - ys
        x2 = gt_kpts[..., 2] - xs
        y2 = gt_kpts[..., 3] - ys
        x3 = gt_kpts[..., 4] - xs
        y3 = gt_kpts[..., 5] - ys
        x4 = gt_kpts[..., 6] - xs
        y4 = gt_kpts[..., 7] - ys
        x5 = gt_kpts[..., 8] - xs
        y5 = gt_kpts[..., 9] - ys
        x6 = gt_kpts[..., 10] - xs
        y6 = gt_kpts[..., 11] - ys
        x7 = gt_kpts[..., 12] - xs
        y7 = gt_kpts[..., 13] - ys
        x8 = gt_kpts[..., 14] - xs
        y8 = gt_kpts[..., 15] - ys
        x9 = gt_kpts[..., 16] - xs
        y9 = gt_kpts[..., 17] - ys
        x10 = gt_kpts[..., 18] - xs
        y10 = gt_kpts[..., 19] - ys
        x11 = gt_kpts[..., 20] - xs
        y11 = gt_kpts[..., 21] - ys
        x12 = gt_kpts[..., 22] - xs
        y12 = gt_kpts[..., 23] - ys
        x13 = gt_kpts[..., 24] - xs
        y13 = gt_kpts[..., 25] - ys
        x14 = gt_kpts[..., 26] - xs
        y14 = gt_kpts[..., 27] - ys
        x15 = gt_kpts[..., 28] - xs
        y15 = gt_kpts[..., 29] - ys
        x16 = gt_kpts[..., 30] - xs
        y16 = gt_kpts[..., 31] - ys
        x17 = gt_kpts[..., 32] - xs
        y17 = gt_kpts[..., 33] - ys


        kpt_targets = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, \
            x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, x13, y13, x14, y14,\
                x15, y15, x16, y16, x17, y17], -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        kpt_targets = kpt_targets[range(num_points), min_area_inds]
        kpt_vis_flag = kpt_vis_flag[range(num_points), min_area_inds]

        return labels, bbox_targets, kpt_targets, kpt_vis_flag, hm_targets

