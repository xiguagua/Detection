from typing import List
# from torch.nn.modules import padding
from detectron2.modeling import META_ARCH_REGISTRY

import torch
import torch.nn as nn

from detectron2.modeling.backbone.build import build_backbone
from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

from xgdetectron.modeling.losses import FocalLoss, RegL1Loss, RegLoss
from xgdetectron.structures.keypoints import draw_umich_gaussian, draw_dense_reg, gaussian_radius

__all__ = ['CenterNet']


@META_ARCH_REGISTRY.register()
class CenterNet(nn.Module):
    """
        Implement CenterNet in :paper:`Objects as Points`.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone,
        head,
        hm_loss,
        hm_weight,
        wh_loss,
        wh_weight,
        num_classes,
        down_ratio,
        # max_objs,
        head_in_features,
        pixel_mean,
        pixel_std,

    ):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.hm_loss = hm_loss
        self.hm_weight = hm_weight
        self.wh_loss = wh_loss
        self.wh_weight = wh_weight
        self.num_classes = num_classes
        self.down_ratio = down_ratio
        # self.max_objs = max_objs
        self.head_in_features = head_in_features


        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f]
                        for f in cfg.MODEL.CENTERNET.IN_FEATURES]
        head = CenterNetHead(cfg, feature_shapes)
        wh_loss = cfg.MODEL.CENTERNET.WH_LOSS
        assert wh_loss in ('l1', 'sl1'), "Only `l1`, `sl1` are supported!"
        return {
            "backbone": backbone,
            "head": head,
            "hm_loss": FocalLoss(),
            "wh_loss": {"l1": RegL1Loss, "sl1": RegLoss}[wh_loss](),
            "hm_weight": cfg.MODEL.CENTERNET.HM_WEIGHT,
            "wh_weight": cfg.MODEL.CENTERNET.WH_WEIGHT,
            "num_classes": cfg.MODEL.CENTERNET.NUM_CLASSES,
            "down_ratio": cfg.MODEL.CENTERNET.DOWN_RATIO,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "head_in_features": cfg.MODEL.CENTERNET.IN_FEATURES,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                See :meth:`postprocess` for details.
        Returns:
            in training, dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
            in inference, the standard output format, described in :doc:`/tutorials/models`.
        """
        images = self.preprocess_images(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        assert (features[0].shape[-2], features[0].shape[-1]) == \
            (images.image_sizes[0]/self.down_ratio, images.image_sizes[1]/self.down_ratio)
        # pred_hm, pred_wh, pred_hm_reg = self.head(features)
        pred_hm, pred_wh = self.head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert 'instances' in batched_inputs[0], 'Instance annotations are missing in training!'
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_hms, gt_whs, gt_inds, gt_reg_masks = self.label_points(images, gt_instances)
            losses = self.losses(gt_inds, gt_reg_masks, pred_hm, gt_hms, pred_wh, gt_whs)

            return losses

    def losses(self, inds, reg_masks, pred_hm, gt_hms, pred_wh, gt_whs):
        loss_hm, loss_wh_reg = 0, 0
        pred_hm = torch.clamp(torch.sigmoid(pred_hm), min=1e-4, max=1-1e-4)
        gt_hms = torch.stack(gt_hms)
        gt_whs = torch.stack(gt_whs)
        inds = torch.stack(inds)
        reg_masks = torch.stack(reg_masks)

        loss_hm += self.hm_loss(pred_hm, gt_hms)
        loss_wh_reg += self.wh_loss(pred_wh, gt_whs, inds, reg_masks)

        # loss = self.hm_weight * loss_hm + self.wh_weight * loss_wh_reg
        return {
            "loss_hm": self.hm_weight * loss_hm,
            "loss_wh_reg": self.wh_weight * loss_wh_reg,
        }

    @torch.no_grad()
    def label_points(self, images, gt_instances):
        down_ratio = self.down_ratio
        input_h, input_w = images.tensor.shape[-2:]
        output_h, output_w = input_h // down_ratio, input_w // down_ratio
        draw_gaussian = draw_umich_gaussian

        max_objs = max([len(i.gt_boxes) for i in gt_instances])

        gt_hms = []
        gt_whs = []
        gt_inds = []
        gt_reg_masks = []
        for gt_per_image in gt_instances:
            scale_x = scale_y = 1 / down_ratio
            boxes = gt_per_image.gt_boxes
            boxes.scale(scale_x, scale_y) 
            classes = gt_per_image.gt_classes

            # num_objs = min(len(boxes), self.max_objs)
            num_objs = len(boxes)
            radius = gaussian_radius(boxes)
            cts = boxes.get_centers().type(torch.int32)
            hm = torch.zeros((self.num_classes, output_h, output_w), dtype=torch.float32)  
            wh = torch.zeros((max_objs, 2), dtype=torch.float32)
            ind = torch.zeros((max_objs), dtype=torch.int16)
            reg_mask = torch.zeros((max_objs), dtype=torch.int8)

            for i in range(num_objs):
                ct = cts[i]
                ind[i] = ct[1] * output_w + ct[0]

                cls_id = classes[i]
                draw_gaussian(hm[cls_id], cts[i], radius[i])

                box = boxes.tensor[i]
                b_w, b_h = box[2] - box[0], box[3] - box[1]
                wh[i] = b_w, b_h
                reg_mask[i] = 1
                # draw_dense_reg(
                #     wh,
                #     hm.max(dim=0).values,
                #     cts[i],
                #     (b_w, b_h),
                #     radius[i]
                # )

            gt_hms.append(hm)
            gt_whs.append(wh)
            gt_inds.append(ind)
            gt_reg_masks.append(reg_mask)

        return gt_hms, gt_whs, gt_inds, gt_reg_masks        
    

    def preprocess_images(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class CenterNetHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        input_shape,
        num_classes,
        conv_dims: List[int],
        ):
        super().__init__()

        hm_subnet = []
        wh_subnet = []

        for in_channels, out_channels in zip([input_shape[0].channels] + conv_dims, conv_dims):
            hm_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            )
            hm_subnet.append(nn.ReLU())
            wh_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
            )
            wh_subnet.append(nn.ReLU())

        self.hm_subnet = nn.Sequential(*hm_subnet)
        self.wh_subnet = nn.Sequential(*wh_subnet)

        self.hm_pred = nn.Conv2d(
            conv_dims[-1], num_classes, kernel_size=1, stride=1, padding=0
        )
        self.wh_pred = nn.Conv2d(
            conv_dims[-1], 2, kernel_size=1, stride=1, padding=0
        )

        # self.hm_reg = nn.Conv2d(
        #     in_channels=input_shape[0].channels,
        #     out_channels=2,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.CENTERNET.NUM_CLASSES,
            'conv_dims': cfg.MODEL.CENTERNET.HEAD_CONV_DIMS, # [64]
        }

    def forward(self, features):
        # return self.hm(features[0]), self.wh(features[0]), self.hm_reg(features[0])
        return ( 
            self.hm_pred(self.hm_subnet(features[0])), 
            self.wh_pred(self.wh_subnet(features[0])),
        )