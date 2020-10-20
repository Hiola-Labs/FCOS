import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.modeling.utils import cat
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_ml_nms
from fcos_core.structures.boxlist_ops import remove_small_boxes


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """

        is_3D = False
        if len(box_cls.shape)==5:
            is_3D = True
        if is_3D:
            N, C, D, H, W = box_cls.shape
        else:
            N, C, H, W = box_cls.shape


        if is_3D:
            # put in the same format as locations
            box_cls = box_cls.view(N, C, D, H, W).permute(0, 2, 3, 4, 1)
            box_cls = box_cls.reshape(N, -1, C).sigmoid()
            box_regression = box_regression.view(N, 6, D, H, W).permute(0, 2, 3, 4, 1)
            box_regression = box_regression.reshape(N, -1, 6)
            centerness = centerness.view(N, 1, D, H, W).permute(0, 2, 3, 4, 1)
            centerness = centerness.reshape(N, -1).sigmoid()

            candidate_inds = box_cls > self.pre_nms_thresh
            pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
            pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

            # multiply the classification scores with centerness scores
            box_cls = box_cls * centerness[:, :, None]

        else:
            # put in the same format as locations
            box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
            box_cls = box_cls.reshape(N, -1, C).sigmoid()
            box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
            box_regression = box_regression.reshape(N, -1, 4)
            centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
            centerness = centerness.reshape(N, -1).sigmoid()

            candidate_inds = box_cls > self.pre_nms_thresh
            pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
            pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

            # multiply the classification scores with centerness scores
            box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            if is_3D:
                stack_list = [
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 2] - per_box_regression[:, 2],
                    per_locations[:, 0] + per_box_regression[:, 3],
                    per_locations[:, 1] + per_box_regression[:, 4],
                    per_locations[:, 2] + per_box_regression[:, 5],
                ]
            else:
                stack_list = [
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 0] + per_box_regression[:, 2],
                    per_locations[:, 1] + per_box_regression[:, 3],
                ]
            detections = torch.stack(stack_list, dim=1)
            if is_3D:
                d, h, w = image_sizes[i]
                boxlist = BoxList(detections, (int(w), int(h), int(d)), mode="xyzxyz")
                boxlist.add_field("scores", torch.pow(per_box_cls, 1/3.0))
            else:
                h, w = image_sizes[i]
                boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
                boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist.add_field("labels", per_class)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        return boxlists


def make_fcos_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector