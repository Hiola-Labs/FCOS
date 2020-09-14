# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
FLIP_FRONT_BEHIND = 2

class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        self.D3 = False
        if mode=='xyzxyz' or mode == 'xyzwhd':
            self.D3 = True
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        #modify 3D
        if bbox.size(-1) != (6 if self.D3 else 4):
            raise ValueError(
                "last dimension of bbox should have a "
                "size of {}, got {}".format((6 if self.D3 else 4), bbox.size(-1))
            )
        if mode not in (("xyzxyz", "xyzwhd") if self.D3 else ("xyxy", "xywh")):
            if self.D3:
                raise ValueError("mode should be 'xyzxyz' or 'xyzwhd'")
            else:
                raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in (("xyzxyz", "xyzwhd") if self.D3 else ("xyxy", "xywh")):
            if self.D3:
                raise ValueError("mode should be 'xyzxyz' or 'xyzwhd'")
            else:
                raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        if self.D3:
            # we only have two modes, so don't need to check
            # self.mode
            xmin, ymin, zmin, xmax, ymax, zmax = self._split_into_xyxy()
            if mode == "xyzxyz":
                bbox = torch.cat((xmin, ymin, zmin, xmax, ymax, zmax), dim=-1)
                bbox = BoxList(bbox, self.size, mode=mode)
            else:
                TO_REMOVE = 1
                bbox = torch.cat(
                    (xmin, ymin, zmin,
                    xmax - xmin + TO_REMOVE,
                    ymax - ymin + TO_REMOVE,
                    zmax - zmin + TO_REMOVE), dim=-1
                )
                bbox = BoxList(bbox, self.size, mode=mode)
        else:
            # we only have two modes, so don't need to check
            # self.mode
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
            if mode == "xyxy":
                bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
                bbox = BoxList(bbox, self.size, mode=mode)
            else:
                TO_REMOVE = 1
                bbox = torch.cat(
                    (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
                )
                bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        elif self.mode == "xyzxyz":
            xmin, ymin, zmin, xmax, ymax, zmax = self.bbox.split(1, dim=-1)
            return xmin, ymin, zmin, xmax, ymax, zmax
        elif self.mode == "xyzwhd":
            TO_REMOVE = 1
            xmin, ymin, zmin, w, h, d = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                zmin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
                zmin + (d - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_same = False
        if self.D3:
            if (ratios[0] == ratios[1] and ratios[0] == ratios[2]):
                ratio_same = True
        else:
            if (ratios[0] == ratios[1]):
                ratio_same = True
        if ratio_same:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox


        if self.D3:
            ratio_width, ratio_height, ratio_depth = ratios
        else:
            ratio_width, ratio_height = ratios
        if self.D3:
            xmin, ymin, zmin, xmax, ymax, zmax = self._split_into_xyxy()
        else:
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        if self.D3:
            scaled_zmin = zmin * ratio_depth
            scaled_zmax = zmax * ratio_depth
            scaled_box = torch.cat(
                (scaled_xmin, scaled_ymin, scaled_zmin, scaled_xmax, scaled_ymax, scaled_zmax), dim=-1
            )
            bbox = BoxList(scaled_box, size, mode="xyzxyz")
        else:
            scaled_box = torch.cat(
                (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
            )
            bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, FLIP_FRONT_BEHIND):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )
        if self.D3:
            image_width, image_height, image_depth = self.size
            xmin, ymin, zmin, xmax, ymax, zmax = self._split_into_xyxy()
        else:
            image_width, image_height = self.size
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
        TO_REMOVE = 1
        if method == FLIP_LEFT_RIGHT:
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
            if self.D3:
                transposed_zmin = zmin
                transposed_zmax = zmax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax - TO_REMOVE
            transposed_ymax = image_height - ymin - TO_REMOVE
            if self.D3:
                transposed_zmin = zmin
                transposed_zmax = zmax
        elif method == FLIP_FRONT_BEHIND:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = ymin
            transposed_ymax = ymax
            #only 3D box allow to use FLIP_FRONT_BEHIND
            transposed_zmin = image_depth - zmax - TO_REMOVE
            transposed_zmax = image_depth - zmin - TO_REMOVE


        if self.D3:
            transposed_boxes = torch.cat(
                (transposed_xmin, transposed_ymin, transposed_zmin, transposed_xmax, transposed_ymax, transposed_zmax), dim=-1
            )
            bbox = BoxList(transposed_boxes, self.size, mode="xyzxyz")
        else:
            transposed_boxes = torch.cat(
                (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
            )
            bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        if self.D3:
            xmin, ymin, zmin, xmax, ymax, zmax = self._split_into_xyxy()
            w, h, d = box[3] - box[0], box[4] - box[1], box[5] - box[2]
        else:
            xmin, ymin, xmax, ymax = self._split_into_xyxy()
            w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)
        if self.D3:
            cropped_zmin = (zmin - box[2]).clamp(min=0, max=d)
            cropped_zmax = (zmax - box[2]).clamp(min=0, max=d)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)
            if self.D3:
                is_empty = is_empty | (cropped_zmin == cropped_zmax)
        if self.D3:
            cropped_box = torch.cat(
                (cropped_xmin, cropped_ymin, cropped_zmin, cropped_xmax, cropped_ymax, cropped_zmax), dim=-1
            )
            bbox = BoxList(cropped_box, (w, h, d), mode="xyzxyz")
        else:
            cropped_box = torch.cat(
                (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
            )
            bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        if self.D3:
            #x
            self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
            self.bbox[:, 3].clamp_(min=0, max=self.size[0] - TO_REMOVE)
            #y
            self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
            self.bbox[:, 4].clamp_(min=0, max=self.size[1] - TO_REMOVE)
            #z
            self.bbox[:, 2].clamp_(min=0, max=self.size[2] - TO_REMOVE)
            self.bbox[:, 5].clamp_(min=0, max=self.size[2] - TO_REMOVE)
            if remove_empty:
                box = self.bbox
                keep = (box[:, 5] > box[:, 2]) & (box[:, 4] > box[:, 1]) & (box[:, 3] > box[:, 0])
                return self[keep]
        else:
            self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
            self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
            self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
            self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
            if remove_empty:
                box = self.bbox
                keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
                return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        elif self.mode == "xyzxyz":
            TO_REMOVE = 1
            area = (box[:, 3] - box[:, 0] + TO_REMOVE) * (box[:, 4] - box[:, 1] + TO_REMOVE) * (box[:, 5] - box[:, 2] + TO_REMOVE)
        elif self.mode == "xyzwhd":
            area = box[:, 3] * box[:, 4] * box[:, 5]
        else:
            raise RuntimeError("Should not be here")
        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        if len(self.size)>2:
            s += "image_depth={}, ".format(self.size[2])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 9, 9], [0, 0, 5, 5]], (10, 10), mode='xyxy')
    s_bbox = bbox.resize((5, 5))
    assert_result = torch.tensor([[0, 0, 4.5, 4.5], [0, 0, 2.5, 2.5]]).float()
    assert torch.equal(s_bbox.bbox, assert_result)

    t_bbox = bbox.transpose(0)
    assert_result = torch.tensor([[0, 0, 9, 9], [4, 0, 9, 5]]).float()
    assert torch.equal(t_bbox.bbox, assert_result)


    crop_bbox = bbox.crop([3, 3, 7, 7])
    assert_result = torch.tensor([[0, 0, 4, 4], [0, 0, 2, 2]]).float()
    assert torch.equal(crop_bbox.bbox, assert_result)
    clip_bbox = BoxList([[2, 2, 15, 25], [0, 0, 8, 18]], (10, 20), mode='xyxy')
    clip_bbox = clip_bbox.clip_to_image()
    assert_result = torch.tensor([[2, 2, 9, 19], [0, 0, 8, 18]]).float()
    assert torch.equal(clip_bbox.bbox, assert_result)

    bbox = BoxList([[0, 0, 0, 8, 6, 4], [8, 6, 4, 16, 12, 8]], (20, 30, 40), mode='xyzxyz')
    s_bbox = bbox.resize((10, 10, 10))
    assert_result = torch.tensor([[0, 0, 0, 4, 2, 1], [4, 2, 1, 8, 4, 2]]).float()
    assert torch.equal(s_bbox.bbox, assert_result)

    t_bbox = bbox.transpose(0)
    assert_result = torch.tensor([[11, 0, 0, 19, 6, 4], [3, 6, 4, 11, 12, 8]]).float()
    assert torch.equal(t_bbox.bbox, assert_result)
    t_bbox = bbox.transpose(1)
    assert_result = torch.tensor([[0, 23, 0, 8, 29, 4], [8, 17, 4, 16, 23, 8]]).float()
    assert torch.equal(t_bbox.bbox, assert_result)
    t_bbox = bbox.transpose(2)
    assert_result = torch.tensor([[0, 0, 35, 8, 6, 39], [8, 6, 31, 16, 12, 35]]).float()
    assert torch.equal(t_bbox.bbox, assert_result)


    crop_bbox = bbox.crop([3, 3, 3, 9, 9, 9])
    assert_result = torch.tensor([[0, 0, 0, 5, 3, 1], [5, 3, 1, 6, 6, 5]]).float()
    assert torch.equal(crop_bbox.bbox, assert_result)
    clip_bbox = BoxList([[2, 2, 2, 15, 25, 35], [0, 0, 0, 8, 18, 28]], (10, 20, 30), mode='xyzxyz')
    clip_bbox = clip_bbox.clip_to_image()
    assert_result = torch.tensor([[2, 2, 2, 9, 19, 29], [0, 0, 0, 8, 18, 28]]).float()
    assert torch.equal(clip_bbox.bbox, assert_result)


    print('done')



