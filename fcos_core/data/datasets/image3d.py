
import torch
from torchvision.datasets.vision import VisionDataset
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.segmentation_mask import SegmentationMask
from fcos_core.structures.keypoint import PersonKeypoints

class ImageDetect3DDataset(VisionDataset):
    def __init__(
        self, root, transforms=None, Image3DDataset=None
    ):
        super(ImageDetect3DDataset, self).__init__(root, transforms=transforms)
        # sort indices for reproducible results
        #self.ids = sorted(Image3DDataset.ids)
        #self.Image3D = Image3DDataset

        #super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        #img, anno = super(COCODataset, self).__getitem__(idx)
        #img, anno = self.__provide_items__(idx)
        img, target, classes, masks, keypoints, img_id = self.__provide_items__(idx)

        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if masks:
            target.add_field("masks", masks)
        if keypoints:
            target.add_field("keypoints", keypoints)

        old_target = target
        target = target.clip_to_image(remove_empty=True)
        assert len(target) == len(target), 'box removed!'
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx, img_id

    def __len__(self):
        return len(self.ids)

    def __provide_items__(self, index):
        raise NotImplementedError

    def get_img_info(self, index):
        raise NotImplementedError

    def getCatIds(self):
        raise NotImplementedError