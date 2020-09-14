

from data.abus_data import AbusNpyFormat
from .image3d import ImageDetect3DDataset
import torch
from fcos_core.structures.bounding_box import BoxList

#usage:
#for COCO
#   imageDataset = torchvision.datasets.coco.CocoDetection(root, ann_file)
#   final_dataset = ImageDetect3DDataset(imageDataset, transforms)
#for ABUS
#   imageDataset = ABUSDetectionDataset(
#       root, transform=None, target_transform=None, transforms=None, \
#       crx_valid=False, crx_fold_num=0, crx_partition='train', \
#       augmentation=False, include_fp=False)
#   final_dataset = ImageDetect3DDataset(imageDataset, transforms)

class ABUSDetectionDataset(ImageDetect3DDataset):
    """`NTU402 ABUS Detection Dataset`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    # read from AbusNpyFormat(root, annFile)
    def __init__(self, root, transform=None, target_transform=None, transforms=None, \
            crx_valid=False, crx_fold_num=0, crx_partition='train', \
            augmentation=False, include_fp=False):
        #for __repr__
        self.abusNpy = AbusNpyFormat(root, \
            crx_valid, crx_fold_num, crx_partition, \
            augmentation, include_fp)

        self.ids=[self.abusNpy.getID(i) for i in range(len(self.abusNpy))]
        self.ids = list(sorted(self.ids))

        super(ABUSDetectionDataset, self).__init__(root, transforms=transforms)

    def getCatIds(self):
        return [0, 1] #background=0  tumor=1

    def __provide_items__(self, index):
        img, target = self.abusNpy[index]

        masks = None
        keypoints = None

        anno = []

        for item in target:
            x = item['x_bot']
            y = item['y_bot']
            z = item['z_bot']
            w = item['x_top'] - item['x_bot'] + 1e-2
            h = item['y_top'] - item['y_bot'] + 1e-2
            d = item['z_top'] - item['z_bot'] + 1e-2
            ann = {'bbox':[x, y, z, w, h, d],
                'category_id':1,
            }
            anno.append(ann)
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 6)  # guard against no boxes
        target = BoxList(boxes, img.size()[-3:], mode="xyzwhd").convert("xyzxyz")
        classes = [obj["category_id"] for obj in anno]

        masks = None
        keypoints = None
        return img, target, classes, masks, keypoints



        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        classes = [obj["category_id"] for obj in anno]

        masks = None
        keypoints = None
        return img, target, classes, masks, keypoints

    def get_img_info(self, index):
        return {'height':self.abusNpy.img_size[0], \
            'width':self.abusNpy.img_size[2], \
            'depth':self.abusNpy.img_size[1]}
