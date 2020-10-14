import shutil
from tqdm import tqdm

#import config.yolov4_config as cfg
import time
import torch.nn.functional as F
import os
import numpy as np
import torch
#from eval import voc_eval
#from utils.data_augment import *
#from utils.tools import *
#from utils.visualize import *
#from utils.heatmap import imshowAtt

current_milli_time = lambda: int(round(time.time() * 1000))


def iou_xyxy_numpy(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    if boxes1.shape[-1]==6:
        boxes1_area = np.multiply.reduce(boxes1[..., 3:6] - boxes1[..., 0:3], axis=-1)
        boxes2_area = np.multiply.reduce(boxes2[..., 3:6] - boxes2[..., 0:3], axis=-1)
        # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
        left_up = np.maximum(boxes1[..., :3], boxes2[..., :3])
        right_down = np.minimum(boxes1[..., 3:], boxes2[..., 3:])
    else:
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = np.multiply.reduce(inter_section, axis=-1)
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / (union_area+1e-3)
    return IOU

def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms', box_top_k=50):
    """
    :param bboxes:
    假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    :return: best_bboxes
    假设NMS后剩下N个bbox，那么best_bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    """
    if bboxes.shape[-1]==8:
        classes_in_img = list(set(bboxes[:, 7].astype(np.int32)))
    else:
        classes_in_img = list(set(bboxes[:, 5].astype(np.int32)))
    classes_in_img = [_ for _ in classes_in_img if not _==0]
    best_bboxes = []
    score_top_k_list = []
    for cls in classes_in_img:
        if bboxes.shape[-1]==8:
            cls_mask = (bboxes[:, 7].astype(np.int32) == cls)
        else:
            cls_mask = (bboxes[:, 5].astype(np.int32) == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            if bboxes.shape[-1]==8:
                max_ind = np.argmax(cls_bboxes[:, 6])
            else:
                max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            if bboxes.shape[-1]==8:
                iou = iou_xyxy_numpy(best_bbox[np.newaxis, :6], cls_bboxes[:, :6])
            else:
                iou = iou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            if bboxes.shape[-1]==8:
                cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
                score_mask = cls_bboxes[:, 6] > score_threshold
            else:
                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    best_bboxes = np.array(best_bboxes)
    top_k_bboxes = []
    print(len(best_bboxes), " bboxes after nms")
    min_conf = -1
    for i in range(min(len(best_bboxes), box_top_k)):
        max_ind = np.argmax(best_bboxes[:, -2])
        best_bbox = best_bboxes[max_ind]
        top_k_bboxes.append(best_bbox)
        best_bboxes = np.concatenate([best_bboxes[: max_ind], best_bboxes[max_ind + 1:]])
        if min_conf==-1 or min_conf>best_bbox[-2]:
            min_conf=best_bbox[-2]
    print("min confidence in top ", len(top_k_bboxes), " boxes:" , min_conf)
    return np.array(top_k_bboxes)

class Evaluator(object):
    def __init__(self, model, showatt, pred_result_path, box_top_k, val_shape, CONF_THRESH=0.000001, NMS_THRESH=0.45):
        self.classes = 2
        self.pred_result_path = pred_result_path


        self.val_shape = val_shape
        self.model = model
        self.device = 'cuda'#next(model.parameters()).device
        self.__visual_imgs = 0
        self.showatt = showatt
        self.inference_time = 0.

        self.conf_thresh = CONF_THRESH
        self.nms_thresh = NMS_THRESH
        self.box_top_k = box_top_k


    def store_bbox(self, img_ind, bboxes_prd):
        #'/data/bruce/CenterNet_ABUS/results/prediction/new_CASE_SR_Li^Ling_1073_201902211146_1.3.6.1.4.1.47779.1.002.npy'
        boxes = bboxes_prd[..., :7]
        if len(boxes)>0:
            boxes=boxes
        np.save(os.path.join(self.pred_result_path, img_ind), boxes)


    def get_bbox(self, img, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(320, 640, 96)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, self.val_shape, (0, np.inf))

        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh, self.box_top_k)

        return bboxes

    def __predict(self, img, test_shape, valid_scale):
        org_img = img
        if len(org_img.size())==4:
            _, org_d, org_h, org_w = org_img.size()
            org_shape = (org_d, org_h, org_w)
            img = img.unsqueeze(0)
            if (test_shape==org_shape):
                pass
            else:
                img = F.interpolate(img, size=test_shape, mode='trilinear')
        else:
            _, org_h, org_w = org_img.size()
            org_shape = (org_h, org_w)
            img = img.unsqueeze(0)
            if (test_shape==org_shape):
                pass
            else:
                img = F.interpolate(img, size=test_shape, mode='bilinear')
        self.model.eval()
        with torch.no_grad():
            start_time = current_milli_time()
            if self.showatt: _, p_d, beta = self.model(img)
            else: bboxes = self.model(img)
            self.inference_time += (current_milli_time() - start_time)
        pred_bbox = np.zeros((len(bboxes[0].bbox), 8), dtype=np.float)
        pred_bbox[:, :6] = bboxes[0].bbox.detach().cpu().numpy()
        pred_bbox[:, 6:7] = bboxes[0].extra_fields['scores'].detach().unsqueeze(-1).cpu().numpy()
        pred_bbox[:, 7:8] = bboxes[0].extra_fields['labels'].detach().unsqueeze(-1).cpu().numpy()

        #pred_bbox = bboxes.squeeze().cpu().numpy()
        #bboxes = self.__convert_pred(pred_bbox, test_shape, org_shape, valid_scale)
        #if self.showatt and len(img):
        #    self.__show_heatmap(beta[2], np.copy(org_img.cpu().numpy()))
        return pred_bbox

    def __show_heatmap(self, beta, img):
        imshowAtt(beta, img)

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        Filter out the prediction box to remove the unreasonable scale of the box
        """

        if len(org_img_shape)==3:
            pred_coor = xyzwhd2xyzxyz(pred_bbox[:, :6])
            pred_conf = pred_bbox[:, 6]
            pred_prob = pred_bbox[:, 7:]
            # (1)
            # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
            # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
            org_d, org_h, org_w = org_img_shape
            resize_ratio = 1.0 * min([test_input_size[i] / org_img_shape[i] for i in range(3)])

            dd = (test_input_size[0] - resize_ratio * org_d) / 2
            dh = (test_input_size[1] - resize_ratio * org_h) / 2
            dw = (test_input_size[2] - resize_ratio * org_w) / 2

            pred_coor[:, 0::3] = 1.0 * (pred_coor[:, 0::3] - dd) / resize_ratio
            pred_coor[:, 1::3] = 1.0 * (pred_coor[:, 1::3] - dh) / resize_ratio
            pred_coor[:, 2::3] = 1.0 * (pred_coor[:, 2::3] - dw) / resize_ratio

        else:
            pred_coor = xywh2xyxy(pred_bbox[:, :4])
            pred_conf = pred_bbox[:, 4]
            pred_prob = pred_bbox[:, 5:]
            # (1)
            # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
            # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
            org_h, org_w = org_img_shape
            resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
            dw = (test_input_size - resize_ratio * org_w) / 2
            dh = (test_input_size - resize_ratio * org_h) / 2
            pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
            pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        if len(org_img_shape)==3:
            # (2)Crop off the portion of the predicted Bbox that is beyond the original image
            pred_coor = np.concatenate([np.maximum(pred_coor[:, :3], [0, 0, 0]),
                                        np.minimum(pred_coor[:, 3:], [org_d - 1, org_h - 1, org_w - 1])], axis=-1)
            # (3)Sets the coor of an invalid bbox to 0
            invalid_mask = np.logical_or((pred_coor[:, 1] > pred_coor[:, 4]), (pred_coor[:, 2] > pred_coor[:, 5]))
            pred_coor[invalid_mask] = 0
            invalid_mask = (pred_coor[:, 0] > pred_coor[:, 3])
            pred_coor[invalid_mask] = 0

            # (4)Remove bboxes that are not in the valid range
            bboxes_scale = np.multiply.reduce(pred_coor[:, 3:6] - pred_coor[:, 0:3], axis=-1)
            v_scale_3 = np.power(valid_scale, 3.0)
            scale_mask = np.logical_and((v_scale_3[0] < bboxes_scale), (bboxes_scale < v_scale_3[1]))
        else:
            # (2)Crop off the portion of the predicted Bbox that is beyond the original image
            pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                        np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
            # (3)Sets the coor of an invalid bbox to 0
            invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
            pred_coor[invalid_mask] = 0

            # (4)Remove bboxes that are not in the valid range
            bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
            scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))


        # (5)Remove bboxes whose score is below the score_threshold
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        return bboxes
    def clear_predict_file(self):
        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.mkdir(self.pred_result_path)
