import numpy as np
import torch.nn as nn
from operator import add
from .misc import categorize_by_size

def nms(heat, kernel=5):
    pad = (kernel - 1) // 2

    m = nn.MaxPool3d(kernel, stride=1, padding=pad)
    hmax = m(heat)
    keep = (hmax == heat).float()
    return heat * keep


def max_in_neighborhood(whmap, kernel=5):
    pad = (kernel - 1) // 2

    m = nn.MaxPool3d(kernel, stride=1, padding=pad)
    hmax = m(whmap)
    return hmax


def compute_iou(box1, box2, scale):
    '''
        by Eason Ho
    '''
    b1_z0, b1_y0, b1_x0, b1_z1, b1_y1, b1_x1 = box1
    b2_z0, b2_y0, b2_x0, b2_z1, b2_y1, b2_x1 = box2

    b1_z0, b1_y0, b1_x0, b1_z1, b1_y1, b1_x1 = int(b1_z0), int(
        b1_y0), int(b1_x0), int(b1_z1), int(b1_y1), int(b1_x1)
    b2_z0, b2_y0, b2_x0, b2_z1, b2_y1, b2_x1 = int(b2_z0*scale[0]), int(
        b2_y0*scale[1]), int(b2_x0*scale[2]), int(b2_z1*scale[0]), int(b2_y1*scale[1]), int(b2_x1*scale[2])

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_z0 = max(b1_z0, b2_z0)

    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)
    int_z1 = min(b1_z1, b2_z1)

    int_x = int_x1 - int_x0
    int_y = int_y1 - int_y0
    int_z = int_z1 - int_z0

    if int_x <= 0 or int_y <= 0 or int_z <= 0:
        return 0.

    int_area = ((int_x) * (int_y) * (int_z))

    b1_area = ((b1_x1 - b1_x0) * (b1_y1 - b1_y0) * (b1_z1 - b1_z0))
    b2_area = ((b2_x1 - b2_x0) * (b2_y1 - b2_y0) * (b2_z1 - b2_z0))
    iou = int_area / (b1_area + b2_area - int_area + 1e-9)
    return iou


def eval_precision_recall(pred_BB, true_BB, det_thresh, scale):
    '''
        by Eason Ho
    '''
    pred_hits = np.zeros(len(pred_BB))
    gt_hits = np.zeros(len(true_BB))
    hits_index = -np.ones(len(true_BB))
    hits_iou = np.zeros(len(true_BB), dtype=float)
    hits_score = np.zeros(len(true_BB), dtype=float)

    for pred_idx, pred_bb in enumerate(pred_BB):
        for gt_idx, gt_roi in enumerate(true_BB):
            pred_iou = compute_iou(pred_bb[:6], gt_roi[:6], scale)
            if pred_iou > det_thresh:
                gt_hits[gt_idx] = 1
                hits_index[gt_idx] = pred_idx
                hits_iou[gt_idx] = pred_iou
                hits_score[gt_idx]=pred_bb[6]
                pred_hits[pred_idx] = 1

    TP = gt_hits.sum()
    FP = len(pred_hits) - pred_hits.sum()
    FN = len(true_BB)-gt_hits.sum()
    return int(TP),int(FP),int(FN),hits_index,hits_iou,hits_score

# scale = (640/line[1], 160/line[2], 640/line[3])
def centroid_distance(box1, box2, scale):
    b1_z0, b1_y0, b1_x0, b1_z1, b1_y1, b1_x1 = box1
    b2_z0, b2_y0, b2_x0, b2_z1, b2_y1, b2_x1 = box2

    b1_z0, b1_y0, b1_x0, b1_z1, b1_y1, b1_x1 = \
        int(b1_z0/scale[0]/4), int(b1_y0/scale[1]/4), int(b1_x0/scale[2]/4), int(b1_z1/scale[0]/4), int(b1_y1/scale[1]/4), int(b1_x1/scale[2]/4)

    b2_z0, b2_y0, b2_x0, b2_z1, b2_y1, b2_x1 = int(b2_z0/4), int(b2_y0/4), int(b2_x0/4), int(b2_z1/4), int(b2_y1/4), int(b2_x1/4)

    b1_centroid_z, b1_centroid_y, b1_centroid_x = (b1_z1+b1_z0)/2, (b1_y1+b1_y0)/2, (b1_x1+b1_x0)/2
    b2_centroid_z, b2_centroid_y, b2_centroid_x = (b2_z1+b2_z0)/2, (b2_y1+b2_y0)/2, (b2_x1+b2_x0)/2

    b1_centroid = np.array([b1_centroid_z, b1_centroid_y, b1_centroid_x])
    b2_centroid = np.array([b2_centroid_z, b2_centroid_y, b2_centroid_x])

    dist = np.linalg.norm(b1_centroid - b2_centroid)

    return dist


def eval_precision_recall_by_dist(pred_BB, true_BB, dist_thresh, scale):

    pred_hits = np.zeros(len(pred_BB))
    gt_hits = np.zeros(len(true_BB))
    hits_index = -np.ones(len(true_BB))
    hits_iou = np.zeros(len(true_BB), dtype=float)
    hits_score = np.zeros(len(true_BB), dtype=float)

    # S(<10mm), M(10mm<15mm), L(15mm<)
    TP_by_size = [0,0,0]

    for gt_idx, gt_roi in enumerate(true_BB):
        for pred_idx, pred_bb in enumerate(pred_BB):
            dist = centroid_distance(pred_bb[:6], gt_roi[:6], scale)
            if dist <= dist_thresh:
                gt_hits[gt_idx] = 1
                hits_index[gt_idx] = pred_idx
                hits_iou[gt_idx] = dist
                hits_score[gt_idx] = pred_bb[6]
                pred_hits[pred_idx] = 1

        if gt_hits[gt_idx] == 1:
            TP_by_size = list(map(add, TP_by_size, categorize_by_size(gt_roi[:6])))

    TP = gt_hits.sum()
    FP = len(pred_hits) - pred_hits.sum()
    FN = len(true_BB)-gt_hits.sum()
    return int(TP), int(FP), int(FN), hits_index, hits_iou, hits_score, TP_by_size

def box_to_string(bbox):
    separator = ','
    return separator.join(list(map(lambda x: str(int(x)), bbox)))

def pick_fp_by_dist(pred_BB, true_BB, dist_thresh, scale):

    pred_hits = np.zeros(len(pred_BB))
    fp_list = []

    for pred_idx, pred_bb in enumerate(pred_BB):
        for gt_idx, gt_roi in enumerate(true_BB):
            dist = centroid_distance(pred_bb[:6], gt_roi[:6], scale)
            if dist <= dist_thresh:
                pred_hits[pred_idx] = 1

        if pred_hits[pred_idx] == 0:
            fp_list.append(box_to_string(pred_bb))

    FP = len(pred_BB) - pred_hits.sum()

    return int(FP), fp_list