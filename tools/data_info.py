# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse
import os

if __name__ == "__main__":
    data_root = 'datasets/abus/'

    with open(data_root + 'annotations/rand_all.txt', 'r') as f:
        lines = f.read().splitlines()
    print(len(lines))

    box_list_big = []
    box_list_small = []
    true_box_cache = {}
    for line in lines:
        line = line.split(',', 4)
        pass_id = line[0]
        # Always use 640,160,640 to compute iou
        size = (640,160,640)
        scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))

        boxes = line[-1].split(' ')
        boxes = list(map(lambda box: box.split(','), boxes))
        true_box = [list(map(float, box)) for box in boxes]
        true_box_s = []
        # For the npy volume (after interpolation by spacing), 4px = 1mm
        for li in true_box:
            axis = [0,0,0]
            axis[0] = (li[3] - li[0]) / 4
            axis[1] = (li[4] - li[1]) / 4
            axis[2] = (li[5] - li[2]) / 4
            if axis[0] < 10 and axis[1] < 10 and axis[2] < 10:
                true_box_s.append(li)
        true_box_cache[pass_id] = (true_box, true_box_s)

        box_list_big.append(true_box)
        box_list_small.append(true_box_s)
    print("[BIG] len(boxes) == 1 :", len([k for k in box_list_big if len(k)==1]))
    print("[BIG] len(boxes) == 2 :", len([k for k in box_list_big if len(k)==2]))
    print("[BIG] len(boxes) >  2 :", len([k for k in box_list_big if len(k)> 2]))


    print("[SMALL] len(boxes) == 1 :", len([k for k in box_list_small if len(k)==1]))
    print("[SMALL] len(boxes) == 2 :", len([k for k in box_list_small if len(k)==2]))
    print("[SMALL] len(boxes) >  2 :", len([k for k in box_list_small if len(k)> 2]))
    """
    [BIG] len(boxes) == 1 : 250
    [BIG] len(boxes) == 2 : 68
    [BIG] len(boxes) >  2 : 30
    [SMALL] len(boxes) == 1 : 78
    [SMALL] len(boxes) == 2 : 7
    [SMALL] len(boxes) >  2 : 5
    """



    """
    分配fold的要求:
        1. 大小腫瘤都可以符合 72%:8%:20%的分布
        2. 同一個pass不能在train/val/test重複出現
    由於包含小腫瘤的pass比較少, 先將包含小腫瘤的pass分配成72%:8%:20%
    再將剩餘的pass填入三個partition使testing的pass數達到所有pass的20%, val達到8%
    """
    small_pass=[_ for _ in range(11)]
    for fold_k in range(5):
        step = len(small_pass)//5
        small_test = small_pass[fold_k*(step):(fold_k+1)*step]
        print(small_test)
    print("done")