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
    奕辰的dataset


    /home/lab402/bak/eason_thesis/program(update_v1)/5_fold_list/five_fold_test_0.txt

    /home/lab402/bak/eason_thesis/program(update_v1)/5_fold_list/five_fold_train_0_separate.txt
    /home/lab402/bak/eason_thesis/program(update_v1)/5_fold_list/five_fold_val_0_separate.txt

    /home/lab402/bak/eason_thesis/program(update_v1)/5_fold_list/five_fold_train_0.txt
    /home/lab402/bak/eason_thesis/program(update_v1)/5_fold_list/five_fold_val_0.txt


    """
    for file_name_format in ["/home/lab402/bak/eason_thesis/program(update_v1)/5_fold_list/five_fold_test_{}.txt",
    "/home/lab402/bak/eason_thesis/program(update_v1)/5_fold_list/five_fold_train_{}.txt",
    "/home/lab402/bak/eason_thesis/program(update_v1)/5_fold_list/five_fold_val_{}.txt",
    "/home/lab402/bak/eason_thesis/program(update_v1)/5_fold_list/five_fold_train_{}_separate.txt",
    "/home/lab402/bak/eason_thesis/program(update_v1)/5_fold_list/five_fold_val_{}_separate.txt",]:
        print(file_name_format)
        for i in range(5):
            crx_fold_num = i
            file_name = file_name_format.format(i)
            lines = open(file_name, 'r').readlines()

            total_big = 0
            total_small = 0
            for line in lines:
                line = line.split(',', 4)
                # Always use 640,160,640 to compute iou
                size = (640,160,640)
                scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))

                boxes = line[-1].split(' ')
                boxes = list(map(lambda box: box.split(','), boxes))
                if 'separate' in file_name_format:
                    boxes = boxes[:1]
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

                total_big+=len(true_box)
                total_small+=len(true_box_s)
            print("crx_fold_num:", crx_fold_num, "total_big:", total_big, "total_small:", total_small)

    """
    分配fold的要求:
        1. 大小腫瘤都可以符合 72%:8%:20%的分布
        2. 同一個pass不能在train/val/test重複出現
    由於包含小腫瘤的pass比較少, 先將包含小腫瘤的pass分配成72%:8%:20%
    再將剩餘的pass填入三個partition使testing的pass數達到所有pass的20%, val達到8%

    先將小腫瘤個數>=2的pass依序填入五個fold
    再將小腫瘤個數=1的pass填入各fold直到每個fold的小腫瘤總數誤差不大於一
    """

    with open(data_root + 'annotations/rand_all.txt', 'r') as f:
        lines = f.read().splitlines()
    data_slots = {
        'small_tumor_at_least_5':[],
        'small_tumor_at_least_2':[],
        'small_tumor_eq_1':[],
        'big_tumor_at_least_2':[],
        'big_tumor_eq_1':[],
    }
    for line in lines:
        line = line.split(',', 4)
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

        big_tumor_amount = len(true_box)
        small_tumor_amount = len(true_box_s)
        item = (line, big_tumor_amount, small_tumor_amount)
        if small_tumor_amount>=5: # 5+
            data_slots['small_tumor_at_least_5'].append(item)
        elif small_tumor_amount>=2: #2, 3, 4
            data_slots['small_tumor_at_least_2'].append(item)
        elif small_tumor_amount>0:
            data_slots['small_tumor_eq_1'].append(item)
        elif big_tumor_amount>=2:
            data_slots['big_tumor_at_least_2'].append(item)
        else:
            data_slots['big_tumor_eq_1'].append(item)

    fold_K = 5
    fold_lines = [[] for _ in range(fold_K)]
    fold_tumor_amount = [0] * fold_K

    max_tumor_num = 0
    fold_i = 0
    for items in [data_slots['small_tumor_at_least_2'], data_slots['small_tumor_at_least_5'], data_slots['small_tumor_eq_1']]:
        for line, big_tumor_amount, small_tumor_amount in items:
            min_amount = min(fold_tumor_amount)
            for fold_i in range(fold_K):
                if min_amount==fold_tumor_amount[fold_i]:
                    break

            # start_fold_i = fold_i
            # fold_i = (fold_i+1) % fold_K
            # while fold_tumor_amount[fold_i]+small_tumor_amount>max_tumor_num and start_fold_i != fold_i:
            #     fold_i = (fold_i+1) % fold_K

            item = (line, big_tumor_amount, small_tumor_amount)
            fold_lines[fold_i].append(item)
            fold_tumor_amount[fold_i] += small_tumor_amount
            max_tumor_num = max(max_tumor_num, fold_tumor_amount[fold_i])

    for i in range(fold_K):
        print(" fold_", i, "small tumor total:", fold_tumor_amount[i])
        print(" fold_", i, "small tumor check:", sum([_[2] for _ in fold_lines[i]]))
        print("fold_", i, "small tomor amount list:", [_[2] for _ in fold_lines[i]])


    max_tumor_num = 0
    fold_tumor_amount = [0] * fold_K
    for i in range(fold_K):
        big_tumor_total = sum([_[1] for _ in fold_lines[i]])
        fold_tumor_amount[i] = big_tumor_total
        print(" fold_", i, "big tumor total:", big_tumor_total)
        print("fold_", i, "big tomor amount list:", [_[1] for _ in fold_lines[i]])
        max_tumor_num = max(max_tumor_num, big_tumor_total)


    fold_i = 0
    for items in [data_slots['big_tumor_at_least_2'], data_slots['big_tumor_eq_1']]:
        for line, big_tumor_amount, small_tumor_amount in items:
            min_amount = min(fold_tumor_amount)
            for fold_i in range(fold_K):
                if min_amount==fold_tumor_amount[fold_i]:
                    break

            # start_fold_i = fold_i
            # fold_i = (fold_i+1) % fold_K
            # while fold_tumor_amount[fold_i]+big_tumor_amount>max_tumor_num and start_fold_i != fold_i:
            #     fold_i = (fold_i+1) % fold_K
            item = (line, big_tumor_amount, small_tumor_amount)
            fold_lines[fold_i].append(item)
            fold_tumor_amount[fold_i] += big_tumor_amount
            max_tumor_num = max(max_tumor_num, fold_tumor_amount[fold_i])
            max_tumor_num = max_tumor_num




    for i in range(fold_K):
        print(" fold_", i, "small tumor total:", fold_tumor_amount[i])
        print("fold_", i, "small tomor amount list:", [_[2] for _ in fold_lines[i]])


    max_tumor_num = 0
    fold_tumor_amount = [0] * fold_K
    for i in range(fold_K):
        big_tumor_total = sum([_[1] for _ in fold_lines[i]])
        fold_tumor_amount[i] = big_tumor_total
        print(" fold_", i, "big tumor total:", big_tumor_total)
        print(" fold_", i, "small tumor check:", sum([_[1] for _ in fold_lines[i]]))
        print("fold_", i, "big tomor amount list:", [_[1] for _ in fold_lines[i]])
        max_tumor_num = max(max_tumor_num, big_tumor_total)

    print("done")

