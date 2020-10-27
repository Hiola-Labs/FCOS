import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from utils_ABUS.postprocess import centroid_distance, eval_precision_recall_by_dist
from utils_ABUS.misc import draw_full, build_threshold, AUC

def check_boundary(ct):
    y = (ct[1] > 130 or ct[1] < 5)
    z = (ct[0] > 600 or ct[0] < 40)
    x = (ct[2] > 600 or ct[2] < 40)
    return y or (z and x)


def check_size(axis, size):
    return axis[0]*axis[1]*axis[2] > size

def interpolate_FROC_data(froc_x, froc_y, max_fp):
        y_interpolate = 0
        take_i = 0
        for i in range(len(froc_x)):
            FP = froc_x[i]
            if FP<=max_fp:
                take_i = i
                x1 = FP
                y1 = froc_y[i]
                if i>0:
                    x2 = froc_x[i-1]
                    y2 = froc_y[i-1]

                    x_interpolate = max_fp
                    y_interpolate = (y1 * (x2-x_interpolate) + y2 * (x_interpolate-x1)) / (x2-x1)
                else:
                    #if no data point for FP > 8
                    #use sensitivity at FP = FP_small
                    y_interpolate = y1
                print("take i = ", i, " FP = ", int(FP*100)/100)
                print("interpolate sen = ", y_interpolate, " for FP=", max_fp)
                break
            else:
                print("skip i = ", i, " FP = ", int(FP*100)/100)
        froc_x = froc_x[take_i:]
        froc_y = froc_y[take_i:]

        if not froc_x[0]==8:
            froc_x = np.insert(froc_x, 0, 8)
            froc_y = np.insert(froc_y, 0, y_interpolate)
        return froc_x, froc_y
def froc_take_max(froc_x, froc_y):
    froc_x_tmp = []
    froc_y_tmp = []
    for i in range(len(froc_x)):
        if i==0 or froc_x_tmp[-1] > froc_x[i]:
            froc_x_tmp.append(froc_x[i])
            froc_y_tmp.append(froc_y[i])
    froc_x = np.array(froc_x_tmp)
    froc_y = np.array(froc_y_tmp)
    return froc_x, froc_y

def calculate_FROC(root, npy_dir, npy_format, size_threshold=0, th_step=0.05, logger=None):
    #size_threshold is 20 in thesis
    num_npy = os.listdir(npy_dir) # dir is your directory path
    total_pass = len(num_npy)
    all_thre=build_threshold(th_step)
    PERF_per_thre=[]
    PERF_per_thre_s=[]
    true_num, true_small_num = 0, 0
    out_boxes_cache = {}
    true_box_cache = {}

    logger.info('size_threshold = {:.3f}'.format(size_threshold))
    with open(os.path.join(root, 'annotations/rand_all.txt'), 'r') as f:
        rand_all_lines = f.read().splitlines()
    for i, score_hit_thre in enumerate(all_thre):
        lines = [s+'' for s in rand_all_lines]

        logger.info('Use threshold: {:.3f}'.format(score_hit_thre))

        TP_table, FP_table, FN_table, \
        TP_table_IOU_1, FP_table_IOU_1, FN_table_IOU_1, \
        pred_num, pred_small_num, file_table, iou_table \
        = [], [], [], [], [], [], [], [], [], []
        # , score_table, mean_score_table, std_score_table
        TP_table_s, FP_table_s, FN_table_s, \
        TP_table_IOU_1_s, FP_table_IOU_1_s, FN_table_IOU_1_s = [], [], [], [], [], []

        current_pass = 0
        for line in lines:
            line = line.split(',', 4)
            # Always use 640,160,640 to compute iou
            size = (640,160,640)
            scale = (size[0]/int(line[1]),size[1]/int(line[2]),size[2]/int(line[3]))
            pred_npy = npy_format.format(line[0].replace('/', '_'))
            if not os.path.exists(pred_npy):
                continue

            current_pass += 1
            logger.info('Processing {}/{} data...'.format(current_pass, total_pass))
            if current_pass == total_pass:
                logger.info("\n")
            if not pred_npy in true_box_cache:
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
                true_box_cache[pred_npy] = (true_box, true_box_s)

            (true_box, true_box_s) = true_box_cache[pred_npy]

            if i == 0:
                true_num += len(true_box)
                true_small_num += len(true_box_s)

            file_name = line[0]
            file_table.append(file_name)

            ##########################################
            if not pred_npy in out_boxes_cache:
                out_boxes = []
                box_list = np.load(pred_npy)
                for bx in box_list:
                    axis = [0,0,0]
                    axis[0] = (bx[3] - bx[0]) / scale[0] / 4
                    axis[1] = (bx[4] - bx[1]) / scale[1] / 4
                    axis[2] = (bx[5] - bx[2]) / scale[2] / 4
                    ct = [0,0,0]
                    ct[0] = (bx[3] + bx[0]) / 2
                    ct[1] = (bx[4] + bx[1]) / 2
                    ct[2] = (bx[5] + bx[2]) / 2
                    if bx[6] >= all_thre[0] and (not check_boundary(ct)) and check_size(axis, size_threshold):
                        out_boxes.append(list(bx))
                out_boxes_cache[pred_npy] = np.array(out_boxes)

            box_list = out_boxes_cache[pred_npy].copy()
            mask=box_list[..., 6] >= score_hit_thre
            out_boxes = box_list[mask]

            pred_num.append(len(out_boxes))

            TP, FP, FN, hits_index, hits_iou, hits_score, TP_by_size_15 = eval_precision_recall_by_dist(
                out_boxes, true_box, 15, scale)

            TP_IOU_1, FP_IOU_1, FN_IOU_1, hits_index_IOU_1, hits_iou_IOU_1, hits_score_IOU_1, TP_by_size_10 = eval_precision_recall_by_dist(
                out_boxes, true_box, 10, scale)

            if FN_IOU_1 > 0 and i is 0:
                logger.info("FN = {}: {}".format(FN_IOU_1, line[0]))

            TP_table.append(TP)
            FP_table.append(FP)
            FN_table.append(FN)

            TP_table_IOU_1.append(TP_IOU_1)
            FP_table_IOU_1.append(FP_IOU_1)
            FN_table_IOU_1.append(FN_IOU_1)

            ##########################################
            # Small tumor

            # TP_s, FP_s, FN_s, hits_index_s, hits_iou_s, hits_score_s = eval_precision_recall_by_dist(
            #     out_boxes, true_box_s, 15, scale)

            # TP_IOU_1_s, FP_IOU_1_s, FN_IOU_1_s, hits_index_IOU_1_s, hits_iou_IOU_1_s, hits_score_IOU_1_s = eval_precision_recall_by_dist(
            #     out_boxes, true_box_s, 10, scale)

            # TP_table_s.append(TP_s)
            # FP_table_s.append(FP_s)
            # FN_table_s.append(FN_s)

            # TP_table_IOU_1_s.append(TP_IOU_1_s)
            # FP_table_IOU_1_s.append(FP_IOU_1_s)
            # FN_table_IOU_1_s.append(FN_IOU_1_s)

        TP_table_sum = np.array(TP_table)
        FP_table_sum = np.array(FP_table)
        FN_table_sum = np.array(FN_table)

        TP_table_sum_IOU_1 = np.array(TP_table_IOU_1)
        FP_table_sum_IOU_1 = np.array(FP_table_IOU_1)
        FN_table_sum_IOU_1 = np.array(FN_table_IOU_1)

        # TP_table_sum_s = np.array(TP_table_s)
        # FP_table_sum_s = np.array(FP_table_s)
        # FN_table_sum_s = np.array(FN_table_s)

        # TP_table_sum_IOU_1_s = np.array(TP_table_IOU_1_s)
        # FP_table_sum_IOU_1_s = np.array(FP_table_IOU_1_s)
        # FN_table_sum_IOU_1_s = np.array(FN_table_IOU_1_s)

        sum_TP, sum_FP, sum_FN = TP_table_sum.sum(), FP_table_sum.sum(), FN_table_sum.sum()
        sensitivity = sum_TP/(sum_TP+sum_FN+1e-10)
        precision = sum_TP/(sum_TP+sum_FP+1e-10)

        sum_TP_IOU_1, sum_FP_IOU_1, sum_FN_IOU_1 = TP_table_sum_IOU_1.sum(), FP_table_sum_IOU_1.sum(), FN_table_sum_IOU_1.sum()
        sensitivity_IOU_1 = sum_TP_IOU_1/(sum_TP_IOU_1+sum_FN_IOU_1+1e-10)
        precision_IOU_1 = sum_TP_IOU_1/(sum_TP_IOU_1+sum_FP_IOU_1+1e-10)

        # sum_TP_s, sum_FP_s, sum_FN_s = TP_table_sum_s.sum(), FP_table_sum_s.sum(), FN_table_sum_s.sum()
        # sensitivity_s = sum_TP_s/(sum_TP_s+sum_FN_s+1e-10)
        # precision_s = sum_TP_s/(sum_TP_s+sum_FP_s+1e-10)

        # sum_TP_IOU_1_s, sum_FP_IOU_1_s, sum_FN_IOU_1_s = TP_table_sum_IOU_1_s.sum(), FP_table_sum_IOU_1_s.sum(), FN_table_sum_IOU_1_s.sum()
        # sensitivity_IOU_1_s = sum_TP_IOU_1_s/(sum_TP_IOU_1_s+sum_FN_IOU_1_s+1e-10)
        # precision_IOU_1_s = sum_TP_IOU_1_s/(sum_TP_IOU_1_s+sum_FP_IOU_1_s+1e-10)

        if sensitivity > 0.125:
            PERF_per_thre.append([
                score_hit_thre,
                total_pass,
                sensitivity, #small FROC Y
                precision,
                sum_FP/total_pass, #small FROC X
                sensitivity_IOU_1, #all FROC Y
                precision_IOU_1,
                sum_FP_IOU_1/total_pass], #all FROC x
                )

        # if sensitivity_s > 0.125:
        #     PERF_per_thre_s.append([
        #         score_hit_thre,
        #         total_pass,
        #         sensitivity_s,
        #         precision_s,
        #         sum_FP_s/total_pass,
        #         sensitivity_IOU_1_s,
        #         precision_IOU_1_s,
        #         sum_FP_IOU_1_s/total_pass])

        logger.info('Threshold:{:.3f}'.format(score_hit_thre))
        logger.info('Dist of Center < 15mm Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}'.format(sensitivity, precision, sum_FP/total_pass))
        logger.info('Dist of Center < 10mm Sen:{:.3f}, Pre:{:.3f}, FP per pass:{:.3f}'.format(sensitivity_IOU_1, precision_IOU_1, sum_FP_IOU_1/total_pass))
        logger.info('\n')


    logger.info('Small/All tumors: {}/{}'.format(true_small_num, true_num))

    data = np.array(PERF_per_thre)
    data_s = np.array(PERF_per_thre_s)

    plt.rc('font',family='Times New Roman', weight='bold')
    area_small, area_big = 0, 0
    if len(data) == 0:
        logger.info('Inference result is empty.')
        area = 0
    else:
        froc_x, froc_y = interpolate_FROC_data(data[..., 7], data[..., 5], max_fp=8)
        froc_x, froc_y = froc_take_max(froc_x, froc_y)
        draw_full(froc_x, froc_y, '#FF6D6C', 'D < 10 mm', '-.', 1, True)
        area_small = AUC(froc_x, froc_y, normalize=True)

        froc_x, froc_y = interpolate_FROC_data(data[..., 4], data[..., 2], max_fp=8)
        froc_x, froc_y = froc_take_max(froc_x, froc_y)
        draw_full(froc_x, froc_y, '#FF0000', 'D < 15 mm', '-', 1, True)
        area_big = AUC(froc_x, froc_y, normalize=True)


    #np.save('result/model_0027500_Fd1.npy', data)
    # if len(data_s) == 0:
    #     print('Inference result for small is empty.')
    # else:
    #     draw_full(data_s[..., 7], data_s[..., 5], '#6D6CFF', 'Dist < 15', ':', 1)
    #     draw_full(data_s[..., 4], data_s[..., 2], '#0000FF', 'Dist < 10', '-', 1)

    # axes = plt.gca()
    # axes.axis([0, 10, 0.5, 1])
    # axes.set_aspect('auto')
    plt.xlim(1, 10)
    x_tick = np.arange(0, 10, 2)
    plt.xticks(x_tick)
    plt.ylim(0.5, 1)
    y_tick = np.arange(0.5, 1, 0.05)
    y_tick = np.append(y_tick, 0.98)
    y_tick = np.sort(y_tick)
    plt.yticks(y_tick)
    plt.legend(loc='lower right')
    # plt.grid(b=True, which='major', axis='x')
    plt.ylabel('Sensitivity')
    plt.xlabel('False Positive Per Pass')
    return area_small, area_big, plt

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--threshold', type=float, default=0,
        help='Threshold for size filtering.'
    )
    parser.add_argument(
        '--root', '-r', type=str, required=True,
        help='folder path for data/sys_ucc/'
    )
    return parser.parse_args()


if __name__ == '__main__':
    root = 'datasets/abus'

    #npy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results/prediction/')
    #npy_format = npy_dir + '{}'

    npy_dir = '/data/Hiola/YOLOv4-pytorch/data/pred_result/evaluate/'
    npy_format = npy_dir + '{}_0.npy'
    args = _parse_args()
    root = args.root
    main(args)
    area_small, area_big, plt = calculate_FROC(root, npy_dir, npy_format)
