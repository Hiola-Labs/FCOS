# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import logging
import os
from fcos_core.config import cfg
import torch
from fcos_core.data import make_data_loader
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train, do_evaluate
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir
from data.abus_data import AbusNpyFormat
from TBLogger import TBLogger
from froc import calculate_FROC
import numpy as np
def train(cfg, local_rank, distributed, tblogger, do_test=False):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    use_fake_dataloader = 0

    data_loader = make_data_loader(
        cfg,
        is_train=False,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    if do_test:
        do_evaluate(
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            tblogger,
            cfg
        )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def froc_test(cfg, args, tblogger, weight_file, fold_num):
    logger = logging.getLogger("fcos_core")

    cfg.defrost()
    cfg.MODEL.WEIGHT = weight_file
    cfg.DATASETS.ABUS_CRX_FOLD_NUM = fold_num
    cfg.freeze()

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = build_detection_model(cfg)
    model = train(cfg, args.local_rank, args.distributed, tblogger, do_test=True)

    npy_dir = 'debug_evaluate_' + weight_file.replace(".", "_")
    npy_format = npy_dir + '/{}_0.npy'
    data_root = 'datasets/abus'
    area_small, area_big, plt = calculate_FROC(data_root, npy_dir, npy_format, size_threshold=0, th_step=0.003, logger=logger)
    plt.savefig('froc_test.png')
    with open('froc_test_log', 'a+') as ff:
        msg = '[\'{}\', {}, {}]\n'.format(cfg.MODEL.WEIGHT, area_small, area_big)
        ff.write(msg)
        ff.close()
    logger.info("evaluation result:" + msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument('--logdir', nargs='?', type=str, default=None,
        help='Path to saved tensorboard log')
    parser.add_argument('--exp_name', nargs='?', type=str, default=None,
        help='experiment name to saved tensorboard log')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    tblogger = TBLogger(args.logdir, args.exp_name)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)


    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger("fcos_core", output_dir, get_rank())
    if output_dir:
        mkdir(output_dir)

    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    """

    a = '/home/lab402/p08922003/FCOS/froc_Fd1_model_0025000_Size_filter0_debug.npy'
    b = '/home/lab402/p08922003/FCOS/froc_Fd1_model_0025000_Size_filter0.npy'
    c = '/home/lab402/p08922003/FCOS/froc_Fd1_model_0025000_Size_filter20.npy'
    a, b, c = np.load(a), np.load(b), np.load(c)
    for i in range(79):
        print("{} {} {} {}".format(i, a[..., 7][i], b[..., 7][i], c[..., 7][i]))

    """

    for fold_k in ['1', '3', ]:
        iter_list = ["{0:07d}".format(i) for i in range(0, 80000, 2500)]

        weight_list = ['trainlog/fcos_imprv_R_50_FPN_1x_ABUS_640All_Ch64_128_Fd' + fold_k + '/model_' + iter_num + '.pth'
                        for iter_num in iter_list]
        for wf in weight_list:
            if os.path.exists(wf):
                froc_test(cfg, args, tblogger, wf, fold_num=int(fold_k))

def fast_FP_sen_calculation(bboxes):
    bboxes = bboxes_sort_by_score_desc(bboxes)
    bboxes = tag_bboxes_is_TP_or_FP_or_FN(bboxes)
    TP = 0
    FP = 0
    FN = 0
    #from score threshold high to low
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        score_threshold = bbox['score']
        if bbox is TP:
            TP+=1
        elif bbox is FP:
            FP+=1
        elif bbox is FN:
            FN+=1
        sensitivity = 0
        precision = 0

    #some threshold is FP free, only TP increases
    result = get_max_sensitivity_group_by_FP(result)

    return 0