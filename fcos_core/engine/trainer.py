# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger
import numpy as np
from time import gmtime, strftime
from tqdm import tqdm
from TBLogger import TBLogger
from .evaluator import Evaluator
def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    tblogger
):
    logger = logging.getLogger("fcos_core.trainer")
    handler = logging.FileHandler("training_log_{}.log".format(strftime("%Y_%m_%d_%H_%M_%S", gmtime())))
    logger.addHandler(handler)
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    if tblogger:
        tblogger.step = (start_iter // 20) +1
    for iteration, batch_data in tqdm(enumerate(data_loader, start_iter)):
        images = batch_data[0]
        targets = batch_data[1]
        img_names = batch_data[3]

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        #from visualize import visualize
        #visualize(images.tensors[0].cpu().numpy()[0][:, :30, :].astype(np.uint8))
        targets = [target.to(device) for target in targets]
        loss_dict = model(images.tensors[0], targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        if loss_dict_reduced['loss_reg']==0:
            iteration-=1
            print("warning loss_reg==0 skip backward ")
            continue
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            #log to tensorboard
            tblogger.write_log('loss', {'tr_total': losses_reduced,
                'tr_cls': loss_dict_reduced['loss_cls'],
                'tr_reg': loss_dict_reduced['loss_reg'],
                'tr_centerness': loss_dict_reduced['loss_centerness']})
            tblogger.write_log('lr', {'tr': optimizer.param_groups[0]["lr"]})
            #tblogger.write_log('loss', {'val': np.mean(loss_valid)})
            #tblogger.write_log('dice', {'val': mean_dsc})
            tblogger.Step()

            #log to console and local file
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )




def do_evaluate(
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
):
    print("Start Evaluate {} items".format(len(data_loader)))
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    img_size=(640, 160, 640)
    evaluator = Evaluator(model, showatt=False, pred_result_path='debug_evaluate', box_top_k=500, val_shape=img_size)
    evaluator.clear_predict_file()
    start_iter = 0
    loss_avg = []
    with torch.no_grad():
        for iteration, batch_data in tqdm(enumerate(data_loader, start_iter)):
            images = batch_data[0]
            targets = batch_data[1]
            img_names = batch_data[3]

            # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
            if not pytorch_1_1_0_or_later:
                scheduler.step()

            images = images.to(device)
            targets = [target.to(device) for target in targets]
            model.train()
            loss_dict = model(images.tensors[0], targets)
            losses = sum(loss for loss in loss_dict.values())

            loss_avg.append({k:loss_dict[k].detach().item() for k in loss_dict})
            model.eval()

            assert len(images.tensors)==1, 'batch_size>1 is not tested'
            bboxes_prd = evaluator.get_bbox(images.tensors[0], multi_test=False, flip_test=False)
            if len(bboxes_prd) > 0:
                bboxes_prd[:, :6] = (bboxes_prd[:, :6] / images.tensors[0].size(1)) * cfg['INPUT']['TEST_IMG_BBOX_ORIGINAL_SIZE']
            #xyz to zyx
            bboxes_prd_zyx = bboxes_prd+0.0
            bboxes_prd_zyx[..., 0] = bboxes_prd[..., 2]
            bboxes_prd_zyx[..., 3] = bboxes_prd[..., 5]
            bboxes_prd_zyx[..., 2] = bboxes_prd[..., 0]
            bboxes_prd_zyx[..., 5] = bboxes_prd[..., 3]
            print(" len (bboxes_prd_zyx) : ", len(bboxes_prd_zyx))
            evaluator.store_bbox(img_names[0], bboxes_prd_zyx)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if loss_dict_reduced['loss_reg']==0:
                print("warning loss_reg==0 ! ")
                continue
            optimizer.zero_grad()


            if pytorch_1_1_0_or_later:
                scheduler.step()

            print("mean loss_cls", np.mean([_['loss_cls'] for _ in loss_avg]))

            print("mean loss_reg", np.mean([_['loss_reg'] for _ in loss_avg]))
            print("mean loss_centerness", np.mean([_['loss_centerness'] for _ in loss_avg]))

    print("mean loss_cls", np.mean([_['loss_cls'] for _ in loss_avg]))

    print("mean loss_reg", np.mean([_['loss_reg'] for _ in loss_avg]))
    print("mean loss_centerness", np.mean([_['loss_centerness'] for _ in loss_avg]))
    print("evaluate ", len(data_loader),"/", len(loss_avg)," items ")


