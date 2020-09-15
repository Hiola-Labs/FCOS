import os
from torch.utils.tensorboard import SummaryWriter
import sys
import abc
import math
import yaml
import torch

class TBLogger():

    def __init__(self, logdir, exp_name):
            TB_FLUSH_FREQ = 180
            # Logger settings
            self.logdir = os.path.join(logdir, exp_name)
            self.log = SummaryWriter(
                self.logdir, flush_secs=TB_FLUSH_FREQ)
            self.step = 1

    def write_log(self, log_name, log_dict):
            '''
            Write log to TensorBoard
                log_name  - <str> Name of tensorboard variable
                log_value - <dict>/<array> Value of variable (e.g. dict of losses), passed if value = None
            '''

            if type(log_dict) is dict:
                log_dict = {key: val for key, val in log_dict.items() if (
                    val is not None and not math.isnan(val))}
            if log_dict is None:
                pass
            elif len(log_dict) > 0:
                if 'align' in log_name or 'spec' in log_name:
                    img, form = log_dict
                    self.log.add_image(
                        log_name, img, global_step=self.step, dataformats=form)
                elif 'text' in log_name or 'hyp' in log_name:
                    self.log.add_text(log_name, log_dict, self.step)
                else:
                    self.log.add_scalars(log_name, log_dict, self.step)
    def SetStep(self, step):
        self.step=step

    def Step(self):
        self.step+=1
