"""
Copyright (C) 2019 NVIDIA Corporation.
All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time
import torch
import numpy as np

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

import scipy.misc

from . import util
from .util import tensor2im, tensor2label

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.win_size = opt.display_winsize
        self.name = opt.name

        # TensorBoard logging (optional)
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        # Instead of creating HTML directories, we just ensure a folder for saving images
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'saved_images')
        util.mkdirs([self.save_dir])

        # Create a log file
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals, epoch, step):
        """
        1. Convert visuals to NumPy arrays.
        2. (Optional) log them to TensorBoard if self.tf_log is True.
        3. Save only 'input', 'label', and 'generated' images to disk.
        """
        # 1) Convert Tensors -> NumPy
        visuals = self.convert_visuals_to_numpy(visuals)

        # 2) (Optional) log to TensorBoard
        if self.tf_log:
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Only proceed if it's a Tensor or ndarray
                if not isinstance(image_numpy, (torch.Tensor, np.ndarray)):
                    continue

                # Convert multi-batch images to single
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]

                # Encode image and add to summary
                try:
                    s = StringIO()
                except:
                    s = BytesIO()

                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                img_sum = self.tf.Summary.Image(
                    encoded_image_string=s.getvalue(),
                    height=image_numpy.shape[0],
                    width=image_numpy.shape[1]
                )
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        # 3) Save specific images
        keys_to_save = ['input', 'label', 'generated']  # We only want to save these three keys

        for key in keys_to_save:
            # If the key (e.g., "generated") doesn't exist in visuals, skip
            if key not in visuals:
                continue

            img = visuals[key]

            # Debugging print (optional):
            print(f"{key} type: {type(img)}")

            # If it's a list, save each item
            if isinstance(img, list):
                for i, single_img in enumerate(img):
                    if len(single_img.shape) >= 4:
                        single_img = single_img[0]
                    img_path = os.path.join(
                        self.save_dir,
                        f"epoch{epoch:03d}_iter{step:03d}_{key}_{i}.png"
                    )
                    util.save_image(single_img, img_path)

            # If it's a dictionary, iterate its items
            elif isinstance(img, dict):
                for subkey, sub_img in img.items():
                    # If shape is (B, H, W, C), pick the first in batch
                    if len(sub_img.shape) >= 4:
                        sub_img = sub_img[0]
                    img_path = os.path.join(
                        self.save_dir,
                        f"epoch{epoch:03d}_iter{step:03d}_{key}_{subkey}.png"
                    )
                    util.save_image(sub_img, img_path)

            # Otherwise, handle a single image (tensor/ndarray)
            else:
                if len(img.shape) >= 4:
                    img = img[0]
                img_path = os.path.join(
                    self.save_dir,
                    f"epoch{epoch:03d}_iter{step:03d}_{key}.png"
                )
                util.save_image(img, img_path)

    def plot_current_errors(self, errors, step):
        """
        If using TensorBoard logging, plot the loss values as scalars.
        """
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                summary = self.tf.Summary(value=[
                    self.tf.Summary.Value(tag=tag, simple_value=value)
                ])
                self.writer.add_summary(summary, step)

    def print_current_errors(self, epoch, i, errors, t):
        """
        Print the current errors to stdout and also save into a log file.
        """
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)
        print(message)

        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        """
        Recursively convert all Tensors in `visuals` dict into NumPy arrays.
        For 'input_label' (if label_nc > 0), use `tensor2label`; otherwise, `tensor2im`.
        """
        new_visuals = {}
        for key, val in visuals.items():
            if isinstance(val, str):
                new_visuals[key] = val
                continue

            if isinstance(val, dict):
                new_visuals[key] = self.convert_visuals_to_numpy(val)
                continue

            if isinstance(val, list):
                processed_list = []
                for item in val:
                    if isinstance(item, torch.Tensor):
                        processed_list.append(self.convert_single_tensor(key, item))
                    elif isinstance(item, dict):
                        processed_list.append(self.convert_visuals_to_numpy(item))
                    else:
                        processed_list.append(item)
                new_visuals[key] = processed_list
                continue

            if isinstance(val, torch.Tensor):
                new_visuals[key] = self.convert_single_tensor(key, val)
                continue

            if isinstance(val, np.ndarray):
                new_visuals[key] = val
            else:
                new_visuals[key] = val

        return new_visuals

    def convert_single_tensor(self, key, tensor):
        """
        Convert a single torch Tensor to a NumPy image.
        If it's 'input_label' and self.opt.label_nc != 0, use tensor2label.
        Otherwise, use tensor2im.
        """
        if key == 'input_label' and self.opt.label_nc != 0:
            return tensor2label(tensor, self.opt.label_nc + 2)
        else:
            return tensor2im(tensor)
