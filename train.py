"""
Copyright (C) 2019 NVIDIA Corporation.  
All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import numpy as np
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer  # <--- uses the code above

# Import the new trainer (CascadedPix2PixTrainer)
from trainers.cascaded_pix2pix_trainer import CascadedPix2PixTrainer

# 1) parse options
opt = TrainOptions().parse()

# 2) print options to help debugging
print(' '.join(sys.argv))

# 3) load the dataset
dataloader = data.create_dataloader(opt)

# 4) Create the trainer
trainer = CascadedPix2PixTrainer(opt)

# 5) Create iteration counter
iter_counter = IterationCounter(opt, len(dataloader) * opt.batchSize)

# 6) Create visualization tool
visualizer = Visualizer(opt)

clear_iter = False
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch, clear_iter)
    clear_iter = True

    start_batch_idx = iter_counter.epoch_iter // opt.batchSize

    for i, data_i in enumerate(dataloader):
        if i < start_batch_idx:
            continue
        iter_counter.record_one_iteration()

        # ---------------------
        #  Training Steps
        # ---------------------
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)
        trainer.run_discriminator_one_step(data_i)

        # ---------------------
        #  Logging & Display
        # ---------------------
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(
                epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter
            )
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            # Pack the images to display (and save) in a dict
            visuals = OrderedDict([
                ('input', data_i['label']),             # input to the network
                ('label', data_i['image']),             # ground truth label
                ('generated', trainer.get_latest_generated())  # network output
            ])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        # ---------------------
        #  Saving the Model
        # ---------------------
        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)'
                  % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    # Update LR after each epoch
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    # Save periodically or at the very end
    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d'
              % (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)
        
print('Training was successfully finished.')
#python train.py --name test --dataset_mode oral --no_instance --label_nc 0 --batchSize 2 --dataroot "/media/irfan/New Volume/Dentalverse/code/pix2pix/SPADE/datasets/oral"