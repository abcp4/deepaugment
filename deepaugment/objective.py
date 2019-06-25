# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

import pandas as pd
import numpy as np


import sys
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
sys.path.insert(0, dir_of_file)

from augmenter import augment_by_policy
from lib.helpers import log_and_print

import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
#from deepaugment.config import AugmentConfig
import utils
from deepaugment.models.augment_cnn import AugmentCNN


#config = AugmentConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

#logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
#config.print_params(logger.info)



class Objective:
    """Objective class for the controller

    """
    def __init__(self, data, child_model, notebook, config):
        self.data = data
        self.child_model = child_model
        self.opt_samples = config["opt_samples"]
        self.opt_last_n_epochs = config["opt_last_n_epochs"]
        self.notebook = notebook
        self.logging = config["logging"]
        
        logger.info("Logger is set - training start")

        # set default gpu device id
        torch.cuda.set_device(config.gpus[0])

        # set seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        torch.backends.cudnn.benchmark = True

        # get data with meta info
        #input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        #    config.dataset, config.data_path, config.cutout_length, validation=True)
        input_size, input_channels, n_classes, _, _, _ = utils.get_data(
            config.dataset, config.data_path, cutout_length=0, validation=True,validation2 = True)
        self.input_size = input_size
        self.input_channels = input_channels
        self.n_classes = n_classes
        criterion = nn.CrossEntropyLoss().to(device)
        use_aux = config.aux_weight > 0.
        #from evaluate


    def evaluate(self, trial_no, trial_hyperparams):
        """Evaluates objective function

        Trains the child model k times with same augmentation hyperparameters.
        k is determined by the user by `opt_samples` argument.

        Args:
            trial_no (int): no of trial. needed for recording to notebook
            trial_hyperparams (list)
        Returns:
            float: trial-cost = 1 - avg. rewards from samples
        """

        augmented_data = augment_by_policy(
            self.data["X_train"], self.data["y_train"], *trial_hyperparams
        )

        sample_rewards = []
        #pytorch
        layers = 2
        init_channels = 24
        use_aux = True
        epochs = 30
        lr = 0.01
        momentum = 0.995
        weight_decay = 0.995
        genotype = "Genotype(normal=[[('dil_conv_3x3', 0), ('sep_conv_5x5', 1)], [('sep_conv_3x3', 1), ('avg_pool_3x3', 0)],[('dil_conv_3x3', 1), ('dil_conv_3x3', 0)], [('sep_conv_3x3', 3), ('skip_connect', 1)]], normal_concat=range(2, 6), reduce=[[('sep_conv_3x3', 1), ('dil_conv_5x5', 0)], [('skip_connect', 0), ('sep_conv_5x5', 1)], [('sep_conv_5x5', 1),('sep_conv_5x5', 0)], [('max_pool_3x3', 1), ('sep_conv_3x3', 0)]], reduce_concat=range(2, 6))"
        model = AugmentCNN(self.input_size, self.input_channels, init_channels, self.n_classes, layers,
                       use_aux,genotype)
        model = nn.DataParallel(model, device_ids='0').to(device)

        # model size
        mb_params = utils.param_size(model)
        logger.info("Model size = {:.3f} MB".format(mb_params))

        # weights optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum = momentum,weight_decay = weight_decay)
        a  =2/0
        """
        for sample_no in range(1, self.opt_samples + 1):
            self.child_model.load_pre_augment_weights()
            # TRAIN
            history = self.child_model.fit(self.data, augmented_data)
            #
            reward = self.calculate_reward(history)
            sample_rewards.append(reward)
            self.notebook.record(
                trial_no, trial_hyperparams, sample_no, reward, history
            )

        """
        best_top1 = -9999
        for epoch in range(epochs):
            lr_scheduler.step()
            drop_prob = config.drop_path_prob * epoch / config.epochs
            model.module.drop_path_prob(drop_prob)

            # training
            train(train_loader, model, optimizer, criterion, epoch)

            # validation
            cur_step = (epoch+1) * len(train_loader)
            top1 = validate(valid_loader, model, criterion, epoch, cur_step)

            # save
            if best_top1 < top1:
                best_top1 = top1
                is_best = True
            else:
                is_best = False
        print('best_top1:',best_top1)
        #sample_rewards.append(reward)
        #self.notebook.record(
        #    trial_no, trial_hyperparams, sample_no, reward, history
        #)
        #trial_cost = 1 - np.mean(sample_rewards)
        #self.notebook.save()

        log_and_print(
            f"{str(trial_no)}, {str(trial_cost)}, {str(trial_hyperparams)}",
            self.logging,
        )

        #return trial_cost
        return best_top1

    def calculate_reward(self, history):
        """Calculates reward for the history.

        Reward is mean of largest n validation accuracies which are not overfitting.
        n is determined by the user by `opt_last_n_epochs` argument. A validation
        accuracy is considered as overfitting if the training accuracy in the same
        epoch is larger by 0.05

        Args:
            history (dict): dictionary of loss and accuracy
        Returns:
            float: reward
        """
        history_df = pd.DataFrame(history)
        history_df["acc_overfit"] = history_df["acc"] - history_df["val_acc"]
        reward = (
            history_df[history_df["acc_overfit"] <= 0.10]["val_acc"]
            .nlargest(self.opt_last_n_epochs)
            .mean()
        )
        return reward
