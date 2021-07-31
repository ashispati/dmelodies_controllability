import os
import json
import time
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import music21
from typing import Tuple
from tensorboardX import SummaryWriter

from src.utils.helpers import to_numpy

from src.utils.trainer import Trainer
from src.dmelodiesvae.dmelodies_vae_trainer import LATENT_ATTRIBUTES, DMelodiesVAETrainer
from src.dmelodiesvae.s2_vae import S2VAE
from src.utils.helpers import to_cuda_variable_long, to_cuda_variable, to_numpy
from src.utils.evaluation import *


class S2VAETrainer(DMelodiesVAETrainer):
    def __init__(
            self,
            dataset,
            model: S2VAE,
            model_type='s2-VAE',
            lr=1e-4,
            # reg_type: Tuple[str] = None,
            beta=0.001,
            gamma=1.0,
            delta=10.0,
            capacity=0.0,
            device=0,
            rand=0,
    ):
        super(S2VAETrainer, self).__init__(
            dataset, model, model_type, lr, beta, gamma, delta, capacity, device, rand
        )
        self.exp_rate = np.log(1 + self.beta) / self.num_iterations
        self.start_beta = 0.0
        self.cur_beta = self.start_beta
        self.start_capacity = self.capacity
        self.cur_capacity = self.capacity
        self.gamma = gamma
        self.trainer_config = f'_{self.model_type}_b_{self.beta}_c_{self.capacity}_'
        self.trainer_config += f'g_{self.gamma}_'
        self.trainer_config += f'r_{self.rand_seed}_'
        self.model.update_trainer_config(self.trainer_config)

    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedulegit sa
        :param batch_num: int
        :param train: bool, True is backward pass is to be performed
        :return: scalar loss value, scalar accuracy value
        """
        if self.cur_epoch_num != epoch_num:
            flag = True
            self.cur_epoch_num = epoch_num
        else:
            flag = False

        # extract data
        score, latent_attributes = batch

        # perform forward pass of src
        weights, samples, z_dist, prior_dist, z_tilde, z_prior = self.model.VAE(
            measure_score_tensor=score,
            measure_metadata_tensor=None,
            train=train
        )

        # compute reconstruction loss
        recons_loss = self.reconstruction_loss(x=score, x_recons=weights)

        # compute KLD
        dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.cur_beta, c=0.0)
        dist_loss = torch.nn.functional.relu(dist_loss - self.cur_capacity)

        # add loses
        loss = recons_loss + dist_loss

        # compute regularization loss for interp VAE
        metadata = self.normalize_latent_attributes(latent_attributes)
        reg_loss = self.compute_reg_loss(z_tilde, metadata, self.gamma)

        # add regularization loss
        loss += reg_loss

        # log values
        if flag:
            self.writer.add_scalar(
                'loss_split/recons_loss', recons_loss.item(), epoch_num
            )
            self.writer.add_scalar(
                'loss_split/dist_loss', dist_loss.item(), epoch_num
            )
            self.writer.add_scalar(
                'loss_split/reg_loss', (reg_loss / self.gamma).item(), epoch_num
            )
            self.writer.add_scalar(
                'params/beta', self.cur_beta, epoch_num
            )
            self.writer.add_scalar(
                'params/capacity', self.cur_capacity, epoch_num
            )

        # compute accuracy
        accuracy = self.mean_accuracy(
            weights=weights, targets=score
        )

        return loss, accuracy

    def compute_reg_loss(self, z, labels, gamma):
        loss = 0
        for i, attr in enumerate(self.attr_dict.keys()):
            dim = self.attr_dict[attr]
            target = labels[:, dim]
            inp = z[:, i]
            loss += torch.nn.functional.binary_cross_entropy(torch.sigmoid(inp), target)
        return gamma * loss

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        if epoch_num > self.warm_up_epochs:
            if self.anneal_iterations < self.num_iterations:
                self.cur_beta = -1.0 + np.exp(self.exp_rate * self.anneal_iterations)
            self.anneal_iterations += 1

    def decode_latent_codes(self, latent_codes):
        batch_size = latent_codes.size(0)
        dummy_score_tensor = to_cuda_variable(
            torch.zeros(batch_size, 16)
        )
        _, tensor_score = self.model.VAE.decoder(latent_codes, dummy_score_tensor, False)
        score = self.dataset.tensor_to_m21score(tensor_score)
        return score, tensor_score
