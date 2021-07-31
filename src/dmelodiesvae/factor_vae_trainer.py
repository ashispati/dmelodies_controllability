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
from src.dmelodiesvae.factor_vae import FactorVAE
from src.utils.helpers import to_cuda_variable_long, to_cuda_variable, to_numpy
from src.utils.evaluation import *

LATENT_ATTRIBUTES = {
    'tonic': 0,
    'octave': 1,
    'mode': 2,
    'rhythm_bar1': 3,
    'rhythm_bar2': 4,
    'contour': 5
}


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


class FactorVAETrainer(Trainer):
    def __init__(
            self,
            dataset,
            model: FactorVAE,
            lr=1e-4,
            reg_type: Tuple[str] = None,
            beta=0.001,
            capacity=0.0,
            rand=0,
    ):
        super(FactorVAETrainer, self).__init__(dataset, model, lr)
        self.attr_dict = LATENT_ATTRIBUTES
        self.reverse_attr_dict = {
            v: k for k, v in self.attr_dict.items()
        }
        self.metrics = {}
        self.beta = beta
        self.start_beta = 0.0
        self.cur_beta = self.start_beta
        self.capacity = to_cuda_variable(torch.FloatTensor([capacity]))
        self.gamma = 1.0
        self.delta = 0.0
        self.cur_epoch_num = 0
        self.warm_up_epochs = 1
        self.num_iterations = 100000
        self.exp_rate = np.log(1 + self.beta) / self.num_iterations
        self.anneal_iterations = 0
        self.reg_type = reg_type
        self.reg_dim = ()
        self.rand_seed = rand
        torch.manual_seed(self.rand_seed)
        np.random.seed(self.rand_seed)
        self.trainer_config = f'_r_{self.rand_seed}_b_{self.beta}_'
        if capacity != 0.0:
            self.trainer_config += f'c_{capacity}_'
        self.model.update_trainer_config(self.trainer_config)

        # re-instantiate optimizers
        self.optimizer = optim.Adam(model.VAE.parameters(), lr=lr)
        self.D_optim = optim.Adam(model.discriminator.parameters(), lr=lr)

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        if epoch_num > self.warm_up_epochs:
            if self.anneal_iterations < self.num_iterations:
                self.cur_beta = -1.0 + np.exp(self.exp_rate * self.anneal_iterations)
            self.anneal_iterations += 1
    
    # Overload trainer method
    def train_model(self, batch_size, num_epochs, log=False):
        """
        Trains the src
        :param batch_size: int,
        :param num_epochs: int,
        :param log: bool, logs epoch stats for viewing in tensorboard if TRUE
        :return: None
        """
        # set-up log parameters
        if log:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime(
                '%Y-%m-%d_%H:%M:%S'
            )
            # configure tensorboardX summary writer
            self.writer = SummaryWriter(
                logdir=os.path.join('runs/' + self.model.__repr__() + st)
            )

        # get dataloaders
        (generator_train,
         generator_val,
         _) = self.dataset.data_loaders(
            batch_size=batch_size,
            split=(0.70, 0.20)
        )
        print('Num Train Batches: ', len(generator_train))
        print('Num Valid Batches: ', len(generator_val))

        # train epochs
        for epoch_index in range(num_epochs):
            # run training loop on training data
            self.model.train()
            returns = self.loss_and_acc_on_epoch(
                data_loader=generator_train,
                epoch_num=epoch_index,
                train=True
            )
            mean_loss_train, mean_accuracy_train = returns[0]
            mean_D_loss_train, mean_D_accuracy_train = returns[1]

            # run evaluation loop on validation data
            self.model.eval()
            returns = self.loss_and_acc_on_epoch(
                data_loader=generator_val,
                epoch_num=epoch_index,
                train=False
            )
            mean_loss_val, mean_accuracy_val = returns[0]
            mean_D_loss_val, mean_D_accuracy_val = returns[1]

            self.eval_model(
                data_loader=generator_val,
                epoch_num=epoch_index,
            )

            # log parameters
            if log:
                # log value in tensorboardX for visualization
                self.writer.add_scalar('loss/train', mean_loss_train, epoch_index)
                self.writer.add_scalar('loss/valid', mean_loss_val, epoch_index)
                self.writer.add_scalar('acc/train', mean_accuracy_train, epoch_index)
                self.writer.add_scalar('acc/valid', mean_accuracy_val, epoch_index)
                self.writer.add_scalar('loss/D_train', mean_D_loss_train, epoch_index)
                self.writer.add_scalar('loss/D_valid', mean_D_loss_val, epoch_index)
                self.writer.add_scalar('acc/D_train', mean_D_accuracy_train, epoch_index)
                self.writer.add_scalar('acc/D_valid', mean_D_accuracy_val, epoch_index)

            # print epoch stats
            data_element = {
                'epoch_index': epoch_index,
                'num_epochs': num_epochs,
                'mean_loss_train': mean_loss_train,
                'mean_accuracy_train': mean_accuracy_train,
                'mean_loss_val': mean_loss_val,
                'mean_accuracy_val': mean_accuracy_val
            }
            self.print_epoch_stats(**data_element)

            # save src
            self.model.save()

    # overload trainer method
    def loss_and_acc_on_epoch(self, data_loader, epoch_num=None, train=True):
        """
        Computes the loss and accuracy for an epoch
        :param data_loader: torch dataloader object
        :param epoch_num: int, used to change training schedule
        :param train: bool, performs the backward pass and gradient descent if TRUE
        :return: loss values and accuracy percentages
        """
        mean_loss = 0
        mean_accuracy = 0
        mean_D_loss = 0
        mean_D_accuracy = 0
        for batch_num, batch in tqdm(enumerate(data_loader)):
            # update training scheduler
            if train:
                self.update_scheduler(epoch_num)

            # process batch data
            batch_1, batch_2 = self.process_batch_data(batch)

            # zero the gradients
            self.zero_grad()

            # compute loss for batch
            vae_loss, accuracy, D_z = self.loss_and_acc_for_batch(
                batch_1, epoch_num, batch_num, train=train
            )

            # compute backward and step if train
            if train:
                vae_loss.backward(retain_graph=True)
                # self.plot_grad_flow()
                self.step()

            # compute Discriminator loss
            D_loss, D_acc = self.loss_and_acc_for_batch_D(
                batch_2, D_z, epoch_num, batch_num, train=train
            )

            if train:
                self.D_optim.zero_grad()
                D_loss.backward()
                self.D_optim.step()

            # log batch_wise:
            self.writer.add_scalar(
                'batch_wise/vae_loss', vae_loss.item(), self.global_iter
            )
            self.writer.add_scalar(
                'batch_wise/D_loss', D_loss.item(), self.global_iter
            )

            # compute mean loss and accuracy
            mean_loss += to_numpy(vae_loss.mean())
            if accuracy is not None:
                mean_accuracy += to_numpy(accuracy)
            mean_D_loss += to_numpy(D_loss.mean())
            mean_D_accuracy += to_numpy(D_acc.mean())

            if train:
                self.global_iter += 1

        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        mean_D_loss /= len(data_loader)
        mean_D_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        ), (
            mean_D_loss,
            mean_D_accuracy
        )

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        FactorVAE takes in two batches of data. So batch size 
        and epoch number will be double normal experiments
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        
        score_tensor, latent_tensor = batch
        batch_size = score_tensor.shape[0]
        # convert input to torch Variables
        score_tensor, latent_tensor = (
            to_cuda_variable_long(score_tensor.squeeze(1)),
            to_cuda_variable_long(latent_tensor.squeeze(1))
        )
        batch_1 = (score_tensor[:batch_size//2], latent_tensor[:batch_size//2])
        batch_2 = (score_tensor[batch_size//2:], latent_tensor[batch_size//2:])
        return (batch_1, batch_2)

    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        """
        Computes the VAE loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
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
        weights, samples, z_dist, prior_dist, z_tilde, z_prior = self.model(
            measure_score_tensor=score,
            measure_metadata_tensor=None,
            train=train
        )

        # compute reconstruction loss
        recons_loss = self.reconstruction_loss(x=score, x_recons=weights)

        # compute KLD loss
        dist_loss = self.compute_kld_loss(z_dist, prior_dist, self.cur_beta, self.capacity)
        # dist_loss = torch.nn.functional.relu(dist_loss - self.capacity)

        # compute TC loss
        D_z = self.model.discriminator(z_tilde)
        tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()

        # add loses
        loss = recons_loss + dist_loss + self.gamma*tc_loss

        # compute and add regularization loss if needed
        # log values
        if flag:
            self.writer.add_scalar(
                'loss_split/recons_loss', recons_loss.item(), epoch_num
            )
            self.writer.add_scalar(
                'loss_split/dist_loss', dist_loss.item(), epoch_num
            )
            self.writer.add_scalar(
                'params/beta', self.cur_beta, epoch_num
            )
            self.writer.add_scalar(
                'loss_split/vae_tc_loss', tc_loss.item(), epoch_num
            )

        # compute accuracy
        accuracy = self.mean_accuracy(
            weights=weights, targets=score
        )

        return loss, accuracy, D_z

    def loss_and_acc_for_batch_D(self, batch, D_z, epoch_num=None, batch_num=None, train=True):
        """
        Computes the Discriminator loss and accuracy for the batch
        Must return (loss, accuracy) as a tuple, accuracy can be None
        :param batch: torch Variable,
        :param epoch_num: int, used to change training schedule
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

        batch_size = score.shape[0]

        ones = torch.ones(batch_size, dtype=torch.long).cuda()
        zeros = torch.zeros(batch_size, dtype=torch.long).cuda()

        z_tilde = self.model.VAE.encode(measure_score_tensor=score)
            
        z_pperm = permute_dims(z_tilde).detach()
        D_z_pperm = self.model.discriminator(z_pperm)
        D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

        if flag:
            self.writer.add_scalar(
                'loss_split/d_tc_loss', D_tc_loss.item(), epoch_num
            )

        soft_D_z = F.softmax(D_z, 1)[:, :1].detach()
        soft_D_z_pperm = F.softmax(D_z_pperm, 1)[:, :1].detach()
        D_acc = ((soft_D_z >= 0.5).sum() + (soft_D_z_pperm < 0.5).sum()).float()
        D_acc /= 2*batch_size

        return D_tc_loss, D_acc

    def compute_representations(self, data_loader, num_batches=None):
        latent_codes = []
        attributes = []
        if num_batches is None:
            num_batches = 200
        for batch_id, batch in tqdm(enumerate(data_loader)):
            inputs, latent_attributes = self.process_batch_data(batch)
            _, _, _, _, z_tilde, _ = self.model(inputs, None, train=False)
            latent_codes.append(to_numpy(z_tilde.cpu()))
            attributes.append(to_numpy(latent_attributes))
            if batch_id == num_batches:
                break
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)
        attr_list = [
            attr for attr in self.attr_dict.keys()
        ]
        return latent_codes, attributes, attr_list

    def eval_model(self, data_loader, epoch_num=0):
        if self.writer is not None:
            # evaluation takes time due to computation of metrics
            # so we skip it during training epochs
            metrics = None
        else:
            metrics = self.compute_eval_metrics()
        return metrics

    def compute_eval_metrics(self):
        """Returns the saved results as dict or computes them"""
        results_fp = os.path.join(
            os.path.dirname(self.model.filepath),
            'results_dict.json'
        )
        batch_size = 256
        _, gen_val, gen_test = self.dataset.data_loaders(batch_size=batch_size, split=(0.70, 0.20))
        latent_codes, attributes, attr_list = self.compute_representations(gen_test)
        self.metrics.update(compute_mig(latent_codes, attributes))
        self.metrics.update(compute_modularity(latent_codes, attributes))
        self.metrics.update(compute_sap_score(latent_codes, attributes))
        self.metrics.update(self.test_model(batch_size=batch_size))
        with open(results_fp, 'w') as outfile:
            json.dump(self.metrics, outfile, indent=2)
        return self.metrics

    def test_model(self, batch_size):
        _, gen_val, gen_test = self.dataset.data_loaders(batch_size)
        mean_loss_test, mean_accuracy_test = self.loss_and_acc_test(gen_test)
        print('Test Epoch:')
        print(
            '\tTest Loss: ', mean_loss_test, '\n'
            '\tTest Accuracy: ', mean_accuracy_test * 100
        )
        return {
            "test_loss": mean_loss_test,
            "test_acc": mean_accuracy_test,
        }

    def loss_and_acc_test(self, data_loader):
        mean_loss = 0
        mean_accuracy = 0

        for sample_id, batch in tqdm(enumerate(data_loader)):
            inputs, _ = self.process_batch_data(batch)
            # compute forward pass
            outputs, _, _, _, _, _ = self.model(
                measure_score_tensor=inputs,
                measure_metadata_tensor=None,
                train=False
            )
            # compute loss
            recons_loss = self.reconstruction_loss(
                x=inputs, x_recons=outputs
            )
            loss = recons_loss
            # compute mean loss and accuracy
            mean_loss += to_numpy(loss.mean())
            accuracy = self.mean_accuracy(
                weights=outputs,
                targets=inputs
            )
            mean_accuracy += to_numpy(accuracy)
        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (
            mean_loss,
            mean_accuracy
        )

    @staticmethod
    def reconstruction_loss(x, x_recons):
        return Trainer.mean_crossentropy_loss(weights=x_recons, targets=x)