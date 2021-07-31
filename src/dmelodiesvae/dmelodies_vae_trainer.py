import os
import json
import torch
from tqdm import tqdm
import music21
from typing import Tuple
import pandas as pd

from helpers import concatenate_scores
from src.utils.trainer import Trainer
from src.dmelodiesvae.dmelodies_vae import DMelodiesVAE
from src.utils.helpers import to_cuda_variable_long, to_cuda_variable, to_numpy
from src.utils.evaluation import *
from src.utils.plotting import *


LATENT_ATTRIBUTES = {
    'tonic': 0,
    'octave': 1,
    'mode': 2,
    'rhythm_bar1': 3,
    'rhythm_bar2': 4,
    'arp_chord1': 5,
    'arp_chord2': 6,
    'arp_chord3': 7,
    'arp_chord4': 8
}

LATENT_NORMALIZATION_FACTORS = torch.tensor(
    [11, 2, 2, 27, 27, 1, 1, 1, 1],
    dtype=torch.float32
)


class DMelodiesVAETrainer(Trainer):
    def __init__(
            self,
            dataset,
            model: DMelodiesVAE,
            model_type='beta-VAE',
            lr=1e-4,
            beta=0.001,
            gamma=1.0,
            delta=10.0,
            capacity=0.0,
            device=0,
            rand=0,
    ):
        super(DMelodiesVAETrainer, self).__init__(dataset, model, lr)
        self.model_type = model_type
        self.attr_dict = LATENT_ATTRIBUTES
        self.attr_norm_factors = LATENT_NORMALIZATION_FACTORS
        self.reverse_attr_dict = {
            v: k for k, v in self.attr_dict.items()
        }
        self.metrics = {}
        self.beta = beta
        self.capacity = capacity
        # self.capacity = to_cuda_variable(torch.FloatTensor([capacity]))
        self.cur_epoch_num = 0
        self.warm_up_epochs = 10
        self.num_iterations = 100000
        if self.model_type == 'beta-VAE':
            self.exp_rate = np.log(1 + self.beta) / self.num_iterations
            self.start_beta = 0.0
            self.cur_beta = self.start_beta
            self.start_capacity = self.capacity
            self.cur_capacity = self.capacity
        elif self.model_type == 'annealed-VAE':
            self.exp_rate = np.log(1 + self.capacity) / self.num_iterations
            self.start_beta = 0.0
            self.cur_beta = self.start_beta
            self.start_capacity = 0.0
            self.cur_capacity = self.start_capacity
        elif self.model_type == 'ar-VAE':
            self.exp_rate = np.log(1 + self.beta) / self.num_iterations
            self.start_beta = 0.0
            self.cur_beta = self.start_beta
            self.start_capacity = self.capacity
            self.cur_capacity = self.capacity
            self.gamma = gamma
            self.delta = delta
        self.anneal_iterations = 0
        self.device = device
        self.rand_seed = rand
        torch.manual_seed(self.rand_seed)
        np.random.seed(self.rand_seed)
        self.trainer_config = f'_{self.model_type}_b_{self.beta}_c_{self.capacity}_'
        if model_type == 'ar-VAE':
            self.trainer_config += f'g_{self.gamma}_d_{self.delta}_'
        self.trainer_config += f'r_{self.rand_seed}_'
        self.model.update_trainer_config(self.trainer_config)

    def update_scheduler(self, epoch_num):
        """
        Updates the training scheduler if any
        :param epoch_num: int,
        """
        if epoch_num > self.warm_up_epochs:
            if self.anneal_iterations < self.num_iterations:
                if self.model_type == 'beta-VAE':
                    self.cur_beta = -1.0 + np.exp(self.exp_rate * self.anneal_iterations)
                elif self.model_type == 'annealed-VAE':
                    self.cur_beta = self.beta
                    self.cur_capacity = -1.0 + np.exp(self.exp_rate * self.anneal_iterations)
                elif self.model_type == 'ar-VAE':
                    self.cur_beta = -1.0 + np.exp(self.exp_rate * self.anneal_iterations)
            self.anneal_iterations += 1

    def process_batch_data(self, batch):
        """
        Processes the batch returned by the dataloader iterator
        :param batch: object returned by the dataloader iterator
        :return: tuple of Torch Variable objects
        """
        score_tensor, latent_tensor = batch
        # convert input to torch Variables
        batch_data = (
            to_cuda_variable_long(score_tensor.squeeze(1), self.device),
            to_cuda_variable_long(latent_tensor.squeeze(1), self.device)
        )
        return batch_data

    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        """
        Computes the loss and accuracy for the batch
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
        if self.model_type == 'beta-VAE':
            dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.cur_beta, c=0.0)
            dist_loss = torch.nn.functional.relu(dist_loss - self.cur_capacity)
        elif self.model_type == 'annealed-VAE':
            dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.cur_beta, c=self.cur_capacity)
        elif self.model_type == 'ar-VAE':
            dist_loss = self.compute_kld_loss(z_dist, prior_dist, beta=self.cur_beta, c=0.0)
            dist_loss = torch.nn.functional.relu(dist_loss - self.cur_capacity)
        else:
            raise ValueError('Invalid Model Type')

        # add loses
        loss = recons_loss + dist_loss

        # add regularization loss for ar-VAE
        reg_loss = 0.0
        if self.model_type == 'ar-VAE':
            # process latent attributes
            metadata = self.normalize_latent_attributes(latent_attributes)
            # compute regularization loss
            for attr in self.attr_dict.keys():
                dim = self.attr_dict[attr]
                labels = metadata[:, dim]
                reg_loss += self.compute_reg_loss(
                    z_tilde, labels, dim, gamma=self.gamma, factor=self.delta
                )
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
            if self.model_type == 'ar-VAE':
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

    def normalize_latent_attributes(self, latent_attributes):
        metadata = latent_attributes.clone().float()
        metadata = torch.div(metadata, to_cuda_variable(self.attr_norm_factors))
        return metadata

    def compute_representations(self, data_loader, num_batches=None, return_input=False):
        latent_codes = []
        attributes = []
        if return_input:
            input_data = []
        if num_batches is None:
            num_batches = 200
        for batch_id, batch in tqdm(enumerate(data_loader)):
            inputs, latent_attributes = self.process_batch_data(batch)
            _, _, _, _, z_tilde, _ = self.model(inputs, None, train=False)
            latent_codes.append(to_numpy(z_tilde.cpu()))
            attributes.append(to_numpy(latent_attributes))
            if return_input:
                input_data.append(to_numpy(inputs))
            if batch_id == num_batches:
                break
        latent_codes = np.concatenate(latent_codes, 0)
        attributes = np.concatenate(attributes, 0)
        attr_list = [
            attr for attr in self.attr_dict.keys()
        ]
        if return_input:
            input_data = np.concatenate(input_data, 0)
            return latent_codes, attributes, attr_list, input_data
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
        if os.path.exists(results_fp):
            with open(results_fp, 'r') as infile:
                self.metrics = json.load(infile)
                return self.metrics
        batch_size = 512
        _, _, gen_test = self.dataset.data_loaders(batch_size=batch_size, split=(0.70, 0.20))
        latent_codes, attributes, attr_list = self.compute_representations(gen_test)
        self.metrics.update(compute_mig(latent_codes, attributes))
        mig_factors = self.metrics["mig_factors"]
        self.metrics["mig_factors"] = {attr: mig for attr, mig in zip(attr_list, mig_factors)}
        self.metrics.update(compute_modularity(latent_codes, attributes))
        self.metrics.update(compute_sap_score(latent_codes, attributes))
        self.metrics.update(self.test_model(batch_size=batch_size))
        with open(results_fp, 'w') as outfile:
            json.dump(self.metrics, outfile, indent=2)
        return self.metrics

    def test_model(self, batch_size):
        _, _, gen_test = self.dataset.data_loaders(batch_size)
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

    def plot_data_dist(self, latent_codes, attributes, attr_str, dim1=0, dim2=1):
        save_filename = os.path.join(
            Trainer.get_save_dir(self.model),
            'data_dist_' + attr_str + '.png'
        )
        img = plot_dim(
            latent_codes, attributes[:, self.attr_dict[attr_str]], save_filename, dim1=dim1, dim2=dim2,
        )
        return img

    def compute_latent_hole_metric(self, ):
        pass

    def plot_latent_surface(self, z, attr_str, dim1=0, dim2=1, dim1_low=-5.0, dim1_high=5.0):
        """
        Plots the value of an attribute over a surface defined by the dimensions
        :param z: input latent code
        :param dim1: int,
        :param dim2: int,
        :param grid_res: float,
        :return:
        """
        # create the dataspace
        x1 = torch.linspace(dim1_low, dim1_high, steps=200)
        x2 = torch.linspace(-3., 3., steps=200)
        z1, z2 = torch.meshgrid([x1, x2])
        num_points = z1.size(0) * z1.size(1)
        # z = torch.randn(1, self.model.latent_space_dim)
        z = z.repeat(num_points, 1)
        z[:, dim1] = z1.contiguous().view(1, -1)
        z[:, dim2] = z2.contiguous().view(1, -1)
        z = to_cuda_variable(z)

        mini_batch_size = 500
        num_mini_batches = num_points // mini_batch_size
        attr_labels_all = []
        for i in tqdm(range(num_mini_batches)):
            z_batch = z[i * mini_batch_size:(i+1) * mini_batch_size, :]
            _, samples = self.decode_latent_codes(z_batch)
            # dummy_score_tensor = to_cuda_variable(
            #     torch.zeros(z_batch.size(0), 16)
            # )
            # _, samples = self.model.decoder(
            #     z=z_batch,
            #     score_tensor=dummy_score_tensor,
            #     train=False
            # )
            samples = samples.view(z_batch.size(0), -1)
            labels = self.compute_attribute_labels(samples)
            attr_labels_all.append(labels)

        attr_labels_all = np.concatenate(attr_labels_all, 0)
        z = to_numpy(z)[:num_mini_batches*mini_batch_size, :]
        # remove points with undefined attributes
        plot_attr = attr_labels_all[:, self.attr_dict[attr_str]]
        a = z[~(plot_attr == -1), :]
        b = plot_attr[~(plot_attr == -1)]
        # save_filename = os.path.join(
        #     Trainer.get_save_dir(self.model),
        #     f'data_surface_{attr_str}.png'
        # )
        return a, b
        # plot_dim(a, b, save_filename, dim1=dim1, dim2=dim2)

    def plot_latent_interpolations(self):
        results_fp = os.path.join(
            os.path.dirname(self.model.filepath),
            'results_dict.json'
        )
        with open(results_fp, 'r') as infile:
            metrics = json.load(infile)
        reg_lim_dict = None
        if "reg_dim_limits" in metrics.keys():
            reg_lim_dict = metrics["reg_dim_limits"]
        _, _, gen_test = self.dataset.data_loaders(batch_size=256)
        latent_codes, attributes, attr_list, input_data = self.compute_representations(
            gen_test, num_batches=1, return_input=True
        )

        # n = min(num_points, latent_codes.shape[0])
        # interp_dict = self.compute_eval_metrics()["mig_factors"]
        n = 121
        lc = latent_codes[n:n+1, :]
        orig_data = input_data[n, :]
        # attr_labels = self.compute_attribute_labels(torch.from_numpy(orig_data).unsqueeze(0))
        # save original
        orig_score = self.dataset.tensor_to_m21score(torch.from_numpy(orig_data))
        orig_save_filepath = os.path.join(
            Trainer.get_save_dir(self.model),
            f'orig_{n}.mid'
        )
        orig_score.write('midi', fp=orig_save_filepath)
        # compute reconstruction as music21 score
        recons_score, _ = self.decode_latent_codes(torch.from_numpy(lc))
        recons_save_filepath = os.path.join(
            Trainer.get_save_dir(self.model),
            f'recons_{n}.mid'
        )
        recons_score.write('midi', fp=recons_save_filepath)
        # compute interpolations
        for i, attr_str in enumerate(attr_list):
            dim = self.attr_dict[attr_str]
            if reg_lim_dict is not None:
                max_lim = reg_lim_dict[attr_str][0]
                min_lim = reg_lim_dict[attr_str][1]
            else:
                max_lim = 4.0
                min_lim = -4.0
            score, tensor_score = self.compute_latent_interpolations(
                lc, orig_score, dim, num_points=5, max_lim=max_lim, min_lim=min_lim
            )
            # compute attributes for interpolations
            attr_labels = self.compute_attribute_labels(tensor_score.cpu())
            # write MIDI file
            save_filepath = os.path.join(
                Trainer.get_save_dir(self.model),
                f'latent_interpolations_{attr_str}_{n}.mid'
            )
            score.write('midi', fp=save_filepath)
            # plot MIDI
            plot_pianoroll_from_midi(save_filepath, attr_labels[:, i], attr_str)
            # plot_score_from_midi(save_filepath, attr_labels[:, i], attr_str)

    def decode_latent_codes(self, latent_codes):
        batch_size = latent_codes.size(0)
        dummy_score_tensor = to_cuda_variable(
            torch.zeros(batch_size, 16)
        )
        _, tensor_score = self.model.decoder(latent_codes, dummy_score_tensor, False)
        score = self.dataset.tensor_to_m21score(tensor_score)
        return score, tensor_score

    def compute_latent_interpolations(self, latent_code, original_score, dim1=0, num_points=6, max_lim=4.0, min_lim=-4.0):
        # assert num_points % 2 == 0
        x1 = torch.linspace(min_lim, max_lim, num_points)
        num_points = x1.size(0)
        z = to_cuda_variable(torch.from_numpy(latent_code))
        z = z.repeat(num_points, 1)
        z[:, dim1] = x1.contiguous()
        num_measures = z.size(0)
        score_list = []
        tensor_score_list = []
        for n in range(num_measures):
            score, tensor_score = self.decode_latent_codes(z[n:n+1, :])
            score_list.append(score)
            tensor_score_list.append(tensor_score)
        # score_list[num_points // 2] = original_score
        concatenated_score = concatenate_scores(score_list)
        concatenated_tensor_score = torch.cat(tensor_score_list)
        concatenated_tensor_score = torch.squeeze(concatenated_tensor_score, dim=1)
        return concatenated_score, concatenated_tensor_score

    def compute_attribute_labels(self, tensor_score):
        """
        Computes the attribute values for a score generated by the decoder
        Args:
            tensor_score: pytorch Tensor, N x 16, N is the batch size
        """
        attr_labels = np.zeros((tensor_score.shape[0], len(self.attr_dict.keys())))
        for i in range(tensor_score.shape[0]):
            attr_labels[i, :] = np.array(self.dataset.compute_attributes(tensor_score[i, :]))
        return attr_labels.astype('int')

    def update_non_reg_dim_limits(self, overwrite=False):
        results_fp = os.path.join(
            os.path.dirname(self.model.filepath),
            'results_dict.json'
        )
        with open(results_fp, 'r') as infile:
            metrics = json.load(infile)
        if "non_reg_dim_limits" in metrics.keys() and not overwrite:
            non_reg_lim_dict = np.array(metrics["non_reg_dim_limits"])
        else:
            _, gen_val, _ = self.dataset.data_loaders(batch_size=512)
            latent_codes, attributes, attr_list = self.compute_representations(gen_val)
            non_reg_lim_dict = {}
            attr_dims = [d for d in self.attr_dict.values()]
            latent_dims = list(np.arange(0, self.model.latent_space_dim))
            non_reg_dims = list(set(latent_dims) - set(attr_dims))
            for i in non_reg_dims:
                non_reg_lim_dict[str(i)] = (np.max(latent_codes[:, i]).item(), np.min(latent_codes[:, i]).item())
            metrics["non_reg_dim_limits"] = non_reg_lim_dict
            with open(results_fp, 'w') as outfile:
                json.dump(metrics, outfile, indent=2)
        return non_reg_lim_dict

    def update_reg_dim_limits(self, overwrite=False):
        results_fp = os.path.join(
            os.path.dirname(self.model.filepath),
            'results_dict.json'
        )
        with open(results_fp, 'r') as infile:
            metrics = json.load(infile)
        if "reg_dim_limits" in metrics.keys() and not overwrite:
            reg_lim_dict = np.array(metrics["reg_dim_limits"])
        else:
            _, gen_val, _ = self.dataset.data_loaders(batch_size=512)
            latent_codes, attributes, attr_list = self.compute_representations(gen_val)
            reg_lim_dict = {}
            for i, attr in enumerate(attr_list):
                reg_lim_dict[attr] = (np.max(latent_codes[:, i]).item(), np.min(latent_codes[:, i]).item())
            metrics["reg_dim_limits"] = reg_lim_dict
            with open(results_fp, 'w') as outfile:
                json.dump(metrics, outfile, indent=2)
        return reg_lim_dict

    def evaluate_latent_interpolations(self, overwrite=False, plot=False):
        results_fp = os.path.join(
            os.path.dirname(self.model.filepath),
            'results_dict.json'
        )
        with open(results_fp, 'r') as infile:
            metrics = json.load(infile)
        if "eval_interpolations" in metrics.keys() and not overwrite:
            attr_change_mat = np.array(metrics["eval_interpolations"])
        else:
            reg_lim_dict = metrics["reg_dim_limits"]
            _, _, gen_test = self.dataset.data_loaders(batch_size=256)
            latent_codes, attributes, attr_list, input_data = self.compute_representations(
                gen_test, num_batches=4-1, return_input=True
            )
            num_datapoints = latent_codes.shape[0]
            eval_mat = np.zeros((
                len(self.attr_dict.keys()),
                num_datapoints,
                len(self.attr_dict.keys())
            ))
            for n in tqdm(range(num_datapoints)):
                lc = latent_codes[n:n + 1, :]
                orig_data = input_data[n, :]
                orig_score = self.dataset.tensor_to_m21score(torch.from_numpy(orig_data))
                orig_attr_labels = self.compute_attribute_labels(torch.from_numpy(orig_data).unsqueeze(0))

                # compute interpolations
                for i, attr_str in enumerate(attr_list):
                    dim = self.attr_dict[attr_str]
                    lims = reg_lim_dict[attr_str]
                    score, tensor_score = self.compute_latent_interpolations(
                        lc,
                        orig_score,
                        dim,
                        num_points=5,
                        max_lim=lims[0],
                        min_lim=lims[1]
                    )
                    # compute attributes for interpolations
                    attr_labels = self.compute_attribute_labels(tensor_score.cpu())
                    diff_array = attr_labels - orig_attr_labels
                    diff_array[diff_array != 0] = 1
                    attr_change = np.sum(diff_array, axis=0)
                    eval_mat[i, n, :] = attr_change
            attr_change_mat = np.sum(eval_mat, axis=1) / eval_mat.shape[1]

            metrics["eval_interpolations"] = attr_change_mat.tolist()
            with open(results_fp, 'w') as outfile:
                json.dump(metrics, outfile, indent=2)

        # # save as heatmap
        # if plot:
        #     index = [i for i, _ in enumerate(self.attr_dict.keys())]
        #     columns = [k for _, k in enumerate(self.attr_dict.keys())]
        #     attr_change_mat = attr_change_mat / 6
        #     np.fill_diagonal(attr_change_mat, 1.0)
        #     data = pd.DataFrame(
        #         data=attr_change_mat,
        #         index=index,
        #         columns=columns,
        #     )
        #     save_filepath = os.path.join(
        #         Trainer.get_save_dir(self.model),
        #         f'eval_interpolations_norm.pdf'
        #     )
        #     create_heatmap(data, xlabel='Factor of Variation', ylabel='Regularized Dimension', save_path=save_filepath)

        return attr_change_mat

    @staticmethod
    def reconstruction_loss(x, x_recons):
        return Trainer.mean_crossentropy_loss(weights=x_recons, targets=x)

    @staticmethod
    def compute_reg_loss(z, labels, reg_dim, gamma, factor=1.0):
        """
        Computes the regularization loss
        """
        x = z[:, reg_dim]
        reg_loss = Trainer.reg_loss_sign(x, labels, factor=factor)
        return gamma * reg_loss
