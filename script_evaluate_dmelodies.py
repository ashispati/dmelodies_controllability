import os
import shutil
import numpy as np
from tqdm import tqdm
import json
import torch
import pandas as pd
from dmelodies_torch_dataloader import DMelodiesTorchDataset
from src.utils.helpers import to_cuda_variable, to_numpy
from src.dmelodiesvae.dmelodies_vae import DMelodiesVAE
from src.dmelodiesvae.dmelodies_cnnvae import DMelodiesCNNVAE
from src.dmelodiesvae.dmelodies_vae_trainer import DMelodiesVAETrainer, LATENT_NORMALIZATION_FACTORS
from src.dmelodiesvae.dmelodies_cnnvae_trainer import DMelodiesCNNVAETrainer
from src.dmelodiesvae.interp_vae import InterpVAE
from src.dmelodiesvae.interp_vae_trainer import InterpVAETrainer
from src.dmelodiesvae.s2_vae import S2VAE
from src.dmelodiesvae.s2_vae_trainer import S2VAETrainer
from src.utils.plotting import create_heatmap, plot_dim_pdf, plot_dim

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=str,
    default='beta-VAE',
    choices=['beta-VAE', 'annealed-VAE', 'ar-VAE', 'interp-VAE', 's2-VAE']
)
parser.add_argument("--net_type", type=str, default='rnn', choices=['rnn', 'cnn'])
parser.add_argument("--gamma", type=float, default=None)
parser.add_argument("--delta", type=float, default=10.0)
parser.add_argument("--interp_num_dims", type=int, default=1)
parser.add_argument("--no_log", action='store_false')

args = parser.parse_args()

# Select the Type of VAE-model and the network architecture
model_type_list = ['interp-VAE', 's2-VAE', 'ar-VAE']  # , 'ar-VAE', 'interp-VAE', 's2-VAE']
net_type_list = ['rnn']

# Specify training params
seed_list = [0]  # , 1, 2]
model_dict = {
    'beta-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2],
        'gamma_list': [1.0]
    },
    'ar-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2],
        'gamma_list': [1.0],
        'delta': args.delta,
    },
    'interp-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2],
        'gamma_list': [1.0],
        'num_dims': args.interp_num_dims
    },
    's2-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2],
        'gamma_list': [1.0],
    }

}
num_epochs = 100
batch_size = 512

for m in model_type_list:
    for net_type in net_type_list:
        # Specify the network and trainer classes
        if m == 'interp-VAE':
            model = InterpVAE
            trainer = InterpVAETrainer
        elif m == 's2-VAE':
            model = S2VAE
            trainer = S2VAETrainer
        else:
            if net_type == 'cnn':
                model = DMelodiesCNNVAE
                trainer = DMelodiesCNNVAETrainer
            else:
                model = DMelodiesVAE
                trainer = DMelodiesVAETrainer
        c_list = model_dict[m]['capacity_list']
        b_list = model_dict[m]['beta_list']
        g_list = model_dict[m]['gamma_list']

        for c in c_list:
            for b in b_list:
                for g in g_list:
                    attr_change_mat = np.zeros((9, 9))
                    for seed in seed_list:
                        dataset = DMelodiesTorchDataset(seed=seed)
                        if m == 'interp-VAE':
                            vae_model = model(dataset, vae_type=net_type, num_dims=model_dict[m]['num_dims'])
                        elif m == 's2-VAE':
                            vae_model = model(dataset, vae_type=net_type)
                        else:
                            vae_model = model(dataset)
                        if torch.cuda.is_available():
                            vae_model.cuda()
                        trainer_args = {
                            'model_type': m,
                            'beta': b,
                            'capacity': c,
                            'lr': 1e-4,
                            'rand': seed
                        }
                        if m == 'ar-VAE':
                            trainer_args.update({'gamma': g})
                            trainer_args.update({'delta': model_dict[m]['delta']})
                        elif m == 'interp-VAE' or m == 's2-VAE':
                            trainer_args.update({'gamma': g})
                        vae_trainer = trainer(
                            dataset,
                            vae_model,
                            **trainer_args
                        )
                        if os.path.exists(vae_model.filepath):
                            print('Model exists. Running evaluation.')
                        else:
                            raise ValueError(f"Trained model doesn't exist {net_type}_{trainer_args}")

                        vae_trainer.load_model()
                        vae_trainer.dataset.load_dataset()
                        metrics = vae_trainer.compute_eval_metrics()
                        # print(json.dumps(metrics["mig_factors"], indent=2))
                        print(f"Model: {net_type}_{trainer_args}")

                        # # GENERATING LATENT DISTRIBUTION PLOTS
                        _, _, gen_test = vae_trainer.dataset.data_loaders(batch_size=256)
                        latent_codes, attributes, attr_list = vae_trainer.compute_representations(
                            gen_test
                        )
                        for attr in vae_trainer.attr_dict.keys():
                            save_filename = os.path.join(
                                vae_trainer.get_save_dir(vae_trainer.model),
                                f'data_dist_{m}_{attr}.png'
                            )
                            dim1 = vae_trainer.attr_dict[attr]
                            dim2 = 31
                            plot_dim(
                                latent_codes, attributes[:, dim1], save_filename, dim1=dim1, dim2=dim2,
                            )

                        # # GENERATING LATENT DENSITY RATIO METRIC
                        num_points_to_sample = 100000
                        latent_codes = np.random.normal(
                            0, 1, (num_points_to_sample, 32)
                        ).astype(np.float32)
                        for attr in vae_trainer.attr_dict.keys():
                            dim = vae_trainer.attr_dict[attr]
                            lims = metrics["reg_dim_limits"][attr]
                            latent_codes[dim, :] = np.random.uniform(
                                low=lims[1], high=lims[0], size=32
                            )

                        latent_codes = to_cuda_variable(torch.from_numpy(latent_codes))

                        mini_batch_size = 1000
                        num_mini_batches = num_points_to_sample // mini_batch_size
                        attr_labels_all = []
                        for i in tqdm(range(num_mini_batches)):
                            z_batch = latent_codes[i * mini_batch_size:(i + 1) * mini_batch_size, :]
                            _, samples = vae_trainer.decode_latent_codes(z_batch)
                            samples = samples.view(z_batch.size(0), -1)
                            labels = vae_trainer.compute_attribute_labels(samples)
                            attr_labels_all.append(labels)

                        attr_labels_all = np.concatenate(attr_labels_all, 0)
                        results_dict = {}
                        for attr in vae_trainer.attr_dict.keys():
                            curr_attr = attr_labels_all[:, vae_trainer.attr_dict[attr]]
                            results_dict[attr] = np.sum(curr_attr != -1) / num_points_to_sample

                        raw_values = np.array([val for val in results_dict.values()])
                        weights = to_numpy(LATENT_NORMALIZATION_FACTORS)
                        agg_metric = np.mean(raw_values)
                        weight_agg_metric = np.sum(raw_values * weights) / np.sum(weights)

                        print(json.dumps(results_dict, indent=2))
                        print(f"LDR Metric: {agg_metric:.3f}, {weight_agg_metric:.3f}")

                        # # GENERATING LATENT SURFACE PLOTS
                        _, _, gen_test = vae_trainer.dataset.data_loaders(batch_size=256)
                        latent_codes, attributes, attr_list = vae_trainer.compute_representations(
                            gen_test, num_batches=1,
                        )
                        n = 121
                        lc = latent_codes[n:n + 1, :]
                        for attr in vae_trainer.attr_dict.keys():
                            save_filename = os.path.join(
                                vae_trainer.get_save_dir(vae_trainer.model),
                                f'data_surface_{m}_{attr}.png'
                            )
                            dim1 = vae_trainer.attr_dict[attr]
                            dim2 = 31
                            lims = metrics["reg_dim_limits"][attr]
                            a, b = vae_trainer.plot_latent_surface(
                                torch.from_numpy(lc), attr, dim1=dim1, dim2=dim2, dim1_high=lims[0], dim1_low=lims[1]
                            )
                            plot_dim(a, b, save_filename, dim1=dim1, dim2=dim2)

                        # GENERATING LATENT INTERPOLATIONS (Scores)
                        vae_trainer.plot_latent_interpolations()