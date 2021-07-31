import os
import numpy as np
import json
import torch
from dmelodies_torch_dataloader import DMelodiesTorchDataset
from src.dmelodiesvae.dmelodies_vae import DMelodiesVAE
from src.dmelodiesvae.dmelodies_cnnvae import DMelodiesCNNVAE
from src.dmelodiesvae.dmelodies_vae_trainer import DMelodiesVAETrainer
from src.dmelodiesvae.dmelodies_cnnvae_trainer import DMelodiesCNNVAETrainer
from src.dmelodiesvae.interp_vae import InterpVAE
from src.dmelodiesvae.interp_vae_trainer import InterpVAETrainer
from src.dmelodiesvae.s2_vae import S2VAE
from src.dmelodiesvae.s2_vae_trainer import S2VAETrainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=str,
    default='beta-VAE',
    choices=['beta-VAE', 'ar-VAE', 'interp-VAE', 's2-VAE']
)
parser.add_argument("--net_type", type=str, default='cnn', choices=['rnn', 'cnn'])
parser.add_argument("--gamma", type=float, default=None)
parser.add_argument("--delta", type=float, default=10.0)
parser.add_argument("--interp_num_dims", type=int, default=1)
parser.add_argument("--no_log", action='store_false')

args = parser.parse_args()

# Select the Type of VAE-model and the network architecture
m = args.model_type
net_type = args.net_type

# Specify training params
seed_list = [0, 1, 2]
model_dict = {
    'beta-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2, 1.0, 4.0],
        'gamma_list': [1.0]
    },
    'ar-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2],
        'gamma_list': [0.1, 1.0, 10.0],
        'delta': args.delta,
    },
    'interp-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2],
        'gamma_list': [0.1, 1.0, 10.0],
        'num_dims': args.interp_num_dims
    },
    's2-VAE': {
        'capacity_list': [50.0],
        'beta_list': [0.2],
        'gamma_list': [0.1, 1.0, 10.0],
    }

}
num_epochs = 100
batch_size = 512

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
for seed in seed_list:
    for c in c_list:
        for b in b_list:
            for g in g_list:
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
                if not os.path.exists(vae_model.filepath):
                    vae_trainer.train_model(batch_size=batch_size, num_epochs=num_epochs, log=args.no_log)
                else:
                    print('Model exists. Running evaluation.')
                vae_trainer.load_model()
                metrics = vae_trainer.compute_eval_metrics()
                print(f"Model: {net_type}_{trainer_args}")
                print(json.dumps(metrics, indent=2))
                print(vae_trainer.test_model(batch_size=512))
                vae_trainer.update_reg_dim_limits()
                vae_trainer.evaluate_latent_interpolations()
