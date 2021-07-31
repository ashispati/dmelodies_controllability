import torch
from torch import nn
from torch.autograd import Variable

from src.utils.helpers import to_cuda_variable
from src.utils.model import Model
from src.dmelodiesvae.dmelodies_vae import DMelodiesVAE
from src.dmelodiesvae.dmelodies_cnnvae import DMelodiesCNNVAE


class FactorVAE(Model):
    def __init__(
            self,
            dataset,
            vae_type='CNN',
            discriminator_hidden_size=64,
            discriminator_layers=3,
    ):
        """
        Initializes the Measure VAE class object
        :param dataset: torch.Dataset object
        :param vae_type: string, 'CNN' or 'RNN'
        """
        super(FactorVAE, self).__init__()

        self.discriminator_layers = discriminator_layers
        self.discriminator_hidden_size = discriminator_hidden_size
        # define VAE
        if vae_type == 'RNN':
            self.VAE = DMelodiesVAE(dataset).cuda()
        elif vae_type == 'CNN':
            self.VAE = DMelodiesCNNVAE(dataset).cuda()
        self.vae_type = vae_type
        # define Discriminator
        discriminator = []
        hidden_sizes = [self.VAE.latent_space_dim]
        hidden_sizes.extend([self.discriminator_hidden_size]*self.discriminator_layers)
        for i in range(self.discriminator_layers):
            discriminator.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            discriminator.append(nn.LeakyReLU(0.2, True))
        # add output layer
        discriminator.append(nn.Linear(self.discriminator_hidden_size, 2))
        self.discriminator = nn.Sequential(*discriminator)

        self.xavier_initialization()

        # location to save src
        self.update_filepath()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return 'FactorVAE_{}'.format(self.vae_type) + self.trainer_config

    def forward(self, measure_score_tensor: Variable,
                measure_metadata_tensor: Variable, train=True):
        """
        Implements the forward pass of the VAE
        :param measure_score_tensor: torch Variable,
                (batch_size, measure_seq_length)
        :param measure_metadata_tensor: torch Variable,
                (batch_size, measure_seq_length, num_metadata)
        :param train: bool,
        :return: torch Variable,
                (batch_size, measure_seq_length, self.num_notes)
        """
        # check input
        seq_len = measure_score_tensor.size(1)
        assert(seq_len == 16)
        # compute output of encoding layer
        z_dist = self.VAE.encoder(measure_score_tensor)

        # sample from distribution
        z_tilde = z_dist.rsample()

        # compute prior
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()

        # compute output of decoding layer
        weights, samples = self.VAE.decoder(
            z=z_tilde,
            score_tensor=measure_score_tensor,
            train=train
        )
        return weights, samples, z_dist, prior_dist, z_tilde, z_prior

    def forward_test(self, measure_score_tensor: Variable):
        """
        Implements the forward pass of the VAE
        :param measure_score_tensor: torch Variable,
                (batch_size, num_measures, measure_seq_length)
        :return: torch Variable,
                (batch_size, measure_seq_length, self.num_notes)
        """
        # check input
        batch_size, num_measures, seq_len = measure_score_tensor.size()
        assert(seq_len == self.num_ticks_per_measure)

        # compute output of encoding layer
        z = []
        for i in range(num_measures):
            z_dist = self.VAE.encoder(measure_score_tensor[:, i, :])
            z.append(z_dist.rsample().unsqueeze(1))
        z_tilde = torch.cat(z, 1)

        # compute output of decoding layer
        weights = []
        samples = []
        dummy_measure_tensor = to_cuda_variable(torch.zeros(batch_size, seq_len))
        for i in range(num_measures):
            w, s = self.VAE.decoder(
                z=z_tilde[:, i, :],
                score_tensor=dummy_measure_tensor,
                train=False
            )
            samples.append(s)
            weights.append(w.unsqueeze(1))
        samples = torch.cat(samples, 2)
        weights = torch.cat(weights, 1)
        return weights, samples
