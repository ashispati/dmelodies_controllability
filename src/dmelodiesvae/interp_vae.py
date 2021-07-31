import torch
from torch import nn
from torch.autograd import Variable

from src.utils.helpers import to_cuda_variable
from src.utils.model import Model
from src.dmelodiesvae.dmelodies_vae import DMelodiesVAE
from src.dmelodiesvae.dmelodies_cnnvae import DMelodiesCNNVAE


class InterpVAE(Model):
    def __init__(
            self,
            dataset,
            vae_type='cnn',
            num_dims=None,
            attr_class_list=tuple([12, 3, 3, 28, 28, 2, 2, 2, 2]),
    ):
        """
        Initializes the Measure VAE class object
        :param dataset: torch.Dataset object
        :param vae_type: string, 'CNN' or 'RNN'
        """
        super(InterpVAE, self).__init__()

        # define VAE
        if vae_type == 'rnn':
            self.VAE = DMelodiesVAE(dataset)
        elif vae_type == 'cnn':
            self.VAE = DMelodiesCNNVAE(dataset)
        self.vae_type = vae_type

        # define linear classifiers
        self.attr_class_list = attr_class_list
        self.num_dims = num_dims
        if self.num_dims is None:
            self.attr_classifiers = nn.ModuleList(
                [nn.Linear(self.VAE.latent_space_dim, i) for i in self.attr_class_list]
            )
        elif self.num_dims <= 3:
            self.attr_classifiers = nn.ModuleList(
                [nn.Linear(self.num_dims, i) for i in self.attr_class_list]
            )
        else:
            raise ValueError("Invalid number of dimensions. Can be 1, 2 or 3 only currently")

        # initialize params
        self.xavier_initialization()

        # location to save src
        self.update_filepath()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return f'DMelodiesVAE_{self.vae_type}' + self.trainer_config

    def forward(self, measure_score_tensor: Variable,
                measure_metadata_tensor: Variable, train=True):
        """
        Implements the forward pass of the VAE
        :param measure_score_tensor: torch Variable,
                (batch_size, measure_seq_length)git s
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
