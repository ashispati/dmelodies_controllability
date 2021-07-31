import torch
from torch import nn
from torch.autograd import Variable

from src.utils.helpers import to_cuda_variable
from src.utils.model import Model
from src.dmelodiesvae.cnn_encoder import CNNEncoder
from src.dmelodiesvae.cnn_decoder import CNNDecoder


class DMelodiesCNNVAE(Model):
    def __init__(
            self,
            dataset,
            note_embedding_dim=10,
            metadata_embedding_dim=2,
            latent_space_dim=32,
            has_metadata=True,
            encoder_cnn_hidden_size=[16, 32, 64, 128],
            encoder_cnn_filter_len=[4, 4, 4, 4],
            encoder_cnn_stride=[2, 2, 2, 2],
            dropout=0.1,
            decoder_cnn_hidden_size=[128, 64, 32, 16],
            decoder_cnn_filter_len=[4, 4, 4, 4],
            decoder_cnn_stride=[2, 2, 2, 2]
    ):
        """
        Initializes the Measure VAE class object
        :param dataset: torch.Dataset object
        :param note_embedding_dim: int,
        :param metadata_embedding_dim: int,
        :param encoder_cnn_hidden_size: list,
        :param encoder_cnn_filter_len: list,
        :param encoder_cnn_stride: list,
        :param latent_space_dim: int,
        :param decoder_cnn_hidden_size: list,
        :param decoder_cnn_filter_len: list,
        :param decoder_cnn_stride: list,
        :param dropout: float, from 0. to 1.
        :param has_metadata: bool
        :param dataset_type: folk or bach
        """
        super(DMelodiesCNNVAE, self).__init__()

        # define constants
        self.num_beats_per_measure = 4  # TODO: remove this hardcoding
        self.num_ticks_per_measure = 8  # TODO: remove this hardcoding
        self.num_ticks_per_beat = int(self.num_ticks_per_measure / self.num_beats_per_measure)

        # initialize members
        self.dataset = dataset.__repr__()
        self.note_embedding_dim = note_embedding_dim
        self.metadata_embedding_dim = metadata_embedding_dim
        self.encoder_cnn_hidden_size = encoder_cnn_hidden_size
        self.encoder_cnn_filter_len = encoder_cnn_filter_len
        self.encoder_cnn_stride = encoder_cnn_stride
        self.latent_space_dim = latent_space_dim
        self.decoder_cnn_hidden_size = decoder_cnn_hidden_size
        self.decoder_cnn_filter_len = decoder_cnn_filter_len
        self.decoder_cnn_stride = decoder_cnn_stride
        self.dropout = dropout
        self.has_metadata = has_metadata
        self.num_notes = 62  # TODO: remove this hardcoding
        # Encoder
        self.encoder = CNNEncoder(
            note_embedding_dim=self.note_embedding_dim,
            cnn_hidden_size=self.encoder_cnn_hidden_size,
            cnn_filter_len=self.encoder_cnn_filter_len,
            cnn_stride=self.encoder_cnn_stride,
            num_notes=self.num_notes,
            dropout=self.dropout,
            z_dim=self.latent_space_dim
        )

        # Decoder
        self.decoder = CNNDecoder(
            z_dim=self.latent_space_dim,
            cnn_hidden_size=self.decoder_cnn_hidden_size,
            cnn_filter_len=self.decoder_cnn_filter_len,
            cnn_stride=self.decoder_cnn_stride,
            num_notes=self.num_notes,
            dropout=self.dropout
        )

        # location to save src
        self.update_filepath()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return 'DMelodiesVAE_CNN' + self.trainer_config

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
        z_dist = self.encoder(measure_score_tensor)

        # sample from distribution
        z_tilde = z_dist.rsample()

        # compute prior
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()

        # compute output of decoding layer
        weights, samples = self.decoder(
            z=z_tilde,
        )
        return weights, samples, z_dist, prior_dist, z_tilde, z_prior

    def encode(self, measure_score_tensor: Variable):
        """
        Encodes the input data
        :param measure_score_tensor: torch Variable,
                (batch_size, measure_seq_length)
        :return: z_tilde torch Variable,
                (batch_size, latent_space_dim)
        """
        # check input
        seq_len = measure_score_tensor.size(1)
        assert(seq_len == 16)
        # compute output of encoding layer
        z_dist = self.encoder(measure_score_tensor)

        # sample from distribution
        z_tilde = z_dist.rsample()

        return z_tilde

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
            z_dist = self.encoder(measure_score_tensor[:, i, :])
            z.append(z_dist.rsample().unsqueeze(1))
        z_tilde = torch.cat(z, 1)

        # compute output of decoding layer
        weights = []
        samples = []
        dummy_measure_tensor = to_cuda_variable(torch.zeros(batch_size, seq_len))
        for i in range(num_measures):
            w, s = self.decoder(
                z=z_tilde[:, i, :],
                score_tensor=dummy_measure_tensor,
                train=False
            )
            samples.append(s)
            weights.append(w.unsqueeze(1))
        samples = torch.cat(samples, 2)
        weights = torch.cat(weights, 1)
        return weights, samples
