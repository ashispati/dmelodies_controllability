import torch
from torch import nn, distributions

from src.utils.helpers import to_cuda_variable
from src.utils.model import Model


class Encoder(Model):
    def __init__(self,
                 note_embedding_dim,
                 rnn_hidden_size,
                 num_layers,
                 num_notes,
                 dropout,
                 bidirectional,
                 z_dim,
                 rnn_class):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.note_embedding_dim = note_embedding_dim
        self.num_layers = num_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.z_dim = z_dim
        self.dropout = dropout
        self.rnn_class = rnn_class
        self.lstm = self.rnn_class(
            input_size=note_embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        self.num_notes = num_notes
        self.note_embedding_layer = nn.Embedding(self.num_notes,
                                                 self.note_embedding_dim)

        self.linear_mean = nn.Sequential(
            nn.Linear(self.rnn_hidden_size * self.num_directions * self.num_layers,
                      self.rnn_hidden_size * self.num_directions),
            nn.SELU(),
            nn.Linear(self.rnn_hidden_size * self.num_directions, z_dim)
        )

        self.linear_log_std = nn.Sequential(
            nn.Linear(self.rnn_hidden_size * self.num_directions * self.num_layers,
                      self.rnn_hidden_size * self.num_directions),
            nn.SELU(),
            nn.Linear(self.rnn_hidden_size * self.num_directions, z_dim)
        )

        self.xavier_initialization()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return f'Encoder(' \
               f'{self.note_embedding_dim},' \
               f'{self.rnn_class},' \
               f'{self.num_layers},' \
               f'{self.rnn_hidden_size},' \
               f'{self.dropout},' \
               f'{self.bidirectional},' \
               f'{self.z_dim},' \
               f')'

    def hidden_init(self, batch_size):
        """
        Initializes the hidden state of the encoder GRU
        :param batch_size: int
        :return: torch tensor,
               (self.num_encoder_layers x self.num_directions, batch_size, self.encoder_hidden_size)
        """
        hidden = torch.zeros(self.num_layers * self.num_directions,
                             batch_size,
                             self.rnn_hidden_size
                             )
        return to_cuda_variable(hidden)

    def embed_forward(self, score_tensor):
        """
        Performs the forward pass of the embedding layer
        :param score_tensor: torch tensor,
                (batch_size, measure_seq_len)
        :return: torch tensor,
                (batch_size, measure_seq_len, embedding_size)
        """
        x = self.note_embedding_layer(score_tensor)
        return x

    def forward(self, score_tensor):
        """
        Performs the forward pass of the src, overrides torch method
        :param score_tensor: torch Variable
                (batch_size, measure_seq_len)
        :return: torch distribution
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nan_check = torch.isnan(param.data)
                if nan_check.nonzero().size(0) > 0:
                    print('Encoder has become nan')
                    raise ValueError

        batch_size, measure_seq_len = score_tensor.size()

        # embed score
        embedded_seq = self.embed_forward(score_tensor=score_tensor)

        # pass through RNN
        hidden = self.hidden_init(batch_size)
        _, hidden = self.lstm(embedded_seq, hidden)
        hidden = hidden.transpose(0, 1).contiguous()
        hidden = hidden.view(batch_size, -1)

        # compute distribution parameters
        z_mean = self.linear_mean(hidden)
        z_log_std = self.linear_log_std(hidden)

        z_distribution = distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
        return z_distribution
