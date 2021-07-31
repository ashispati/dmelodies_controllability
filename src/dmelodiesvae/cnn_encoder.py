import torch
from torch import nn, distributions

from src.utils.helpers import to_cuda_variable
from src.utils.model import Model


class CNNEncoder(Model):
    def __init__(self,
                 note_embedding_dim,
                 cnn_hidden_size,
                 cnn_filter_len,
                 cnn_stride,
                 num_notes,
                 dropout,
                 z_dim):
        super(CNNEncoder, self).__init__()
        self.note_embedding_dim = note_embedding_dim
        self.z_dim = z_dim
        self.dropout = dropout
        self.cnn_hidden_size = cnn_hidden_size

        assert len(cnn_hidden_size) == len(cnn_filter_len)
        cnn_layer = []
        # cnn_hidden_size.insert(0, note_embedding_dim)
        for i in range(len(cnn_hidden_size)):
            if i == 0:
                cnn_layer.append(nn.Conv1d(note_embedding_dim, cnn_hidden_size[i], cnn_filter_len[i], cnn_stride[i], 1))
            else:
                cnn_layer.append(nn.Conv1d(cnn_hidden_size[i-1], cnn_hidden_size[i], cnn_filter_len[i], cnn_stride[i], 1))
            cnn_layer.append(nn.SELU())
            cnn_layer.append(nn.Dropout(self.dropout))
        # Last layer
        self.cnn = nn.Sequential(*cnn_layer)

        self.num_notes = num_notes
        self.note_embedding_layer = nn.Embedding(self.num_notes,
                                                 self.note_embedding_dim)

        self.linear_mean = nn.Sequential(
            nn.Linear(cnn_hidden_size[-1], cnn_hidden_size[-1]//2),
            nn.SELU(),
            nn.Linear(cnn_hidden_size[-1]//2, z_dim)
        )

        self.linear_log_std = nn.Sequential(
            nn.Linear(cnn_hidden_size[-1], cnn_hidden_size[-1]//2),
            nn.SELU(),
            nn.Linear(cnn_hidden_size[-1]//2, z_dim)
        )

        self.xavier_initialization()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return f'CNNEncoder(' \
               f'{self.note_embedding_dim},' \
               f'{self.cnn_hidden_size},' \
               f'{self.cnn_filter_len},' \
               f'{self.cnn_stride},' \
               f'{self.dropout},' \
               f'{self.z_dim},' \
               f')'

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

        # embed score and transpose for Conv1D
        embedded_seq = self.embed_forward(score_tensor=score_tensor).transpose(1, 2)

        # pass through CNN
        hidden = self.cnn(embedded_seq)

        # average pool anything time-steps left
        hidden = hidden.mean(2)

        # compute distribution parameters
        z_mean = self.linear_mean(hidden)
        z_log_std = self.linear_log_std(hidden)

        z_distribution = distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
        return z_distribution
