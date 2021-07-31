import torch
from torch import nn
from src.dmelodiesvae.decoder import Decoder

class CNNDecoder(Decoder):
    def __init__(self,
                 z_dim,
                 cnn_hidden_size,
                 cnn_filter_len,
                 cnn_stride,
                 num_notes,
                 dropout):
        # note_embedding_dim is set to 1 because this doesn't need an embedding
        super(CNNDecoder, self).__init__(1, num_notes, z_dim)

        assert len(cnn_hidden_size) == len(cnn_filter_len)
        cnn_layer = []
        # cnn_hidden_size.insert(0, z_dim)
        for i in range(len(cnn_hidden_size)):
            if i == 0:
                cnn_layer.append(nn.ConvTranspose1d(z_dim, cnn_hidden_size[i], cnn_filter_len[i], cnn_stride[i], 1))
            else:
                cnn_layer.append(nn.ConvTranspose1d(cnn_hidden_size[i-1], cnn_hidden_size[i], cnn_filter_len[i], cnn_stride[i], 1))
            cnn_layer.append(nn.SELU())
            # cnn_layer.append(nn.Dropout(self.dropout))
        # Last layer
        self.cnn = nn.Sequential(*cnn_layer)

        self.hidden_to_notes = nn.Linear(cnn_hidden_size[-1], num_notes)

        self.xavier_initialization()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return f'CNNDecoder(' \
               f'{self.cnn_hidden_size},' \
               f'{self.cnn_filter_len},' \
               f'{self.cnn_stride},' \
               f'{self.dropout},' \
               f'{self.z_dim},' \
               f')'
    
    def forward(self, z, score_tensor=None, train=None):
        """
        Performs the forward pass of the src, overrides torch method
        :param z: torch tensor,
                (batch_size, self.z_dim)
        :return: weights: torch tensor,
                (batch_size, measure_seq_len, self.num_notes)
                samples: torch tensor,
                (batch_size, measure_seq_len)
        """

        out = self.cnn(z.unsqueeze(-1))
        # batch_size, h_size, measure_seq_len

        note_logits = self.hidden_to_notes(out.transpose(1, 2))

        # argmax to get notes
        notes = torch.argmax(note_logits.detach(), 2)

        return note_logits, notes