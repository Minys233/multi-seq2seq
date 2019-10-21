import random
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout_p, bidirectional):
        """
        Seq2seq encoder
        :param input_dim: Vocabulary size
        :param embed_dim: Embedding feature size
        :param hidden_dim: Number of features in the hidden state of GRU
        :param num_layers: Number of layers of GRU
        :param dropout_p: Dropout after embedding layer and between GRUs' output
        :param bidirectional: If GRU is bidirectional
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, dropout=dropout_p, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, src: torch.Tensor):
        """
        Seq2seq encode
        :param src: shape [seq_len, batch_size]
        :return: The outputs of shape [seq_len, batch_size, directions*hidden_dim]
                 and the hidden state of shape [num_layers*directions, batch, hidden_dim]
        """
        embedding = self.dropout(self.embedding(src))
        # embedding: [seq_len, batch_size, embed_dim]
        outputs, hidden = self.gru(embedding)
        # outputs: [seq_len, batch_size, directions*hidden_dim]
        # hidden: [num_layers*directions, batch, hidden_dim]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout_p, bidirectional):
        super().__init__()
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, dropout=dropout_p, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(2*hidden_dim, output_dim)
        else:
            self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        """
        Decode the sequence into string
        :param input: shape [batch_size] of single tokens.
        :param hidden: shape [num_layers * directions, batch, hidden_dim] of previous time
        :return: prediction, hidden
        """
        input = input.unsqueeze(0)
        # input: [1, batch_size]
        embedded = self.embedding(input)
        # embedded: [1, batch_size, embed_dim]
        output, hidden = self.gru(embedded, hidden)
        # output: [1, batch_size, directions*hidden_dim]
        # hidden: [num_layers*directions, batch_size, hidden_dim]
        prediction = self.linear(output.squeeze(0))
        # prediction: [batch_size, output_dim]
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_dim == decoder.hidden_dim, \
            "Hidden dimensions for encoder and decoder GRU/LSTM should be same"
        assert encoder.num_layers == decoder.num_layers, \
            "The GRU/LSTM layers should be same for encoder or decoder." \
            "Or you should manually add a linear layer after encoder."

    def forward(self, src, trg, teacher_force_ratio=0.5):
        """

        :param src: Input sequences of indexes of tokens, shape: [seq_len1, batch_size]
        :param trg: Output sequences of indexes of tokens, shape: [seq_len2, batch_size]
        :param teacher_force_ratio:
        :return:
        """

        # output should have shape: [max_len, batch_size, output_dim]
        # Here assume src and trg is padded by patch to their max_len
        decoded = torch.zeros(trg.shape[0], src.shape[1], self.decoder.output_dim).to(self.device)
        # encode, encode_output is not used in this simple seq2seq model
        encode_output, hidden = self.encoder(src)
        # decoder's first input, <SOS> for every sample in a batch, shape [batch_size]
        decode_input = trg[0]
        # for the rest of decoder's input
        for idx in range(1, trg.shape[0]):
            decoded[idx], hidden = self.decoder(decode_input, hidden)
            teacher_force = random.random() < teacher_force_ratio
            top1 = decoded[idx].argmax(dim=1)
            # top1: [batch_size]
            decode_input = trg[idx] if teacher_force else top1
        return decoded




