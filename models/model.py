"""
    The CRNN architecture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.div(torch.matmul(Q, K.permute(0, 1, 3, 2)), np.sqrt(self.head_dim))

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]

        return x


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return self._get_name() + \
               '(in_channels={}, out_channels={}, key_channels={})'.format(
                   self.conv_Q.in_channels,
                   self.conv_V.out_channels,
                   self.conv_K.out_channels
               )


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.relu_(self.bn(self.conv(x)))
        return x


class CRNN(torch.nn.Module):
    def __init__(self, task, nb_cnn2d_filt, f_pool_size, t_pool_size, dropout_rate, subject_classes, sound_direction,
                 nb_rnn_layers, rnn_size, self_attn, nb_heads, nb_fnn_layers, fnn_size,
                 parameterization='normal', non_linearity='ReLu'):
        super().__init__()

        task_type = task.split('_')[0]
        self.task = task_type
        in_feat_shape = [10, 64, 64]

        if task_type == 'ide':
            out_shape = [subject_classes]
        elif task_type == 'loc':
            out_shape = [sound_direction]
        elif task_type == 'ideloc':
            out_shape_ide = [subject_classes]
            out_shape_loc = [sound_direction]
        elif task_type == 'accil':
            out_shape = [2*subject_classes]

        self.conv_block_list = torch.nn.ModuleList()
        if len(f_pool_size):
            for conv_cnt in range(len(f_pool_size)):
                # convolutional layer
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=nb_cnn2d_filt[conv_cnt - 1] if conv_cnt else in_feat_shape[0],
                        out_channels=nb_cnn2d_filt[conv_cnt]
                    )
                )

                # batch normalization layer, batch normalization involved in 'ConvBlock'
                # self.conv_block_list.append(
                #     torch.nn.BatchNorm2d(nb_cnn2d_filt[conv_cnt])
                # )

                # max pooling layer
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((f_pool_size[conv_cnt], t_pool_size[conv_cnt]))
                )

                # drop out layer
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=dropout_rate)
                )

        if nb_rnn_layers:
            self.in_gru_size = nb_cnn2d_filt[-1] * int(np.floor(in_feat_shape[-1] / np.prod(f_pool_size)))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=rnn_size,
                                    num_layers=nb_rnn_layers, batch_first=True,
                                    dropout=dropout_rate, bidirectional=True)
        # self.attn = None
        # if self_attn:
        #     # self.attn = AttentionLayer(params['rnn_size'], params['rnn_size'], params['rnn_size'])
        #     self.attn = MultiHeadAttentionLayer(rnn_size, nb_heads, dropout_rate)

        self.fnn_list = torch.nn.ModuleList()
        if nb_rnn_layers and nb_fnn_layers:
            for fc_cnt in range(nb_fnn_layers):
                self.fnn_list.append(
                    torch.nn.Linear(fnn_size if fc_cnt else rnn_size, fnn_size, bias=True)
                )

        # for identification or localization ide or loc <<<==============
        if task_type == 'ide' or task_type == 'loc':
            self.fnn_list.append(
                torch.nn.Linear(fnn_size if nb_fnn_layers else rnn_size, out_shape[-1], bias=True),
            )

        # for identification and localization with two branches ide&loc <<<==============
        if task_type == 'ideloc':
            self.ide_fnn_list = torch.nn.ModuleList()
            self.loc_fnn_list = torch.nn.ModuleList()
            self.ide_fnn_list.append(
                torch.nn.Linear(fnn_size if nb_fnn_layers else rnn_size, out_shape_ide[-1], bias=True),
            )
            self.loc_fnn_list.append(
                torch.nn.Linear(fnn_size if nb_fnn_layers else rnn_size, out_shape_loc[-1], bias=True),
            )

        # for ACCDOA representation based accdoa_ide&loc  <<<==============
        if task_type == 'accil':
            self.fnn_list.append(
                torch.nn.Linear(fnn_size if nb_fnn_layers else rnn_size, out_shape[-1], bias=True),
            )

        # # fc layer network
        # fc = [nn.Dropout(0.1)]
        # fc.append(nn.Linear(128, 64))
        # fc.append(nn.Linear(64, 40))
        # self.tail = nn.Sequential(*fc)

    def forward(self, input_fea):
        """
        input: dict: {"spec": spec_fea, "gcc": gcc_fea}
        """
        spec_fea = input_fea['spec']  # (batch_size x freq_dim x time_dim x channel_num_1)  64 x 64 x 4
        gcc_fea = input_fea['gcc']  # (batch_size x freq_dim x time_dim x channel_num_1)  64 x 64 x 6

        # (batch_size x channel_num x freq_dim x time_dim)  ==>> batch_size x (ch_num=ch1+ch2) x 64 x 64
        x = torch.cat((spec_fea.permute(0, 3, 1, 2).contiguous(), gcc_fea.permute(0, 3, 1, 2).contiguous()), 1)

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_map_num, mel_bins, time_steps)'''

        x = x.transpose(1, 3).contiguous()  # (batch_size, time_steps, mel_bins, feature_map_num)
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()  # (batch_size, time_steps, mel_bins x feature_map_num)

        # None is for initial hidden state
        x, ht = self.gru(x, None)  # r_out shape (batch, time_step, output_size)

        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]  # (batch_size, time_steps, hidden_size)

        # if self.attn is not None:
        #     x = self.attn.forward(x, x, x)
        #     # out - batch x hidden x seq
        #     x = torch.tanh(x)

        x = x[:, -1, :]  # last output
        # x = torch.mean(x, dim=1)  # mean_output

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)

        if self.task == 'ide':
            identity = self.fnn_list[-1](x)
            return identity

        elif self.task == 'loc':
            location = self.fnn_list[-1](x)
            return location.squeeze(-1)

        elif self.task == 'ideloc':
            identity = self.ide_fnn_list[-1](x)
            location = self.loc_fnn_list[-1](x)
            # concatenate identification and localization results
            output = torch.concat((identity, location), dim=1)
            return output

        elif self.task == 'accil':
            # ACCDOA output
            output = self.fnn_list[-1](x)
            return output


