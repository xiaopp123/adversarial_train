# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self,
                 vocab_size=100000,
                 num_filter=256,
                 embedding_dim=128,
                 filter_size_list=[2, 3, 4],
                 dropout=0.5,
                 num_classes=2):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=vocab_size-1)
        self.conv_2d_list = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=num_filter,
                       kernel_size=(t, embedding_dim))
             for t in filter_size_list])
        self.conv_3d_list = nn.ModuleList(
            [nn.Conv3d(in_channels=1, out_channels=num_filter,
                       kernel_size=(t, embedding_dim), stride=[1, 1], padding=[0, 0], dilation=[1, 1])
             for t in filter_size_list]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filter * len(filter_size_list), num_classes)

    def get_embedding(self, x):
        out = self.embedding(x[0])
        return out

    def forward(self, x, emb_init=None):
        # [B, L, D]
        # out = self.embedding(x[0])
        if emb_init is not None:
            out = emb_init
        else:
            out = self.get_embedding(x)
        # [B, L, D] -> [B, 1, L, D]
        out = out.unsqueeze(1)
        out_conv_list = []
        for conv in self.conv_2d_list:
            out_conv_res = F.relu(conv(out)).squeeze(3)
            out_pool_res = F.max_pool1d(out_conv_res, out_conv_res.size(2)).squeeze(2)
            out_conv_list.append(out_pool_res)

        out = torch.cat(out_conv_list, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def main():
    pass


if __name__ == '__main__':
    main()
