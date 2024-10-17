from torch import nn

from graphtransformer import Encoder
from .Temodel import Model
from .Decoding import MultiHeadAttention
import torch

from TSTransformer.config import parse_signal_args


class TSTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temporal_model = Model(config)
        self.spatial_model = Encoder(d_model=128, ffn_hidden=128, n_head=4, n_layers=2,
                                     drop_prob=0.0)
        self.decode = MultiHeadAttention(d_model=128, n_head=4)
        self.conv1 = nn.Conv1d(config.d_model, config.seq_len, 1)
        self.conv2 = nn.Conv1d(config.seq_len, config.seq_len // 2, 1)
        self.conv3 = nn.Conv1d(config.seq_len // 2, 2, 1)
        self.act = nn.ReLU()
        self.act1 = nn.Sigmoid()
        # self.act2 = nn.Tanh()
        # self.dropout = nn.Dropout(head_dropout)
        # self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(config.seq_len // 2)
    def forward(self, spatial_feature, dis, temporal_feature, sys):
        temporal_feature = temporal_feature.permute(0, 2, 1, 3)
        temporal_feature = self.temporal_model(temporal_feature)
        spatial_feature = self.spatial_model(spatial_feature, dis, sys, src_mask=None)
        out = self.decode(temporal_feature, spatial_feature, spatial_feature)
        # print(out.size())
        # batch_size = out.size(0)
        output = out.permute(0, 2, 1)
        output = self.conv1(output)
        # output = self.bn1(output)
        output = self.act(output)
        # output = output.view(batch_size, -1, 1)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.act(output)
        out = self.conv3(output)
        return out


if __name__ == "__main__":
    config = parse_signal_args()
    x1 = torch.randn([2, 10, 4]).cuda()
    x2 = torch.randn([2, 10, 10]).cuda()
    x3 = torch.randn([2, 10, 10, 4]).cuda()
    # adj_matrix = torch.randint(2, (10, 10)).cuda()
    model = TSTModel(config).cuda()
    out = model(x1, x2, x3)
    print(out.size())
