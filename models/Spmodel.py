import os
import time
import re
import numpy as np
from TSTransformer.config import parse_signal_args
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import BatchNorm
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch import nn
class GTNN(torch.nn.Module):
    def __init__(self, configs):
        super(GTNN, self).__init__()

        torch.cuda.manual_seed(123)
        # self.fc = Linear(configs.enc_in, 32)
        self.fc = nn.Conv1d(configs.enc_in, 32, 1)
        self.conv1 = TransformerConv(32, 8, heads=4, concat=True, beta=False, dropout=0.3)
        self.bn1 = BatchNorm(32)
        self.conv2 = TransformerConv(32, 8, heads=4, concat=True, beta=False, dropout=0.3)
        self.bn2 = BatchNorm(32)
        self.conv3 = TransformerConv(32, 32, heads=1, concat=True, beta=False, dropout=0.3)
        self.bn3 = BatchNorm(32)
        self.conv4 = TransformerConv(64, 32, heads=1, concat=True, beta=False, dropout=0.3)
        self.bn4 = BatchNorm(32)
        self.conv5 = TransformerConv(64, 2, heads=1, concat=True, beta=False, dropout=0.)

    # 5层图网络
    def forward(self, data, aj):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = input
        edge_index = aj
        x = self.fc(x)
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        x4 = self.conv4(torch.cat([x1, x3], -1), edge_index)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)

        x5 = self.conv5(torch.cat([x2, x4], -1), edge_index)
        # 分类层
        out = F.softmax(x5)
        return out

if __name__ == "__main__":
    config = parse_signal_args()
    input = torch.randn([2, 4, 10]).cuda()
    adj_matrix = torch.randint(2, (10, 10)).cuda()
    model = GTNN(config).cuda()
    out = model(input, adj_matrix)
    print(out.size())
