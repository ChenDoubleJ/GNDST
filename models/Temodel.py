import torch
from torch import nn
from TSTransformer.config import parse_signal_args
from typing import Callable, Optional
from torch import Tensor
import torch.nn.functional as F
from help import *


class mMultiHeadSelfAttention(nn.Module):
    def __init__(self, config, feature_dim, d_model, n_heads):
        super(mMultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(feature_dim, d_model)
        self.W_k = nn.Linear(feature_dim, d_model)
        self.W_v = nn.Linear(feature_dim, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # Reshape the input to split into multiple heads
        x = x.view(x.size(0), -1, self.n_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        batch_size, samples, seq_len, _ = x.size()
        x = torch.reshape(x, (batch_size*samples, seq_len, -1))

        # Linearly transform input to query, key, and value
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Split the heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, v)

        # Reshape and combine the heads
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size*samples, seq_len, self.d_model)

        # Linearly transform the combined heads
        output = self.W_o(output)
        output = torch.reshape(output, (batch_size, samples, seq_len, self.d_model))
        weight = F.softmax(output, dim=2)
        output = torch.sum(output * weight, dim=2)

        return output


class Model(nn.Module):
    def __init__(self, configs, act: str = "gelu", res_attention: bool = True, head_type='flatten'):
        super().__init__()

        # load parameters
        c_in = configs.enc_in  # 4
        context_window = configs.seq_len  # 输入序列长度
        target_window = configs.pred_len  # 预测序列长度

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model  # 每个patch的编码维度
        d_ff = configs.d_ff
        dropout = configs.dropout
        # fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        patch_len = configs.patch_len  # patch的长度
        stride = configs.stride  # 相邻两个patch之间的长度
        padding_patch = configs.padding_patch  # 填充的方式“end”

        # self.args = configs
        #
        # self.liner1 = nn.Linear(4 * configs.seq_len, 2 * configs.seq_len)
        # self.liner2 = nn.Linear(2 * configs.seq_len, configs.seq_len)
        # self.liner3 = nn.Linear(configs.seq_len, 2)

        self.model = PatchTST_backbone(configs, c_in=c_in, context_window=context_window, target_window=target_window,
                                       patch_len=patch_len, stride=stride,
                                       n_layers=n_layers, d_model=d_model,
                                       n_heads=n_heads, d_ff=d_ff,
                                       dropout=dropout, act=act,
                                       res_attention=res_attention,
                                       head_dropout=head_dropout,
                                       padding_patch=padding_patch,
                                       head_type=head_type)

    def forward(self, x):  # x: [Batch, n, Input length, Channel]

        x = x.permute(0, 1, 3, 2)  # x: [Batch, n, Channel, Input length]
        # print(x.size())
        x = self.model(x)  # b x n x d_model

        return x


# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, configs, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 n_layers: int = 3, d_model=128, n_heads=8,
                 d_ff: int = 512, dropout: float = 0.,
                 act: str = "gelu", res_attention: bool = True,
                 head_dropout=0, padding_patch=None,
                 head_type='flatten'):

        super().__init__()

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)  # patch的数量
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad2d(
                (0, stride, 0, 0))  # 进行填充处理0表示左边不填充，stride表示在右边复制填充 思维padding_layer = nn.ReplicationPad1d((0, 2, 0, 0))
            patch_num += 1
        self.mhsa = mMultiHeadSelfAttention(config=configs, feature_dim=4, d_model=configs.d_model, n_heads=configs.de_heads)
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                    dropout=dropout, act=act,
                                    res_attention=res_attention)
        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        # self.pretrain_head = pretrain_head  # False
        self.head_type = head_type  # flatten
        # self.individual = individual  # 是否使用独立头 False

        # if self.pretrain_head:
        #     self.head = self.create_pretrain_head(self.head_nf, c_in,
        #                                           fc_dropout)  # custom head passed as a partial func with all its kwargs
        if head_type == 'flatten':
            self.head = Flatten_Head(configs, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)

    def forward(self, z):  # z: [bs x nvars x seq_len]

        # do patching 进行patch操作 并且尾部填充
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)  # 进行尾部填充
        # 进行分片，在最后一个维度上，每个patch的长度和每次滑动的长度stride
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x n x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 2, 4, 3)  # z: [bs x n x nvars x patch_len x patch_num]  # 分patch结束
        z = self.backbone(z)  # [bs x n x nvars x d_model x patch_num]
        z = self.head(z)  # [bs x n x target_window x nvars]
        z = self.mhsa(z)  # [b, n, d_model]

        return z


class Flatten_Head(nn.Module):
    def __init__(self, configs, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.n_vars = n_vars
        self.liner1 = nn.Linear(configs.seq_len * n_vars, configs.seq_len)
        # self.liner2 = nn.Linear(int(configs.seq_len * n_vars/2), configs.seq_len)
        self.liner3 = nn.Linear(configs.seq_len, 2)

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.conv1 = nn.Conv1d(nf, target_window, 1)
        self.conv2 = nn.Conv1d(configs.seq_len * n_vars, int(n_vars * target_window / 2), 1)
        self.conv3 = nn.Conv1d(int(n_vars * target_window / 2), target_window, 1)
        self.conv4 = nn.Conv1d(configs.seq_len, int(configs.seq_len / 2), 1)
        self.conv5 = nn.Conv1d(int(configs.seq_len / 2), 2, 1)
        self.linear5 = nn.Linear(int(configs.seq_len / 2), 2)
        self.act = nn.ReLU()
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()
        self.dropout = nn.Dropout(head_dropout)
        self.bn1 = nn.BatchNorm1d(target_window)
        self.bn2 = nn.BatchNorm1d(int(n_vars * target_window / 2))
        self.bn3 = nn.BatchNorm1d(target_window)
        self.bn4 = nn.BatchNorm1d(int(target_window / 2))
        # self.bn3 = nn.BatchNorm1d(target_window)

    def forward(self, x):  # x: [bs x n x nvars x d_model x patch_num]
        x = self.flatten(x)  # [b, n, m, d]
        b = x.shape[0]
        samples = x.shape[1]
        x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[3], x.shape[2]))
        # x = x.permute(0, 2, 1)
        # bs = x.size(0)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = torch.reshape(x, (b, samples, x.shape[1], x.shape[2]))
        return x

class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len,
                 n_layers=3, d_model=128, n_heads=8,
                 d_ff=256, norm='BatchNorm', dropout=0., act="gelu",
                 res_attention=True, ):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        # self.W_P = nn.Conv1d(patch_len, d_model, 1)
        self.seq_len = q_len
        pe = 'zeros'
        learn_pe = True
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_ff=d_ff, norm=norm,
                                  dropout=dropout,
                                  activation=act, res_attention=res_attention, n_layers=n_layers)

    def forward(self, x) -> Tensor:  # x: [bs x n x nvars x patch_len x patch_num]

        n_vars = x.shape[2]  # 通道数（变量数）
        bs = x.shape[0]
        pn = x.shape[3]
        samples = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 2, 4, 3)  # x: [bs x n x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x n x nvars x patch_num x d_model]

        u = torch.reshape(x, (
            x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))  # u: [bs * n * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * n * nvars x patch_num x d_model] 进行编码处理线性层加上偏置
        u = self.dropout(u)
        # Encoder
        z = self.encoder(u)  # z: [bs * n * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, samples, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 2, 4, 3)  # z: [bs x nvars x d_model x patch_num]

        return z

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                             dropout=dropout,
                             activation=activation, res_attention=res_attention) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor):
        output = src
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                             dropout=dropout,
                             activation=activation, res_attention=res_attention) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor):
        output = src
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_ff=256,
                 norm='BatchNorm', dropout=0., bias=True, activation="gelu", res_attention=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v,
                                             proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = False
        self.store_attn = False

    def forward(self, src: Tensor) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, proj_dropout=0.,
                 qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, n_heads, res_attention=False, lsa=False):
        super().__init__()
        # attn_dropout = 0.
        # self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        # attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


if __name__ == "__main__":
    config = parse_signal_args()
    model = Model(config).cuda()
    input = torch.randn([2, 10, 10, 4]).cuda()  # [b, n, s, m]
    out = model(input)
    print(out.size())
