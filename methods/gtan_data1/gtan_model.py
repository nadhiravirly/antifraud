import torch
import torch.nn as nn
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np


class PosEncoding(nn.Module):
    def __init__(self, dim, device, base=10000, bias=0):
        super(PosEncoding, self).__init__()
        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


class TransEmbedding(nn.Module):
    def __init__(self, df=None, device='cpu', dropout=0.2, in_feats=142, cat_features=None):
        """
        Inisialisasi embedding fitur kategori untuk DATA1.
        """
        super(TransEmbedding, self).__init__()
        self.time_pe = PosEncoding(dim=in_feats, device=device, base=100)
        self.cat_table = nn.ModuleDict({
            col: nn.Embedding(max(df[col].unique()) + 1, in_feats).to(device)
            for col in cat_features if col not in {"is_fraud", "Time"}
        })
        self.label_table = nn.Embedding(3, in_feats, padding_idx=2).to(device)
        self.dropout = nn.Dropout(dropout)
        self.cat_features = cat_features
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats, in_feats) for _ in cat_features]
        )

    def forward_emb(self, df):
        support = {
            col: self.cat_table[col](df[col]) for col in self.cat_features if col not in {"is_fraud", "Time"}
        }
        return support

    def forward(self, df):
        support = self.forward_emb(df)
        output = 0
        for i, k in enumerate(support.keys()):
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            output = output + support[k]
        return output


class TransformerConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
        """
        Inisialisasi layer Transformer dengan parameter yang disesuaikan untuk GTAN.
        """
        super(TransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        self.lin_query = nn.Linear(self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        self.lin_key = nn.Linear(self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        self.lin_value = nn.Linear(self._in_src_feats, self._out_feats * self._num_heads, bias=bias)

        if skip_feat:
            self.skip_feat = nn.Linear(self._in_src_feats, self._out_feats * self._num_heads, bias=bias)
        else:
            self.skip_feat = None
        if gated:
            self.gate = nn.Linear(3 * self._out_feats * self._num_heads, 1, bias=bias)
        else:
            self.gate = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats * self._num_heads)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        graph = graph.local_var()
        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('Ada node dengan degree 0 di graf.')

        h_src = feat
        h_dst = h_src[:graph.number_of_dst_nodes()]

        q_src = self.lin_query(h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(h_src).view(-1, self._num_heads, self._out_feats)

        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'] / self._out_feats**0.5)
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'), fn.sum('attn', 'agg_u'))
        rst = graph.dstdata['agg_u'].reshape(-1, self._out_feats * self._num_heads)

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            if self.gate is not None:
                gate = torch.sigmoid(self.gate(torch.cat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        return rst


class GraphAttnModel(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 drop=None,
                 ref_df=None,
                 cat_features=None,
                 device='cpu'):
        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        self.input_drop = nn.Dropout(drop[0])
        self.output_drop = nn.Dropout(drop[1])

        self.n2v_mlp = TransEmbedding(ref_df, device=device, in_feats=in_feats, cat_features=cat_features)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(n_classes+1, in_feats, padding_idx=n_classes))
        self.layers.append(nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        self.layers.append(nn.Linear(self.in_feats, self.hidden_dim * self.heads[0]))
        self.layers.append(nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim * self.heads[0]),
            nn.PReLU(),
            nn.Dropout(drop[1]),
            nn.Linear(self.hidden_dim * self.heads[0], in_feats)
        ))

        self.layers.append(TransformerConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           num_heads=self.heads[0],
                                           skip_feat=skip_feat,
                                           gated=gated,
                                           layer_norm=layer_norm,
                                           activation=self.activation))

        for l in range(1, self.n_layers):
            self.layers.append(TransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                               out_feats=self.hidden_dim,
                                               num_heads=self.heads[l],
                                               skip_feat=skip_feat,
                                               gated=gated,
                                               layer_norm=layer_norm,
                                               activation=self.activation))

        self.layers.append(nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
            nn.BatchNorm1d(self.hidden_dim * self.heads[-1]),
            nn.PReLU(),
            nn.Dropout(drop[1]),
            nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)
        ))

    def forward(self, blocks, features, labels, n2v_feat=None):
        h = features if n2v_feat is None else features + self.n2v_mlp(n2v_feat)

        label_embed = self.input_drop(self.layers[0](labels))
        label_embed = self.layers[1](h) + self.layers[2](label_embed)
        label_embed = self.layers[3](label_embed)
        h = h + label_embed

        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l + 4](blocks[l], h))

        logits = self.layers[-1](h)
        return logits
