"""Torch modules for graph attention networks(GAT)."""
import torch

# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair


# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        self.hidden = 64

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                in_feats, self.hidden, bias=False)
            self.fc2 = nn.Linear(
                self.hidden, out_feats, bias=False)

        self.weight_for_agg = nn.Parameter(th.FloatTensor(edge_feats, edge_feats))
        # self.connect = nn.Parameter(th.FloatTensor(265694, edge_feats))

        # self.w0 = nn.Parameter(th.FloatTensor(size=(edge_feats, edge_feats)))
        # self.w1 = nn.Parameter(th.FloatTensor(size=(edge_feats, edge_feats)))
        # self.w2 = nn.Parameter(th.FloatTensor(size=(edge_feats, edge_feats)))
        # self.w3 = nn.Parameter(th.FloatTensor(size=(edge_feats, edge_feats)))
        # self.w4 = nn.Parameter(th.FloatTensor(size=(edge_feats, edge_feats)))
        # self.w5 = nn.Parameter(th.FloatTensor(size=(edge_feats, edge_feats)))
        # self.w6 = nn.Parameter(th.FloatTensor(size=(edge_feats, edge_feats)))
        # self.w7 = nn.Parameter(th.FloatTensor(size=(edge_feats, edge_feats)))
        # self.w8 = nn.Parameter(th.FloatTensor(size=(edge_feats, edge_feats)))


        # self.transform_funcs = {
        #     0: self.w0,
        #     1: self.w1,
        #     2: self.w2,
        #     3: self.w3,
        #     4: self.w4,
        #     5: self.w5,
        #     6: self.w6,
        #     7: self.w7,
        #     8: self.w8
        # }

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)

        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)

        nn.init.xavier_normal_(self.weight_for_agg.data, gain=gain)
        # nn.init.xavier_normal_(self.connect.data, gain=gain)

        # nn.init.xavier_normal_(self.w0, gain=gain)
        # nn.init.xavier_normal_(self.w1, gain=gain)
        # nn.init.xavier_normal_(self.w2, gain=gain)
        # nn.init.xavier_normal_(self.w3, gain=gain)
        # nn.init.xavier_normal_(self.w4, gain=gain)
        # nn.init.xavier_normal_(self.w5, gain=gain)
        # nn.init.xavier_normal_(self.w6, gain=gain)
        # nn.init.xavier_normal_(self.w7, gain=gain)
        # nn.init.xavier_normal_(self.w8, gain=gain)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = h_src
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            e_embed = self.edge_emb(e_feat)
            graph.srcdata.update({'ft': feat_src})
            graph.dstdata.update({'fd': feat_dst})
            e_f = e_embed.matmul(self.weight_for_agg)
            # e_f = e_embed
            # e_f = self.feat_drop(e_f)
            # transform_funcs = self.transform_funcs
            # for node_type in transform_funcs:
            #     nodes = (e_feat == node_type)
            #     transform_func = transform_funcs[node_type]
            #     e_f[nodes] = e_f[nodes].matmul(transform_func)
            e_feature = self.leaky_relu(e_f)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph,  e_feature))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1 - self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._out_feats)
                rst = self.fc(rst).view(
                    rst.shape[0], self.hidden)
                rst = self.fc2(rst).view(
                    rst.shape[0], self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)

            return rst, graph.edata.pop('a').detach()
