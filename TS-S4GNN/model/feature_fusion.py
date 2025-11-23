from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import (
    GINEConv,
    GATv2Conv,
    SAGEConv,
)
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch_geometric
import scipy
import math
from model.graph_learner import *
from model.s4 import S4Model
from model.decoders import SequenceDecoder

def calculate_cosine_decay_weight(max_weight, epoch, epoch_total, min_weight=0):
   
    curr_weight = min_weight + \
        0.5 * (max_weight - min_weight) * (1 + math.cos(epoch / epoch_total  * math.pi))
    return curr_weight

def calculate_normalized_laplacian(adj):
    

    batch, num_nodes, _ = adj.shape
    d = adj.sum(-1)  # (batch, num_nodes)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # (batch, num_nodes, num_nodes)

    identity = (torch.eye(num_nodes).unsqueeze(0).repeat(batch, 1, 1)).to(
        adj.device
    )  # (batch, num_nodes, num_nodes)
    normalized_laplacian = identity - torch.matmul(
        torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt
    )

    return normalized_laplacian


def feature_smoothing(adj, X):

    # normalized laplacian
    L = calculate_normalized_laplacian(adj)

    feature_dim = X.shape[-1]
    mat = torch.matmul(torch.matmul(X.transpose(1, 2), L), X) / (feature_dim**2)
    loss = mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
    return loss


def get_knn_graph(x, k, dist_measure="cosine", undirected=True):

    if dist_measure == "euclidean":
        dist = torch.cdist(x, x, p=2.0)
        dist = (dist - dist.min()) / (dist.max() - dist.min())
        knn_val, knn_ind = torch.topk(
            dist, k, dim=-1, largest=False
        )  # smallest distances
    elif dist_measure == "cosine":
        norm = torch.norm(x, dim=-1, p="fro")[:, :, None]
        x_norm = x / norm
        dist = torch.matmul(x_norm, x_norm.transpose(1, 2))
        knn_val, knn_ind = torch.topk(
            dist, k, dim=-1, largest=True
        )  # largest similarities
    else:
        raise NotImplementedError

    adj_mat = (torch.ones_like(dist) * 0).scatter_(-1, knn_ind, knn_val).to(x.device)

    adj_mat = torch.clamp(adj_mat, min=0.0)  # remove negatives

    if undirected:
        adj_mat = (adj_mat + adj_mat.transpose(1, 2)) / 2

    # add self-loop
    I = (
        torch.eye(adj_mat.shape[-1], adj_mat.shape[-1])
        .unsqueeze(0)
        .repeat(adj_mat.shape[0], 1, 1)
        .to(bool)
    ).to(x.device)
    adj_mat = adj_mat * (~I) + I

    # to sparse graph
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_mat)

    return edge_index, edge_weight, adj_mat


def prune_adj_mat(adj_mat, num_nodes, method="thresh", edge_top_perc=None, knn=None, thresh=None):
    
    if method == "thresh":
        sorted, indices = torch.sort(
            adj_mat.reshape(-1, num_nodes * num_nodes),
            dim=-1,
            descending=True,
        )
        K = int((num_nodes**2) * edge_top_perc)
        mask = adj_mat > sorted[:, K].unsqueeze(1).unsqueeze(2)
        adj_mat = adj_mat * mask
    elif method == "knn":
        knn_val, knn_ind = torch.topk(
            adj_mat, knn, dim=-1, largest=True
        )
        adj_mat = (torch.ones_like(adj_mat) * 0).scatter_(-1, knn_ind, knn_val).to(adj_mat.device)
    elif method == "thresh_abs":
        mask = (adj_mat > thresh).float()
        adj_mat = adj_mat * mask
    else:
        raise NotImplementedError

    return adj_mat

class TemporalConv(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_chans, out_chans, kernel_size=15, stride=1, padding=7)
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv1d(out_chans, out_chans, kernel_size=3, padding=1)
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv1d(out_chans, out_chans, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        # 输入形状: (B*N, seq_len, 1)
        B, seq_len, _ = x.shape        #batch*nodes,seq_len,1
        
        # 调整形状以适应1D卷积层
        x = rearrange(x, 'B T C -> B C T')
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C T -> B T C')
        return x
    
class Fusion(nn.Module):
    def __init__(
        self,
        input_dim,
        num_nodes,
        dropout,
        num_temporal_layers,
        g_conv,
        num_gnn_layers,
        hidden_dim,
        max_seq_len,
        resolution,
        state_dim=64,
        channels=1,
        temporal_model="s4",
        bidirectional=False,
        prenorm=False,
        postact=None,
        metric="self_attention",
        adj_embed_dim=10,
        gin_mlp=False,
        train_eps=False,
        prune_method="thresh",
        edge_top_perc=0.5,
        thresh=None,
        temporal_pool="mean",
        graph_pool="sum",
        activation_fn="relu",
        num_classes=1,            
        undirected_graph=True,
        K=3,
        regularizations=["feature_smoothing", "degree", "sparse"],
        residual_weight=0.0,
        decay_residual_weight=False,
        cnn_out_chans=8,                                                   
        **kwargs
    ):
        super().__init__()
        #序列长度需要整除resolution，每resolution个时间长度里得到一个图
        if (resolution is not None) and (max_seq_len % resolution != 0):
            raise ValueError("max_seq_len must be divisible by resolution!")


        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.edge_top_perc = edge_top_perc
        self.graph_pool = graph_pool
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.metric = metric
        self.undirected_graph = undirected_graph
        self.K = K
        self.regularizations = regularizations
        self.residual_weight = residual_weight
        self.temporal_pool = temporal_pool
        self.temporal_model = temporal_model
        self.max_seq_len = max_seq_len
        self.resolution = resolution
        self.prune_method = prune_method
        self.thresh = thresh
        self.decay_residual_weight = decay_residual_weight

        self.cnn_out_chans = cnn_out_chans
        self.cnn = TemporalConv(in_chans=1, out_chans=cnn_out_chans)
       

        # 添加一个线性层将CNN输出映射到S4的输入维度
        self.cnn_proj = nn.Linear(cnn_out_chans, input_dim)



        # temporal layer
        if temporal_model == "s4":
            self.t_model = S4Model(
                d_input=input_dim,
                d_model=hidden_dim,
                d_state=state_dim,
                channels=channels,
                n_layers=num_temporal_layers,
                dropout=dropout,
                prenorm=prenorm,
                l_max=max_seq_len,
                bidirectional=bidirectional,
                postact=postact,
                add_decoder=False,
                pool=False,
                temporal_pool=None,
            )

        else:
            raise NotImplementedError

        # graph learning layer
        self.attn_layers = GraphLearner(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_nodes=num_nodes,
            embed_dim=adj_embed_dim,
            metric_type=metric,
        )

        # gnn layers
        self.gnn_layers = nn.ModuleList()
        if g_conv == "graphsage":
            for _ in range(num_gnn_layers):
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim, **kwargs))
        elif g_conv == "gine":
            for _ in range(num_gnn_layers):
                if gin_mlp:
                    gin_nn = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                else:
                    gin_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
                self.gnn_layers.append(
                    GINEConv(
                        nn=gin_nn, eps=0.0, train_eps=train_eps, edge_dim=1, **kwargs
                    )
                )
        else:
            raise NotImplementedError

        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_fn == "elu":
            self.activation = nn.ELU()
        elif activation_fn == "gelu":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim*2, num_classes)

    def forward(
        self, 
        data, 
        return_attention=False, 
        lengths=None,
        epoch=None,
        epoch_total=None,
    ):
       
        x = data.x        #导入脑电信号，形状为（batch*电极数，序列长度，1）
        batch = x.shape[0] // self.num_nodes
        num_nodes = self.num_nodes
        _, seq_len, _ = x.shape
        batch_idx = data.batch

        if lengths is not None:
            lengths = torch.repeat_interleave(lengths, num_nodes, dim=0)


        x_cnn = self.cnn(x)       # CNN 经过TemporalConv处理
        x = self.cnn_proj(x_cnn)  # 投影到S4输入维度

        # temporal layer 经过s4层
        if self.temporal_model == "s4":
            x = self.t_model(x, lengths)  # (batch * num_nodes, seq_len, hidden_dim)
        else:
            if lengths is not None:
                x = pack_padded_sequence(x, lengths, batch_first=True)
            x, _ = self.t_model(x)
            if lengths is not None:
                x, lengths = pad_packed_sequence(x)

        # get output with <resolution> as interval
        if lengths is None:                                              #对于固定长度
            x = x.view(
                batch, num_nodes, seq_len, -1                            #重塑
            )  # (batch, num_nodes, seq_len, hidden_dim)
            x_tmp = []
            num_dynamic_graphs = self.max_seq_len // self.resolution     #计算图数量
            for t in range(num_dynamic_graphs):
                start = t * self.resolution
                stop = start + self.resolution
                curr_x = torch.mean(x[:, :, start:stop, :], dim=2)       #对每个时间段特征取平均
                x_tmp.append(curr_x)
            x_tmp = torch.stack(
                x_tmp, dim=1
            )  # (batch, num_dynamic_graphs, num_nodes, hidden_dim)           堆叠
            x = x_tmp.reshape(
                -1, num_nodes, self.hidden_dim 
            )  # (batch * num_dynamic_graphs, num_nodes, hidden_dim)          重塑
            del x_tmp
        else:  # for variable lengths, mean pool over actual lengths
            x = torch.stack(
                [
                    torch.mean(out[:length, :], dim=0)
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
            x = x.reshape(batch, num_nodes, -1)  # (batch, num_nodes, hidden_dim)
            num_dynamic_graphs = 1

        #距离图
        adj_mat_prior = torch_geometric.utils.to_dense_adj(
                edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr
            )
        
        # knn cosine graph
        edge_index_knn, edge_weight_knn, adj_mat_knn = get_knn_graph(
            x,
            self.K,
            dist_measure="cosine",
            undirected=self.undirected_graph,
        )
        edge_index_knn = edge_index_knn.to(x.device)
        edge_weight_knn = edge_weight_knn.to(x.device)
        adj_mat_knn = adj_mat_knn.to(x.device)

         # learn adj mat
        attn_weight = self.attn_layers(
            x
        )  # (batch*num_dynamic_graphs, num_nodes, num_nodes)

        # to undirected 去方向
        if self.undirected_graph:
            adj_mat_prior = (adj_mat_prior + adj_mat_prior.transpose(1, 2)) / 2
            attn_weight = (attn_weight + attn_weight.transpose(1, 2)) / 2
        raw_attn_weight = attn_weight.clone()

        # add residual
        if len(adj_mat_knn.shape) == 2:
            adj_mat_knn = torch.cat([adj_mat_knn] * num_dynamic_graphs * batch, dim=0)
        elif len(adj_mat_knn.shape) == 3 and (adj_mat_knn.shape != attn_weight.shape):
            adj_mat_knn = torch.cat([adj_mat_knn] * num_dynamic_graphs, dim=0)
        
        # knn graph weight (aka residual weight) decay
        if self.decay_residual_weight:
            assert (epoch is not None) and (epoch_total is not None)
            residual_weight = calculate_cosine_decay_weight(
                max_weight=self.residual_weight, epoch=epoch, epoch_total=epoch_total, min_weight=0
            )
        else:
            residual_weight = self.residual_weight
        # add knn graph
        adj_mat_knn = (
            residual_weight * adj_mat_knn + (1 - residual_weight) * attn_weight
        )
        
        # prune graph
        adj_mat_prior = prune_adj_mat(
            adj_mat_prior,
            num_nodes,
            method=self.prune_method,
            edge_top_perc=self.edge_top_perc,
            knn=self.K,
            thresh=self.thresh,
        )
        adj_mat_knn = prune_adj_mat(
            adj_mat_knn,
            num_nodes,
            method=self.prune_method,
            edge_top_perc=self.edge_top_perc,
            knn=self.K,
            thresh=self.thresh,
        )

        # regularization loss 可学习图正则化损失
        reg_losses = self.regularization_loss(x, adj=adj_mat_knn)

        # back to sparse graph
        edge_index_prior, edge_weight_prior = torch_geometric.utils.dense_to_sparse(adj_mat_prior)
        edge_index_knn, edge_weight_knn = torch_geometric.utils.dense_to_sparse(adj_mat_knn)

        # add self-loop
        edge_index_prior, edge_weight_prior = torch_geometric.utils.remove_self_loops(
            edge_index=edge_index_prior, edge_attr=edge_weight_prior
        )
        edge_index_prior, edge_weight_prior = torch_geometric.utils.add_self_loops(
            edge_index=edge_index_prior,
            edge_attr=edge_weight_prior,
            fill_value=1,
        )

        edge_index_knn, edge_weight_knn = torch_geometric.utils.remove_self_loops(
            edge_index=edge_index_knn, edge_attr=edge_weight_knn
        )
        edge_index_knn, edge_weight_knn = torch_geometric.utils.add_self_loops(
            edge_index=edge_index_knn,
            edge_attr=edge_weight_knn,
            fill_value=1,
        )
        
        x_prior = x.view(
            batch * num_dynamic_graphs * num_nodes, -1
        )  # (batch * num_dynamic_graphs * num_nodes, hidden_dim)
        x_knn = x.view(
            batch * num_dynamic_graphs * num_nodes, -1
        )  # (batch * num_dynamic_graphs * num_nodes, hidden_dim)
        # 距离 GNN 层
        for i in range(len(self.gnn_layers)):
            # gnn layer
            x_prior = self.gnn_layers[i](
                x_prior, edge_index=edge_index_prior, edge_attr=edge_weight_prior.reshape(-1, 1)
            )
            x_prior = self.dropout(
                self.activation(x_prior)
            )  # (batch * num_dynamic_graphs * num_nodes, hidden_dim)
        x_prior = x_prior.view(batch * num_dynamic_graphs, num_nodes, -1).view(
            batch, num_dynamic_graphs, num_nodes, -1
        )  # (batch, num_dynamic_graphs, num_nodes, hidden_dim)

        # temporal pool
        if self.temporal_pool == "last":
            x_prior = x_prior[:, -1, :, :]  # (batch, num_nodes, hidden_dim)
        elif self.temporal_pool == "mean":
            x_prior = torch.mean(x_prior, dim=1)
        else:
            raise NotImplementedError
        # 可学习 GNN层
        for i in range(len(self.gnn_layers)):
            # gnn layer
            x_knn = self.gnn_layers[i](
                x_knn, edge_index=edge_index_knn, edge_attr=edge_weight_knn.reshape(-1, 1)
            )
            x_knn = self.dropout(
                self.activation(x_knn)
            )  # (batch * num_dynamic_graphs * num_nodes, hidden_dim)
        x_knn = x_knn.view(batch * num_dynamic_graphs, num_nodes, -1).view(
            batch, num_dynamic_graphs, num_nodes, -1
        )  # (batch, num_dynamic_graphs, num_nodes, hidden_dim)

        # temporal pool
        if self.temporal_pool == "last":
            x_knn = x_knn[:, -1, :, :]  # (batch, num_nodes, hidden_dim)
        elif self.temporal_pool == "mean":
            x_knn = torch.mean(x_knn, dim=1)
        else:
            raise NotImplementedError
        
        # prior graph pool
        if self.graph_pool == "sum":
            x_prior = torch.sum(x_prior, dim=1)  # (batch, hidden_dim)
        elif self.graph_pool == "mean":
            x_prior = torch.mean(x_prior, dim=1)
        elif self.graph_pool == "max":
            x_prior, _ = torch.max(x_prior, dim=1)
        else:
            raise NotImplementedError
        
        # knn graph pool
        if self.graph_pool == "sum":
            x_knn = torch.sum(x_knn, dim=1)  # (batch, hidden_dim)
        elif self.graph_pool == "mean":
            x_knn = torch.mean(x_knn, dim=1)
        elif self.graph_pool == "max":
            x_knn, _ = torch.max(x_knn, dim=1)
        else:
            raise NotImplementedError
        feat = x_knn.clone()

        # classifier
        x_combined = torch.cat([x_knn, x_prior], dim=-1)  # (batch, hidden_dim * 2)
        x = self.classifier(x_combined)
        return x, reg_losses
    
    def regularization_loss(self, x, adj, reduce="mean"):
        """
        Referred to https://github.com/hugochan/IDGL/blob/master/src/core/model_handler.py#L1116
        """
        batch, num_nodes, _ = x.shape
        n = num_nodes

        loss = {}

        if "feature_smoothing" in self.regularizations:
            curr_loss = feature_smoothing(adj=adj, X=x) / (n**2)
            if reduce == "mean":
                loss["feature_smoothing"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["feature_smoothing"] = torch.sum(curr_loss)
            else:
                loss["feature_smoothing"] = curr_loss

        if "degree" in self.regularizations:
            ones = torch.ones(batch, num_nodes, 1).to(x.device)
            curr_loss = -(1 / n) * torch.matmul(
                ones.transpose(1, 2), torch.log(torch.matmul(adj, ones))
            ).squeeze(-1).squeeze(-1)
            if reduce == "mean":
                loss["degree"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["degree"] = torch.sum(curr_loss)
            else:
                loss["degree"] = curr_loss

        if "sparse" in self.regularizations:
            curr_loss = (
                1 / (n**2) * torch.pow(torch.norm(adj, p="fro", dim=(-1, -2)), 2)
            )

            if reduce == "mean":
                loss["sparse"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["sparse"] = torch.sum(curr_loss)
            else:
                loss["sparse"] = curr_loss

        if "symmetric" in self.regularizations and self.undirected_graph:
            curr_loss = torch.norm(adj - adj.transpose(1, 2), p="fro", dim=(-1, -2))
            if reduce == "mean":
                loss["symmetric"] = torch.mean(curr_loss)
            elif reduce == "sum":
                loss["symmetric"] = torch.sum(curr_loss)
            else:
                loss["symmetric"] = curr_loss

        return loss