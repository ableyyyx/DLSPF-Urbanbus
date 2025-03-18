import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class MultiGraphFusion(nn.Module):
    """多图融合模块"""

    def __init__(self):
        super().__init__()
        self.weight_gen = nn.Sequential(
            nn.Linear(3, 32),  # 输入维度修正为3
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, A_dist, A_trans, A_dyn):
        # 计算各图的全局均值 [batch_size]
        mean_dist = A_dist.mean(dim=(1, 2)).unsqueeze(1)  # [B, 1]
        mean_trans = A_trans.mean(dim=(1, 2)).unsqueeze(1)
        mean_dyn = A_dyn.mean(dim=(1, 2)).unsqueeze(1)

        # 特征拼接 [B, 3]
        features = torch.cat([mean_dist, mean_trans, mean_dyn], dim=1)

        # 生成融合权重 [B, 3]
        weights = F.softmax(self.weight_gen(features), dim=1)

        # 维度调整后进行加权融合
        return (weights[:, 0].view(-1, 1, 1) * A_dist +
                weights[:, 1].view(-1, 1, 1) * A_trans +
                weights[:, 2].view(-1, 1, 1) * A_dyn)


class LineAGC(nn.Module):
    """参数自适应图卷积"""

    def __init__(self, in_dim, out_dim, num_nodes):
        super().__init__()
        self.route_weights = nn.Parameter(torch.randn(num_nodes, in_dim, out_dim))
        self.route_bias = nn.Parameter(torch.randn(1, num_nodes, out_dim))

    def forward(self, x, adj):
        B, T, M, _ = x.shape
        x = x.view(B * T, M, -1)
        # 修改处：使用 repeat_interleave 将 adj 扩展到 [B*T, M, M]
        adj_expanded = adj.repeat_interleave(T, dim=0)
        h = torch.bmm(adj_expanded, x)
        h = torch.einsum('bmd,mdl->bml', h, self.route_weights) + self.route_bias
        return h.view(B, T, M, -1)


class STSA(nn.Module):
    """时空注意力模块"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads)

    def forward(self, x, route_neighbors):
        B, T, M, D = x.shape
        # 时间注意力
        x_t = x.permute(1, 0, 2, 3).reshape(T, B * M, D)
        t_out, _ = self.temporal_attn(x_t, x_t, x_t)
        t_out = t_out.view(T, B, M, D).permute(1, 0, 2, 3)

        # 空间注意力
        s_out = []
        for i in range(M):
            neighbors = route_neighbors[i]
            if len(neighbors) == 0:
                s_out.append(torch.zeros_like(x[:, :, i]))
                continue
            # 处理邻居数据
            k = x[:, :, neighbors].permute(1, 0, 2, 3).reshape(T, B * len(neighbors), D)
            q = x[:, :, i].permute(1, 0, 2).reshape(T, B, D)
            attn_out, _ = self.spatial_attn(q, k, k)
            s_out.append(attn_out.view(T, B, D))
        s_out = torch.stack(s_out, dim=2).permute(1, 0, 2, 3)
        return t_out + s_out


class LinkGCRN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # 数据参数
        self.num_nodes = data_feature['num_nodes']
        self.feature_dim = data_feature['feature_dim']
        self.output_dim = data_feature['output_dim']

        # 图数据初始化（增强鲁棒性），注册为 buffer 以确保设备一致性
        self.register_buffer('A_dist', self._init_adj(data_feature.get('A_distance')))
        self.register_buffer('A_trans', self._init_adj(data_feature.get('A_transfer')))
        self.register_buffer('A_dyn', self._init_adj(data_feature.get('A_dynamic')))

        # 邻居关系初始化
        self.route_neighbors = self._init_neighbors(data_feature)

        # 网络模块
        self.graph_fusion = MultiGraphFusion()
        self.route_conv = LineAGC(self.feature_dim, 64, self.num_nodes)
        self.sts_attn = STSA(64, n_heads=4)
        self.fc = nn.Linear(64, self.output_dim)

    def _init_adj(self, adj):
        """邻接矩阵初始化"""
        if adj is None:
            return torch.eye(self.num_nodes)
        if isinstance(adj, np.ndarray):
            return torch.FloatTensor(adj)
        return adj

    def _init_neighbors(self, data_feature):
        """邻居关系初始化"""
        if 'route_neighbors' in data_feature:
            return data_feature['route_neighbors']
        # 基于距离矩阵自动生成
        adj_np = self.A_dist.cpu().numpy() if isinstance(self.A_dist, torch.Tensor) else self.A_dist
        return [np.where(row > np.mean(row))[0].tolist() for row in adj_np]

    def forward(self, batch):
        # 输入数据 [B, T_in, M, D]
        x = batch['X']

        # 图融合（注意：A_* 已注册为 buffer，会自动跟随模型迁移）
        A_final = self.graph_fusion(
            self.A_dist.unsqueeze(0).repeat(x.size(0), 1, 1),
            self.A_trans.unsqueeze(0).repeat(x.size(0), 1, 1),
            self.A_dyn.unsqueeze(0).repeat(x.size(0), 1, 1)
        )

        # 特征提取
        h = F.relu(self.route_conv(x, A_final))
        h = self.sts_attn(h, self.route_neighbors)
        return self.fc(h[:, -1:, :, :])

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_pred = self.predict(batch)
        return loss.masked_mae_torch(y_pred, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
