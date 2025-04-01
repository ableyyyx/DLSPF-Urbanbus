import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class MultiGraphFusion(nn.Module):
    """
    Multi-Graph Fusion Module (MGF)
    Fuse the static (distance, transfer) and dynamic (generated at station level) adjacency matrices
    using two learnable parameters.
    Formula:
      A_final = α * A_dist + β * A_trans + (1-α-β) * A_dyn
    """

    def __init__(self, init_alpha=0.33, init_beta=0.33):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, A_dist, A_trans, A_dyn):
        # A_dist, A_trans, A_dyn: [B, M, M]
        return self.alpha * A_dist + self.beta * A_trans + (1 - self.alpha - self.beta) * A_dyn


class LineAGC(nn.Module):
    """
    Parameter-Adaptive Graph Convolution (Line-AGC)
    Each route uses an independent convolution kernel.
    Formula:
      H_out(m) = ReLU( A_final * H_in(m) * W_m + b_m )
    """

    def __init__(self, in_dim, out_dim, num_nodes):
        super().__init__()
        self.route_weights = nn.Parameter(torch.randn(num_nodes, in_dim, out_dim))
        self.route_bias = nn.Parameter(torch.randn(1, num_nodes, out_dim))

    def forward(self, x, adj):
        """
        x: [B, T, M, in_dim]
        adj: [B, M, M]
        Output: [B, T, M, out_dim]
        """
        B, T, M, _ = x.shape
        x_reshaped = x.view(B * T, M, -1)
        adj_expanded = adj.repeat_interleave(T, dim=0)
        h = torch.bmm(adj_expanded, x_reshaped)
        h_trans = torch.einsum('bmd,mdl->bml', h, self.route_weights)
        h_out = h_trans + self.route_bias
        h_out = F.relu(h_out)
        return h_out.view(B, T, M, -1)


class STSA(nn.Module):
    """
    Spatio-Temporal Self-Attention Module (ST-SA)
    Includes:
      - Temporal self-attention: uses nn.MultiheadAttention to model each route's time series.
      - Cross-route spatial attention: exchanges information between each route and its neighbors
        based on predefined neighbor indices.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads)

    def forward(self, x, route_neighbors):
        """
        x: [B, T, M, d_model]
        route_neighbors: a list of length M, where each element is a list of neighbor indices for that route.
        Output: Fused features with temporal and spatial information, [B, T, M, d_model]
        """
        B, T, M, D = x.shape
        x_time = x.permute(1, 0, 2, 3).reshape(T, B * M, D)
        t_out, _ = self.temporal_attn(x_time, x_time, x_time)
        t_out = t_out.reshape(T, B, M, D).permute(1, 0, 2, 3)
        s_out_list = []
        for m in range(M):
            neighbors = route_neighbors[m]
            if len(neighbors) == 0:
                s_out_list.append(x[:, :, m, :])
            else:
                q = x[:, :, m, :].permute(1, 0, 2)
                k = x[:, :, neighbors, :].permute(1, 0, 2, 3).reshape(T, B * len(neighbors), D)
                v = k
                s_out, _ = self.spatial_attn(q, k, v)
                s_out_list.append(s_out.transpose(0, 1))
        s_out = torch.stack(s_out_list, dim=2)
        return t_out + s_out


class PriorFusion(nn.Module):
    """
    将全局特征与经过深层 MLP 映射后的站点先验特征进行动态融合，
    融合公式为：
       h_fused = g ⊙ h + (1 - g) ⊙ T(F_route_prior)
    其中 g = sigmoid(W_g([h; T(F_route_prior)]) + b_g)
    """
    def __init__(self, station_dim, global_dim, hidden_dim=128):
        """
        station_dim: STSANet 输出的维度
        global_dim: 全局特征维度（例如 64）
        hidden_dim: 融合模块内部隐藏层维度
        """
        super().__init__()
        # 深层映射 station 特征至 global 维度
        self.station_mlp = nn.Sequential(
            nn.Linear(station_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, global_dim)
        )
        # 门控机制，输入为全局特征和映射后的 station 特征拼接
        self.gate = nn.Sequential(
            nn.Linear(global_dim * 2, global_dim),
            nn.Sigmoid()
        )

    def forward(self, h, route_prior):
        """
        h: [B, T, M, global_dim]
        route_prior: [B, T, M, station_dim]
        """
        # 将站点先验映射到全局特征维度
        route_mapped = self.station_mlp(route_prior)
        fusion_input = torch.cat([h, route_mapped], dim=-1)
        gate_weight = self.gate(fusion_input)
        h_fused = gate_weight * h + (1 - gate_weight) * route_mapped
        return h_fused


class LinkGCRN(AbstractTrafficStateModel):
    """
    LinkGCRN Model with optional STSANet prior injection.
    当 use_stsanet_prior=True 时，将加载预训练的 STSANet 模型，
    利用其站点级预测作为先验，通过 route_station_mapping 聚合为线路级先验，
    并与全局特征采用门控融合进行动态融合。
    """
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')
        self.device = config.get('device', torch.device('cpu'))
        self.num_nodes = data_feature['num_nodes']      # 线路数：308
        self.feature_dim = data_feature['feature_dim']
        self.output_dim = data_feature['output_dim']

        # Static graph data
        self.register_buffer('A_dist', self._init_adj(data_feature.get('A_distance')))
        self.register_buffer('A_trans', self._init_adj(data_feature.get('A_transfer')))
        self.register_buffer('A_dyn', self._init_adj(data_feature.get('A_dynamic')))
        self.route_neighbors = self._init_neighbors(data_feature)

        # Module definitions
        self.graph_fusion = MultiGraphFusion(init_alpha=0.33, init_beta=0.33)
        self.route_conv = LineAGC(self.feature_dim, 64, self.num_nodes)
        self.sts_attn = STSA(64, n_heads=4)
        self.fc = nn.Linear(64, self.output_dim)

        # 联合训练开关：是否启用 STSANet 先验注入
        self.use_stsanet_prior = config.get('use_stsanet_prior', False)
        if self.use_stsanet_prior:
            stsanet_path = config.get('stsanet_path', None)
            if stsanet_path is None:
                raise ValueError("联合训练模式下必须提供 'stsanet_path'")
            if 'num_stations' not in data_feature:
                raise ValueError("联合训练模式下需要在 data_feature 中提供 'num_stations'（站点数量）")
            # 构造一个新的 data_feature，用于 STSANet，num_nodes 设为站点数量（例如4401）
            station_data_feature = data_feature.copy()
            station_data_feature['num_nodes'] = data_feature['num_stations']
            from libcity.model.traffic_flow_prediction.STSANet import STSANet
            self.stsanet = STSANet(config, station_data_feature).to(self.device)
            stsanet_state = torch.load(stsanet_path, map_location=self.device)
            if isinstance(stsanet_state, tuple):
                stsanet_state = stsanet_state[0]
            self.stsanet.load_state_dict(stsanet_state)
            self.stsanet.eval()
            for param in self.stsanet.parameters():
                param.requires_grad = False
            # 获取 STSANet 输出的维度
            stsanet_out_dim = data_feature.get('stsanet_output_dim', self.output_dim)
            # 使用新的 PriorFusion 模块代替单层线性映射
            self.prior_fusion = PriorFusion(station_dim=stsanet_out_dim, global_dim=64, hidden_dim=128)
            if 'route_station_mapping' not in data_feature:
                raise ValueError("联合训练模式下需要在 data_feature 中提供 'route_station_mapping'")

    def _init_adj(self, adj):
        if adj is None:
            return torch.eye(self.num_nodes, device=self.device)
        if isinstance(adj, np.ndarray):
            return torch.FloatTensor(adj).to(self.device)
        return adj.to(self.device)

    def _init_neighbors(self, data_feature):
        if 'route_neighbors' in data_feature:
            return data_feature['route_neighbors']
        A_dist_np = self.A_dist.cpu().numpy()
        neighbors = []
        for row in A_dist_np:
            nb = np.where(row > row.mean())[0].tolist()
            neighbors.append(nb)
        return neighbors

    def forward(self, batch):
        """
        输入 batch 中 'X' 为 (B, T, M, D)，标准线路数据。
        利用预训练 STSANet 模型和 data_feature 中传入的 route_station_mapping，
        生成站点级先验并经过 PriorFusion 模块与全局特征融合。
        """
        x = batch['X']  # (B, T, M, D)
        B, T, M, _ = x.shape
        A_dist_batch = self.A_dist.unsqueeze(0).repeat(B, 1, 1)
        A_trans_batch = self.A_trans.unsqueeze(0).repeat(B, 1, 1)
        A_dyn_batch = self.A_dyn.unsqueeze(0).repeat(B, 1, 1)
        A_final = self.graph_fusion(A_dist_batch, A_trans_batch, A_dyn_batch)
        h = F.relu(self.route_conv(x, A_final))  # [B, T, M, 64]

        if self.use_stsanet_prior:
            # 构造一个全零的 dummy 输入，形状为 (B, T, num_stations, D_station)
            D_station = getattr(self.stsanet, 'input_dim', self.output_dim)
            dummy = torch.zeros(B, T, self.data_feature['num_stations'], D_station, device=self.device)
            with torch.no_grad():
                stsanet_pred = self.stsanet({'X': dummy})
            # 根据 route_station_mapping 聚合站点预测为线路先验
            mapping = self.data_feature.get('route_station_mapping')
            route_prior_list = []
            for route in mapping:
                # 对每条线路，将对应站点特征取均值，得到 (B, T, stsanet_out_dim)
                route_feat = stsanet_pred[:, :, route, :].mean(dim=2)
                route_prior_list.append(route_feat)
            # route_prior: [B, T, M, stsanet_out_dim]
            route_prior = torch.stack(route_prior_list, dim=2)
            # 使用 PriorFusion 模块动态融合全局特征 h 与 station 先验 route_prior
            h = self.prior_fusion(h, route_prior)

        h_attn = self.sts_attn(h, self.route_neighbors)
        out = self.fc(h_attn)
        return out

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
