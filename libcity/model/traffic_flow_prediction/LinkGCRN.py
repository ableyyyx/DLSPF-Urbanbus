import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

class MultiGraphFusion(nn.Module):
    """多图融合模块，参考CONVGCN和RGSL"""

    def __init__(self, num_routes):
        super().__init__()
        self.weight_gen = nn.Sequential(
            nn.Linear(num_routes * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 3))

    def forward(self, A_dist, A_trans, A_dyn):
        features = torch.cat([
            A_dist.mean(dim=(1, 2)),
            A_trans.mean(dim=(1, 2)),
            A_dyn.mean(dim=(1, 2))
        ], dim=1)
        weights = F.softmax(self.weight_gen(features), dim=1)
        return weights[:, 0] * A_dist + weights[:, 1] * A_trans + weights[:, 2] * A_dyn


class LineAGC(nn.Module):
    """参数自适应图卷积，参考AGCRN和PDFormer"""

    def __init__(self, in_dim, out_dim, num_routes):
        super().__init__()
        self.route_weights = nn.Parameter(torch.randn(num_routes, in_dim, out_dim))
        self.route_bias = nn.Parameter(torch.randn(num_routes, 1, out_dim))

    def forward(self, x, adj):
        # x: [B, T, M, D]
        B, T, M, _ = x.shape
        x = x.view(B * T, M, -1)
        h = torch.bmm(adj.expand(B * T, -1, -1), x)  # [B*T, M, D]
        h = torch.einsum('bmd,mdl->bml', h, self.route_weights) + self.route_bias
        return h.view(B, T, M, -1)


class STSA(nn.Module):
    """时空自注意力模块，参考CRANN和PDFormer"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads)

    def forward(self, x, route_neighbors):
        # 时间维度注意力
        t_out, _ = self.temporal_attn(x, x, x)  # [T, B*M, D]

        # 空间跨路由注意力
        s_out = []
        for i in range(x.size(2)):  # 每个路由
            neighbors = route_neighbors[i]
            k = x[:, :, neighbors]  # [B, T, K, D]
            attn_out, _ = self.spatial_attn(x[:, :, i], k, k)
            s_out.append(attn_out)
        return t_out + torch.stack(s_out, dim=2)


class LinkGCRN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_routes = data_feature['num_routes']
        self.feature_dim = data_feature['feature_dim']
        self.output_dim = data_feature['output_dim']

        # 模块定义
        self.graph_fusion = MultiGraphFusion(self.num_routes)
        self.route_conv = LineAGC(self.feature_dim, 64, self.num_routes)
        self.sts_attn = STSA(64, n_heads=4)
        self.fc = nn.Linear(64, self.output_dim)

    def forward(self, batch):
        x = batch['X_route']  # [B, T, M, D]
        A_dist = batch['A_distance']
        A_trans = batch['A_transfer']
        A_dyn = batch['A_dynamic']

        # 图融合
        A_final = self.graph_fusion(A_dist, A_trans, A_dyn)

        # 路由卷积
        h = self.route_conv(x, A_final)  # [B, T, M, 64]
        h = self.sts_attn(h, batch['route_neighbors'])  # 时空注意力
        return self.fc(h[:, -1:, :, :])  # [B, 1, M, D]

    def calculate_loss(self, batch):
        y_true = batch['y_route']
        y_pred = self.predict(batch)
        return loss.masked_mae_torch(y_pred, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)