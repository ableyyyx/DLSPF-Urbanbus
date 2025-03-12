import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from logging import getLogger


class LightDGC(nn.Module):
    def __init__(self, num_nodes, feature_dim, top_k, alpha=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.top_k = top_k
        self.alpha = alpha
        self.E1 = nn.Parameter(torch.randn(num_nodes, feature_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, feature_dim))
        self.theta1 = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.theta2 = nn.Parameter(torch.randn(feature_dim, feature_dim))

    def forward(self, X):
        batch_size, T, N, D = X.shape
        X = X.mean(dim=1)
        sim = F.cosine_similarity(X.unsqueeze(2), X.unsqueeze(1), dim=-1)
        topk_sim, indices = torch.topk(sim, self.top_k, dim=-1)
        A_base = torch.zeros_like(sim).scatter(-1, indices, topk_sim)
        M1 = torch.tanh(self.alpha * (self.E1 @ self.theta1))
        M2 = torch.tanh(self.alpha * (self.E2 @ self.theta2))
        A_dynamic = F.relu(torch.tanh(self.alpha * (M1 @ M2.T - M2 @ M1.T))) * A_base
        return A_dynamic


class MultiScaleTC(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.gate = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        outputs = [conv(x) * self.gate(x) for conv in self.convs]
        return torch.stack(outputs).sum(0).permute(0, 2, 3, 1)


class MultiScaleGC(nn.Module):
    def __init__(self, in_dim, out_dim, K=3):
        super().__init__()
        self.K = K
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, out_dim))
            for _ in range(K + 1)
        ])

    def forward(self, x, A):
        A_power = torch.eye(A.size(0), device=A.device)
        outputs = []
        for k in range(self.K + 1):
            h = torch.einsum('ntij,jd->ntid', A_power, self.weights[k])
            outputs.append(h)
            A_power = A_power @ A
        return torch.stack(outputs).sum(0)


class STSANet(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = data_feature['num_nodes']
        self.feature_dim = data_feature['feature_dim']
        self.output_dim = data_feature['output_dim']
        self.top_k = config.get('top_k', 20)
        self._scaler = data_feature['scaler']
        self.device = config.get('device', torch.device('cpu'))

        # Modules
        self.light_dgc = LightDGC(self.num_nodes, self.feature_dim, self.top_k)
        self.multi_scale_tc = MultiScaleTC(self.feature_dim, 64)
        self.multi_scale_gc = MultiScaleGC(64, self.output_dim)

        # External data handling
        self.ext_dim = data_feature.get('ext_dim', 0)
        if self.ext_dim > 0:
            self.ext_fc = nn.Linear(self.ext_dim, self.output_dim * self.num_nodes)

        self._logger = getLogger()

    def forward(self, batch):
        x = batch['X']  # (B, T, N, D)
        B, T, N, D = x.shape

        # Spatiotemporal Features
        A = self.light_dgc(x)
        h_time = self.multi_scale_tc(x)  # (B, T, N, 64)
        h_space = self.multi_scale_gc(h_time, A)  # (B, T, N, output_dim)

        # External Features
        if self.ext_dim > 0:
            ext = batch['X_ext'][:, -1, :]  # Use last time step
            ext_out = self.ext_fc(ext).view(B, self.output_dim, N)
            h_space += ext_out.permute(0, 2, 1).unsqueeze(1)

        return h_space[:, -1:, :, :]  # (B, 1, N, output_dim)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_pred = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true)
        y_pred = self._scaler.inverse_transform(y_pred)
        return loss.masked_mae_torch(y_pred, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)