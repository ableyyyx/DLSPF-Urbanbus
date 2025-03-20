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
        # Define independent convolution parameters for each route.
        self.route_weights = nn.Parameter(torch.randn(num_nodes, in_dim, out_dim))
        self.route_bias = nn.Parameter(torch.randn(1, num_nodes, out_dim))

    def forward(self, x, adj):
        """
        x: [B, T, M, in_dim]
        adj: [B, M, M]
        Output: [B, T, M, out_dim]
        """
        B, T, M, _ = x.shape
        # Merge the time dimension for neighborhood aggregation.
        x_reshaped = x.view(B * T, M, -1)  # [B*T, M, in_dim]
        # Expand the adjacency matrix to each time step.
        adj_expanded = adj.repeat_interleave(T, dim=0)  # [B*T, M, M]
        # Neighborhood aggregation.
        h = torch.bmm(adj_expanded, x_reshaped)  # [B*T, M, in_dim]
        # Use independent weights for each route via einsum: [B*T, M, in_dim] x [M, in_dim, out_dim]
        h_trans = torch.einsum('bmd,mdl->bml', h, self.route_weights)
        h_out = h_trans + self.route_bias  # Add bias via broadcasting
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
        # Temporal self-attention: merge the route and batch dimensions.
        x_time = x.permute(1, 0, 2, 3).reshape(T, B * M, D)  # [T, B*M, D]
        t_out, _ = self.temporal_attn(x_time, x_time, x_time)
        t_out = t_out.reshape(T, B, M, D).permute(1, 0, 2, 3)  # [B, T, M, D]

        # Cross-route spatial attention: for each route, interact with its neighbors.
        s_out_list = []
        for m in range(M):
            neighbors = route_neighbors[m]
            if len(neighbors) == 0:
                # If no neighbors, use the route's own features.
                s_out_list.append(x[:, :, m, :])
            else:
                # Use the target route's features as query: [B, T, D]
                q = x[:, :, m, :].permute(1, 0, 2)  # [T, B, D]
                # Concatenate neighbor features as key and value: [T, B*len(neighbors), D]
                k = x[:, :, neighbors, :].permute(1, 0, 2, 3).reshape(T, B * len(neighbors), D)
                v = k
                s_out, _ = self.spatial_attn(q, k, v)
                s_out_list.append(s_out.transpose(0, 1))  # [B, T, D]
        # Stack the spatial attention results to get [B, T, M, D]
        s_out = torch.stack(s_out_list, dim=2)
        # Fuse temporal and spatial information (addition)
        return t_out + s_out


class LinkGCRN(AbstractTrafficStateModel):
    """
    LinkGCRN Model
    Designed for route-level traffic flow prediction, it includes:
      - Multi-Graph Fusion Module (MGF)
      - Parameter-Adaptive Graph Convolution (Line-AGC)
      - Spatio-Temporal Self-Attention Module (ST-SA)
    Input: [B, T, M, D]
    Output: [B, 1, M, output_dim]
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.device = config.get('device', torch.device('cpu'))
        self.num_nodes = data_feature['num_nodes']  # Number of routes
        self.feature_dim = data_feature['feature_dim']
        self.output_dim = data_feature['output_dim']


        # Static graph data (precomputed by the data processing module), registered as buffers for device migration.
        self.register_buffer('A_dist', self._init_adj(data_feature.get('A_distance')))
        self.register_buffer('A_trans', self._init_adj(data_feature.get('A_transfer')))
        self.register_buffer('A_dyn', self._init_adj(data_feature.get('A_dynamic')))

        # Neighbor relationships: if provided in data_feature, use them; otherwise, generate based on A_dist.
        self.route_neighbors = self._init_neighbors(data_feature)

        # Module definitions.
        self.graph_fusion = MultiGraphFusion(init_alpha=0.33, init_beta=0.33)
        self.route_conv = LineAGC(self.feature_dim, 64, self.num_nodes)
        self.sts_attn = STSA(64, n_heads=4)
        self.fc = nn.Linear(64, self.output_dim)

    def _init_adj(self, adj):
        """
        Initialize the adjacency matrix. Supports numpy arrays or directly passed tensors.
        """
        if adj is None:
            return torch.eye(self.num_nodes, device=self.device)
        if isinstance(adj, np.ndarray):
            return torch.FloatTensor(adj).to(self.device)
        return adj.to(self.device)

    def _init_neighbors(self, data_feature):
        """
        Initialize neighbor relationships for each route.
        If not provided, automatically generate based on A_dist.
        """
        if 'route_neighbors' in data_feature:
            return data_feature['route_neighbors']
        # Simple strategy: for each route, select indices where the value is greater than the row mean.
        A_dist_np = self.A_dist.cpu().numpy()
        neighbors = []
        for row in A_dist_np:
            nb = np.where(row > row.mean())[0].tolist()
            neighbors.append(nb)
        return neighbors

    def forward(self, batch):
        """
        batch['X']: [B, T, M, D]
        Output: [B, 1, M, output_dim]
        """
        # Move input to device.
        x = batch['X']
        B, T, M, _ = x.shape
        # Expand static adjacency matrices to batch dimension.
        A_dist_batch = self.A_dist.unsqueeze(0).repeat(B, 1, 1)
        A_trans_batch = self.A_trans.unsqueeze(0).repeat(B, 1, 1)
        A_dyn_batch = self.A_dyn.unsqueeze(0).repeat(B, 1, 1)
        # Fuse the graphs to get the final adjacency matrix.
        A_final = self.graph_fusion(A_dist_batch, A_trans_batch, A_dyn_batch)
        # Apply route-level convolution.
        h = F.relu(self.route_conv(x, A_final))
        # Fuse temporal and spatial features using the ST-SA module (using predefined neighbor information).
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
