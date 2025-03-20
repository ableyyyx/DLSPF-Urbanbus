import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

class LightDGC(nn.Module):
    """
    Dynamic Graph Construction Module (Light-DGC)
    Computes cosine similarity → Top-K sparsification → Adaptive dependency injection.
    Formula:
      A_base(i,j) = sim(X_i, X_j)  (keep only Top-K neighbors)
      M1 = tanh(α * (E1 @ Θ1))
      M2 = tanh(α * (E2 @ Θ2))
      A_dynamic = ReLU(tanh(a*(M1@M2^T - M2@M1^T))) ⊙ A_base
    """

    def __init__(self, num_nodes, feature_dim, top_k, alpha=0.2, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.top_k = top_k
        self.alpha = alpha  # Controls the activation scale of the node embeddings
        self.E1 = nn.Parameter(torch.randn(num_nodes, feature_dim, device=self.device))
        self.E2 = nn.Parameter(torch.randn(num_nodes, feature_dim, device=self.device))
        self.theta1 = nn.Parameter(torch.randn(feature_dim, feature_dim, device=self.device))
        self.theta2 = nn.Parameter(torch.randn(feature_dim, feature_dim, device=self.device))
        # Scale factor "a" for the adaptive part
        self.a = nn.Parameter(torch.tensor(1.0, device=self.device))

    def forward(self, X):
        """
        X: [B, T, N, D], input station features over a period of time.
        Constructs the graph using the mean over the time dimension.
        Output: A_dynamic, [N, N]
        """
        batch_size, T, N, D = X.shape
        # Aggregate over the time dimension to obtain [B, N, D] (using mean)
        X_mean = X.mean(dim=1)
        # For simplicity, use the first sample to construct the graph
        X0 = X_mean[0]  # [N, D]
        # Compute cosine similarity after normalization
        X_norm = F.normalize(X0, p=2, dim=1)
        sim = torch.matmul(X_norm, X_norm.t())  # [N, N]
        # Exclude self-connections
        sim.fill_diagonal_(0)
        # Top-K sparsification: keep only the top_k largest similarities in each row
        topk_sim, indices = torch.topk(sim, self.top_k, dim=-1)
        A_base = torch.zeros_like(sim, device=self.device).scatter(-1, indices, topk_sim)
        # Generate adaptive graph
        M1 = torch.tanh(self.alpha * (self.E1 @ self.theta1))  # [N, D]
        M2 = torch.tanh(self.alpha * (self.E2 @ self.theta2))    # [N, D]
        diff = M1 @ M2.t() - M2 @ M1.t()
        adaptive = F.relu(torch.tanh(self.a * diff))
        # Element-wise multiplication to preserve the sparsity structure
        A_dynamic = adaptive * A_base
        return A_dynamic


class SlidingWindowSelfAttn(nn.Module):
    """
    Sliding Window Self-Attention Module
    Performs self-attention within a local time window.
    """

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, n_heads)

    def forward(self, x):
        # x: [B*N, T, embed_dim]
        # First, transpose to [T, B*N, embed_dim] to meet nn.MultiheadAttention input requirements
        x = x.transpose(0, 1)
        attn_output, _ = self.multihead_attn(x, x, x)
        # Transpose back to [B*N, T, embed_dim]
        return attn_output.transpose(0, 1)


class MultiScaleTC(nn.Module):
    """
    Multi-Scale Temporal Convolution Module (Multi-Scale TC)
    Consists of two parts:
      1. Multi-scale 1D convolution + gating mechanism to extract local features.
      2. Sliding window self-attention to capture global temporal dependencies.
    Formula:
      H_out = Conv1D(H_in) ⊙ σ(Conv1D(H_in))
      Attention(Q, K, V) = softmax(QK^T/√d_k)V
    """

    def __init__(self, in_dim, out_dim, kernel_sizes=[3, 5, 7], n_heads=4):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.gates = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.attn = SlidingWindowSelfAttn(out_dim, n_heads)

    def forward(self, x):
        """
        x: [B, T, N, D]
        First, apply convolution and gating to each station's time series independently,
        then use self-attention for global temporal interaction.
        Output: [B, T, N, out_dim]
        """
        B, T, N, D = x.shape
        # Merge station and batch dimensions for 1D convolution processing: [B*N, D, T]
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * N, D, T)
        conv_outputs = []
        for conv, gate in zip(self.convs, self.gates):
            h = conv(x_reshaped)       # [B*N, out_dim, T]
            g = torch.sigmoid(gate(x_reshaped))  # [B*N, out_dim, T]
            conv_outputs.append(h * g)
        # Aggregate multi-scale results (element-wise addition)
        merged = sum(conv_outputs)  # [B*N, out_dim, T]
        # Transpose to [B*N, T, out_dim] for self-attention
        merged = merged.transpose(1, 2)
        attn_out = self.attn(merged)  # [B*N, T, out_dim]
        # Restore shape to [B, T, N, out_dim]
        out = attn_out.reshape(B, N, T, -1).permute(0, 2, 1, 3)
        return out


class MultiScaleGC(nn.Module):
    """
    Multi-Scale Graph Convolution Module (Multi-Scale GC)
    Aggregates multi-hop information using k-th order adjacency matrices:
      H^(l+1) = ReLU(∑_(k=0)^K A_dynamic^k H^(l) W_k + b)
    """

    def __init__(self, in_dim, out_dim, K=2):
        super().__init__()
        self.K = K
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, out_dim))
            for _ in range(K + 1)
        ])
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, A):
        """
        x: [B, T, N, in_dim]
        A: [N, N]
        Output: [B, T, N, out_dim]
        """
        B, T, N, _ = x.shape
        # First, add the 0-th order (identity matrix) of A to the list.
        A_powers = [torch.eye(N, device=A.device)]
        for k in range(1, self.K + 1):
            A_powers.append(A_powers[-1] @ A)
        outputs = []
        # Treat the time dimension as independent and perform graph convolution.
        for k in range(self.K + 1):
            # x: [B, T, N, in_dim] → [B*T, N, in_dim]
            x_reshaped = x.reshape(B * T, N, -1)
            # Linear transformation
            h = torch.matmul(x_reshaped, self.weights[k])  # [B*T, N, out_dim]
            # Neighborhood aggregation: A_powers[k] is [N, N]
            h = torch.matmul(A_powers[k], h)  # [B*T, N, out_dim]
            outputs.append(h)
        # Sum over all orders and add bias
        out = sum(outputs) + self.bias
        out = F.relu(out)
        return out.reshape(B, T, N, -1)


class STSANet(AbstractTrafficStateModel):
    """
    STSANet Model
    Combines the Light-DGC, Multi-Scale TC, and Multi-Scale GC modules for station-level traffic flow prediction.
    Input: [B, T, N, D]
    Output: [B, 1, N, output_dim]
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.device = config.get('device', torch.device('cpu'))
        self.num_nodes = data_feature['num_nodes']
        self.feature_dim = data_feature['feature_dim']
        self.output_dim = data_feature['output_dim']
        self.top_k = config.get('top_k', 20)

        # Verify that the input feature dimension matches the output dimension.
        assert self.feature_dim == self.output_dim, \
            f"Input feature_dim({self.feature_dim}) must match output_dim({self.output_dim})"

        # Module definitions; pass the device to submodules as needed.
        self.light_dgc = LightDGC(self.num_nodes, self.feature_dim, self.top_k, device=self.device)
        self.multi_scale_tc = MultiScaleTC(self.feature_dim, 64, kernel_sizes=[3, 5, 7], n_heads=4)
        self.multi_scale_gc = MultiScaleGC(64, self.output_dim, K=2)

        # Parameter initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        """
        Input: batch containing 'X': [B, T, N, D]
        Output: [B, 1, N, output_dim]
        """
        x = batch['X']
        # Construct the dynamic adjacency matrix (station-level), shape [N, N]
        A = self.light_dgc(x)
        # Extract temporal features via multi-scale temporal convolution, output: [B, T, N, 64]
        h_time = self.multi_scale_tc(x)
        # Fuse spatial information using multi-scale graph convolution, output: [B, T, N, output_dim]
        h_space = self.multi_scale_gc(h_time, A)
        return h_space

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
