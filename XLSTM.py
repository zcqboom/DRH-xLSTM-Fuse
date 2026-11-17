
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Tuple, List, Union
import warnings

from xlstm_test.vision_lstm import (
    ViLBlock, SequenceTraversal,
    CausalConv1d, LinearHeadwiseExpand, MatrixLSTMCell,
    LayerNorm, small_init_, wang_init_,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    warnings.warn("检测到CPU环境，xLSTM在CPU上运行速度较慢，建议使用CUDA GPU运行！", UserWarning)

class HGMambaConfig:
    def __init__(self,
                 d_model: int = 32,
                 top_k: int = 3,
                 n_sequence: int = 8,
                 arw_length_ratio: float = 0.7,
                 n_layers: int = 2,
                 tile_size: int = 16,
                 ssm_d_state: int = 16,
                 ssm_d_conv: int = 4,
                 ssm_expand: int = 2):
        self.d_model = d_model
        self.top_k = top_k
        self.n_sequence = n_sequence
        self.arw_length_ratio = arw_length_ratio
        self.n_layers = n_layers
        self.tile_size = tile_size
        self.ssm_d_state = ssm_d_state
        self.ssm_d_conv = ssm_d_conv
        self.ssm_expand = ssm_expand


def dynamic_pad_img(img: torch.Tensor, tile_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, C, H_origin, W_origin = img.shape
    pad_h = (tile_size - H_origin % tile_size) % tile_size
    pad_w = (tile_size - W_origin % tile_size) % tile_size
    img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img_pad, (H_origin, W_origin)


def img2tile(img: torch.Tensor, tile_size: int) -> Tuple[torch.Tensor, dict]:
    B, C, H, W = img.shape
    n_h = H // tile_size
    n_w = W // tile_size
    N = n_h * n_w
    tiles = img.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, N, C, tile_size, tile_size)
    meta_info = {"n_h": n_h, "n_w": n_w, "N": N, "pad_hw": (H, W), "tile_size": tile_size}
    return tiles, meta_info


def tile2feat_map(tile_feat: torch.Tensor, meta_info: dict, origin_hw: Tuple[int, int]) -> torch.Tensor:
    B, N, d = tile_feat.shape
    n_h, n_w = meta_info["n_h"], meta_info["n_w"]
    tile_size = meta_info["tile_size"]
    pad_h, pad_w = meta_info["pad_hw"]
    H_origin, W_origin = origin_hw

    feat_grid = tile_feat.view(B, n_h, n_w, d).permute(0, 3, 1, 2).contiguous()
    feat_map = F.interpolate(feat_grid, size=(pad_h, pad_w), mode='bilinear', align_corners=False)
    feat_map = feat_map[:, :, :H_origin, :W_origin]
    return feat_map

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.gelu = nn.GELU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.gelu(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


class TileEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 32, tile_size: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.tile_size = tile_size
        self.conv_in = nn.Sequential(
            nn.Conv2d(5 * in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(64, 64) for _ in range(2)])
        self.se_block = SEBlock(64)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def _shift_tile_batch(self, tiles: torch.Tensor, direction: str, pad_mode: str = 'reflect') -> torch.Tensor:
        if direction == 'up':
            padded = F.pad(tiles, (0, 0, 1, 0), mode=pad_mode)
            shifted = padded[:, :, :-1, :]
        elif direction == 'down':
            padded = F.pad(tiles, (0, 0, 0, 1), mode=pad_mode)
            shifted = padded[:, :, 1:, :]
        elif direction == 'left':
            padded = F.pad(tiles, (1, 0, 0, 0), mode=pad_mode)
            shifted = padded[:, :, :, :-1]
        elif direction == 'right':
            padded = F.pad(tiles, (0, 1, 0, 0), mode=pad_mode)
            shifted = padded[:, :, :, 1:]
        else:
            raise ValueError(f"Unknown direction: {direction}")
        return shifted

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = tiles.shape
        assert H == self.tile_size and W == self.tile_size
        flat_tiles = tiles.view(B * N, C, H, W)
        up = self._shift_tile_batch(flat_tiles, 'up')
        down = self._shift_tile_batch(flat_tiles, 'down')
        left = self._shift_tile_batch(flat_tiles, 'left')
        right = self._shift_tile_batch(flat_tiles, 'right')
        combined = torch.cat([flat_tiles, up, down, left, right], dim=1)
        x = self.conv_in(combined)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.se_block(x)
        x = self.conv_out(x)
        tile_feat = x.view(B, N, -1)
        return tile_feat

class HypergraphBuilder(nn.Module):
    def __init__(self, top_k: int = 3):
        super().__init__()
        self.top_k = top_k

    def build_rule_adjacency(self, n_h: int, n_w: int) -> torch.Tensor:
        N = n_h * n_w
        coords = torch.stack(
            torch.meshgrid(torch.arange(n_h, device=device), torch.arange(n_w, device=device), indexing='ij'), dim=-1)
        coords = coords.view(N, 2)
        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=device)
        edges = []
        for i in range(N):
            h_i, w_i = coords[i]
            for dir in directions:
                h_j = h_i + dir[0]
                w_j = w_i + dir[1]
                if 0 <= h_j < n_h and 0 <= w_j < n_w:
                    j = h_j * n_w + w_j
                    if j > i:
                        edges.append((i, j))
        E_rule = len(edges)
        H_rule = torch.zeros((N, E_rule), device=device, dtype=torch.float32)
        for e_idx, (i, j) in enumerate(edges):
            H_rule[i, e_idx] = 1.0
            H_rule[j, e_idx] = 1.0
        return H_rule

    def build_similarity_adjacency(self, tile_feat: torch.Tensor) -> torch.Tensor:
        B, N, d = tile_feat.shape
        feat_norm = F.normalize(tile_feat, dim=-1)
        sim_matrix = torch.bmm(feat_norm, feat_norm.permute(0, 2, 1))
        sim_matrix = sim_matrix - torch.eye(N, device=device).unsqueeze(0) * 1e6
        _, top_k_idx = torch.topk(sim_matrix, k=self.top_k, dim=-1)
        H_sim = torch.zeros((B, N, N), device=device, dtype=torch.float32)
        for b in range(B):
            for i in range(N):
                H_sim[b, i, i] = 1.0
                H_sim[b, top_k_idx[b, i], i] = 1.0
        return H_sim

    def forward(self, tile_feat: torch.Tensor, meta_info: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, d = tile_feat.shape
        n_h, n_w = meta_info["n_h"], meta_info["n_w"]
        H_rule = self.build_rule_adjacency(n_h, n_w).unsqueeze(0).repeat(B, 1, 1)
        H_sim = self.build_similarity_adjacency(tile_feat)
        return H_rule, H_sim


class HGCNMessagePassing(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_node = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.ReLU()

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        B, N, d = X.shape
        B, N, E = H.shape
        D_v = torch.sum(H, dim=-1)
        D_v_sqrt_inv = torch.diag_embed(1.0 / torch.sqrt(D_v + 1e-6))
        D_e = torch.sum(H, dim=1)
        D_e_inv = torch.diag_embed(1.0 / (D_e + 1e-6))
        X_transformed = self.W_node(X)
        term1 = torch.bmm(D_v_sqrt_inv, H)
        term2 = torch.bmm(term1, D_e_inv)
        term3 = torch.bmm(term2, H.permute(0, 2, 1))
        term4 = torch.bmm(term3, D_v_sqrt_inv)
        X_updated = torch.bmm(term4, X_transformed)
        return self.activation(X_updated)


class HypergraphScanning(nn.Module):
    def __init__(self, n_sequence: int = 8, arw_length_ratio: float = 0.7):
        super().__init__()
        self.n_sequence = n_sequence
        self.arw_length_ratio = arw_length_ratio
        self.n_dfs = n_sequence // 2
        self.n_arw = n_sequence - self.n_dfs

    def hypergraph_to_nx(self, H: torch.Tensor) -> nx.Graph:
        N, E = H.shape
        G = nx.Graph()
        G.add_nodes_from(range(N))
        H_np = H.cpu().numpy()
        for e in range(E):
            nodes_in_edge = np.where(H_np[:, e] == 1.0)[0]
            if len(nodes_in_edge) >= 2:
                for i in range(len(nodes_in_edge)):
                    for j in range(i + 1, len(nodes_in_edge)):
                        G.add_edge(nodes_in_edge[i], nodes_in_edge[j])
        return G

    def h_dfs_traversal(self, G: nx.Graph, N: int) -> torch.Tensor:
        root = np.random.randint(0, N)
        dfs_nodes = list(nx.dfs_preorder_nodes(G, source=root))
        all_nodes = set(range(N))
        visited_nodes = set(dfs_nodes)
        unvisited_nodes = list(all_nodes - visited_nodes)
        dfs_sequence = dfs_nodes + unvisited_nodes
        return torch.tensor(dfs_sequence, device=device, dtype=torch.long)

    def h_arw_traversal(self, G: nx.Graph, N: int, T: int) -> torch.Tensor:
        current_node = np.random.randint(0, N)
        visited = set([current_node])
        arw_sequence = [current_node]
        while len(arw_sequence) < T and len(visited) < N:
            neighbors = list(G.neighbors(current_node))
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            if len(unvisited_neighbors) == 0:
                break
            current_node = np.random.choice(unvisited_neighbors)
            visited.add(current_node)
            arw_sequence.append(current_node)
        if len(arw_sequence) < N:
            pad_nodes = list(set(range(N)) - set(arw_sequence))
            arw_sequence += pad_nodes[:N - len(arw_sequence)]
        return torch.tensor(arw_sequence, device=device, dtype=torch.long)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, N, d = X.shape
        T_arw = max(int(N * self.arw_length_ratio), 1)
        all_sequence_feat = []
        all_sequence_idx = []
        for b in range(B):
            H_b = H[b]
            X_b = X[b]
            G = self.hypergraph_to_nx(H_b)
            if not nx.is_connected(G):
                components = list(nx.connected_components(G))
                for i in range(len(components) - 1):
                    u = next(iter(components[i]))
                    v = next(iter(components[i + 1]))
                    G.add_edge(u, v)
            batch_sequences = []
            for _ in range(self.n_dfs):
                seq_idx = self.h_dfs_traversal(G, N)
                batch_sequences.append(seq_idx)
            for _ in range(self.n_arw):
                seq_idx = self.h_arw_traversal(G, N, T_arw)
                batch_sequences.append(seq_idx)
            batch_seq_idx = torch.stack(batch_sequences, dim=0)
            all_sequence_idx.append(batch_seq_idx)
            batch_seq_feat = X_b[batch_seq_idx, :]
            all_sequence_feat.append(batch_seq_feat)
        sequence_feat = torch.stack(all_sequence_feat, dim=0)
        return sequence_feat, all_sequence_idx

class CrossViLLayer(nn.Module):

    def __init__(self, dim, direction, expansion=2, qkv_block_size=4,
                 proj_bias=False, conv_bias=True, kernel_size=4):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.kernel_size = kernel_size

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size

        self.proj_up_rule = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)
        self.proj_up_sim = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)

        self.q_proj_rule = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)
        self.k_proj_rule = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)
        self.v_proj_rule = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)

        self.q_proj_sim = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)
        self.k_proj_sim = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)
        self.v_proj_sim = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)

        self.conv1d_rule = CausalConv1d(dim=inner_dim, kernel_size=kernel_size, bias=conv_bias)
        self.conv1d_sim = CausalConv1d(dim=inner_dim, kernel_size=kernel_size, bias=conv_bias)

        self.mlstm_cell_rule = MatrixLSTMCell(dim=inner_dim, num_heads=qkv_block_size)
        self.mlstm_cell_sim = MatrixLSTMCell(dim=inner_dim, num_heads=qkv_block_size)

        self.learnable_skip_rule = nn.Parameter(torch.ones(inner_dim))
        self.learnable_skip_sim = nn.Parameter(torch.ones(inner_dim))

        self.proj_down_rule = nn.Linear(inner_dim, dim, bias=proj_bias)
        self.proj_down_sim = nn.Linear(inner_dim, dim, bias=proj_bias)

        self.reset_parameters()

    def reset_parameters(self):
        small_init_(self.proj_up_rule.weight, dim=self.dim)
        if self.proj_up_rule.bias is not None:
            nn.init.zeros_(self.proj_up_rule.bias)
        small_init_(self.proj_up_sim.weight, dim=self.dim)
        if self.proj_up_sim.bias is not None:
            nn.init.zeros_(self.proj_up_sim.bias)

        wang_init_(self.proj_down_rule.weight, dim=self.dim, num_blocks=1)
        if self.proj_down_rule.bias is not None:
            nn.init.zeros_(self.proj_down_rule.bias)
        wang_init_(self.proj_down_sim.weight, dim=self.dim, num_blocks=1)
        if self.proj_down_sim.bias is not None:
            nn.init.zeros_(self.proj_down_sim.bias)

        nn.init.ones_(self.learnable_skip_rule)
        nn.init.ones_(self.learnable_skip_sim)

        for proj in [self.q_proj_rule, self.k_proj_rule, self.v_proj_rule,
                     self.q_proj_sim, self.k_proj_sim, self.v_proj_sim]:
            small_init_(proj.weight, dim=self.dim)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        self.mlstm_cell_rule.reset_parameters()
        self.mlstm_cell_sim.reset_parameters()

    def forward(self, x_rule: torch.Tensor, x_sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = x_rule.shape
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x_rule = x_rule.flip(dims=[1])
            x_sim = x_sim.flip(dims=[1])
        else:
            raise NotImplementedError

        x_inner_rule = self.proj_up_rule(x_rule)
        x_mlstm_rule, z_rule = torch.chunk(x_inner_rule, chunks=2, dim=-1)
        x_mlstm_conv_rule = self.conv1d_rule(x_mlstm_rule)
        x_mlstm_conv_act_rule = F.silu(x_mlstm_conv_rule)
        q_rule = self.q_proj_rule(x_mlstm_conv_act_rule)
        k_rule = self.k_proj_rule(x_mlstm_conv_act_rule)
        v_rule = self.v_proj_rule(x_mlstm_rule)

        x_inner_sim = self.proj_up_sim(x_sim)
        x_mlstm_sim, z_sim = torch.chunk(x_inner_sim, chunks=2, dim=-1)
        x_mlstm_conv_sim = self.conv1d_sim(x_mlstm_sim)
        x_mlstm_conv_act_sim = F.silu(x_mlstm_conv_sim)
        q_sim = self.q_proj_sim(x_mlstm_conv_act_sim)
        k_sim = self.k_proj_sim(x_mlstm_conv_act_sim)
        v_sim = self.v_proj_sim(x_mlstm_sim)

        h_tilde_state_rule = self.mlstm_cell_rule(q=q_rule, k=k_sim, v=v_sim)

        h_tilde_state_skip_rule = h_tilde_state_rule + (self.learnable_skip_rule * x_mlstm_conv_act_rule)
        h_state_rule = h_tilde_state_skip_rule * F.silu(z_rule)

        h_tilde_state_sim = self.mlstm_cell_sim(q=q_sim, k=k_rule, v=v_rule)
        h_tilde_state_skip_sim = h_tilde_state_sim + (self.learnable_skip_sim * x_mlstm_conv_act_sim)
        h_state_sim = h_tilde_state_skip_sim * F.silu(z_sim)

        out_rule = self.proj_down_rule(h_state_rule)
        out_sim = self.proj_down_sim(h_state_sim)

        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            out_rule = out_rule.flip(dims=[1])
            out_sim = out_sim.flip(dims=[1])
        else:
            raise NotImplementedError

        return out_rule, out_sim

class DualInteractiveBiXLSM(nn.Module):
    def __init__(self, d_model: int, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.xlstm_forward = CrossViLLayer(
            dim=d_model,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            kernel_size=d_conv,
            expansion=expand
        )
        self.xlstm_backward = CrossViLLayer(
            dim=d_model,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            kernel_size=d_conv,
            expansion=expand
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, seq_feat_rule: torch.Tensor, seq_feat_sim: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, N, d = seq_feat_rule.shape
        seq_flat_rule = seq_feat_rule.reshape(B * M, N, d)
        seq_flat_sim = seq_feat_sim.reshape(B * M, N, d)

        out_rule_fwd, out_sim_fwd = self.xlstm_forward(seq_flat_rule, seq_flat_sim)
        out_rule_bwd, out_sim_bwd = self.xlstm_backward(seq_flat_rule, seq_flat_sim)

        out_rule = out_rule_fwd + out_rule_bwd
        out_sim = out_sim_fwd + out_sim_bwd

        out_rule = self.out_norm(self.out_proj(out_rule))
        out_sim = self.out_norm(self.out_proj(out_sim))

        return out_rule.view(B, M, N, d), out_sim.view(B, M, N, d)


class SequenceAggregator(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, device=device)

    def forward(self, ssm_feat: torch.Tensor, sequence_idx: List[torch.Tensor], N: int) -> torch.Tensor:
        B, M, N_seq, d = ssm_feat.shape
        aggregated_feat = torch.zeros((B, N, d), device=device)
        count_matrix = torch.zeros((B, N), device=device)
        for b in range(B):
            seq_idx_b = sequence_idx[b]
            ssm_feat_b = ssm_feat[b]
            for m in range(M):
                idx_m = seq_idx_b[m]
                feat_m = ssm_feat_b[m]
                aggregated_feat[b, idx_m] += feat_m
                count_matrix[b, idx_m] += 1
        count_matrix = count_matrix.unsqueeze(-1).clamp(min=1e-6)
        aggregated_feat = aggregated_feat / count_matrix
        return self.norm(aggregated_feat)


class HGXLSMBlock(nn.Module):
    def __init__(self, d_model: int, config: HGMambaConfig):
        super().__init__()
        self.d_model = d_model
        self.message_passing_rule = HGCNMessagePassing(d_model, d_model).to(device)
        self.message_passing_sim = HGCNMessagePassing(d_model, d_model).to(device)
        self.hypergraph_scanning_rule = HypergraphScanning(config.n_sequence, config.arw_length_ratio).to(device)
        self.hypergraph_scanning_sim = HypergraphScanning(config.n_sequence, config.arw_length_ratio).to(device)
        self.dual_bi_xlstm = DualInteractiveBiXLSM(
            d_model,
            d_conv=config.ssm_d_conv,
            expand=config.ssm_expand
        ).to(device)
        self.aggregator_rule = SequenceAggregator(d_model).to(device)
        self.aggregator_sim = SequenceAggregator(d_model).to(device)
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model * 2),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model, device=device)
        self.norm_res = nn.LayerNorm(d_model, device=device)

    def forward(self, X: torch.Tensor, H_rule: torch.Tensor, H_sim: torch.Tensor, meta_info: dict,
                origin_hw: Tuple[int, int]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict, Tuple]:

        residual = X
        N = X.shape[1]
        X_mp_rule = self.message_passing_rule(X, H_rule)
        X_mp_sim = self.message_passing_sim(X, H_sim)


        seq_feat_rule, seq_idx_rule = self.hypergraph_scanning_rule(X_mp_rule, H_rule)
        seq_feat_sim, seq_idx_sim = self.hypergraph_scanning_sim(X_mp_sim, H_sim)

        ssm_feat_rule, ssm_feat_sim = self.dual_bi_xlstm(seq_feat_rule, seq_feat_sim)

        X_agg_rule = self.aggregator_rule(ssm_feat_rule, seq_idx_rule, N)
        X_agg_sim = self.aggregator_sim(ssm_feat_sim, seq_idx_sim, N)
        concat_fusion = torch.cat([X_agg_rule, X_agg_sim], dim=-1)
        gates_fusion = self.fusion_gate(concat_fusion)
        gate_rule, gate_sim = gates_fusion.chunk(2, dim=-1)
        X_fused = gate_rule * X_agg_rule + gate_sim * X_agg_sim

        output = self.norm_res(self.out_proj(X_fused) + residual)
        return output, X_mp_rule, X_mp_sim


class HGXLSMFeatureExtractor(nn.Module):
    def __init__(self, config: HGMambaConfig = None, in_channels: int = 3):
        super().__init__()
        self.config = config if config is not None else HGMambaConfig()
        self.tile_size = self.config.tile_size
        self.d_model = self.config.d_model
        self.tile_encoder = TileEncoder(in_channels=in_channels, out_dim=self.d_model, tile_size=self.tile_size).to(
            device)
        self.hypergraph_builder = HypergraphBuilder(top_k=self.config.top_k).to(device)
        self.hgxlstm_layers = nn.ModuleList([
            HGXLSMBlock(d_model=self.d_model, config=self.config).to(device)
            for _ in range(self.config.n_layers)
        ])

    def forward_single_img(self, img: torch.Tensor, return_intermediate: bool = True) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

        B, C, H_origin, W_origin = img.shape
        img_pad, origin_hw = dynamic_pad_img(img, self.tile_size)
        tiles, meta_info = img2tile(img_pad, self.tile_size)
        tile_feat = self.tile_encoder(tiles)
        H_rule, H_sim = self.hypergraph_builder(tile_feat, meta_info)
        X = tile_feat
        inter_mp_feats = []

        for layer in self.hgxlstm_layers:
            X, mp_feat_rule, mp_feat_sim = layer(X, H_rule, H_sim,meta_info, origin_hw)

            inter_mp_feats.append(mp_feat_rule + mp_feat_sim)

        final_feat_map = tile2feat_map(X, meta_info, origin_hw)
        if return_intermediate:
            inter_feat_maps = [tile2feat_map(mp_feat, meta_info, origin_hw) for mp_feat in inter_mp_feats]
            return final_feat_map, inter_feat_maps
        else:
            return final_feat_map

    def forward(self, img1: torch.Tensor, img2: torch.Tensor = None, return_intermediate: bool = True) -> Union[
        torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], Tuple[
            Tuple[torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]]]:
        if img2 is None:
            return self.forward_single_img(img1, return_intermediate=return_intermediate)
        else:
            out1 = self.forward_single_img(img1, return_intermediate=return_intermediate)
            out2 = self.forward_single_img(img2, return_intermediate=return_intermediate)
            return out1, out2
