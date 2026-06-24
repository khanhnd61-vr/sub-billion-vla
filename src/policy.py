"""
policy.py - the VLA-Adapter "Policy": Bridge-Attention + L1-regression action head + proprio projector.

This is the *novel* part of VLA-Adapter and the only large module trained from scratch
(~97M params). It is a faithful, trimmed copy of `prismatic/models/action_heads.py` +
`prismatic/models/projectors.py` for the **Original** variant only (the `use_pro_version=True`
path with RoPE/FiLM is dropped for clarity). Module/parameter names are kept identical to the
parent repo so checkpoints are interchangeable.

The Bridge idea: the VLM is run once with `output_hidden_states=True`, giving one hidden-state
tensor per layer. We keep, at every layer, the 512 vision "task" tokens (h_t) and the 64
ActionQuery tokens (h_a). The action head is an MLP-ResNet over a learnable action-chunk seed;
each of its 24 blocks cross-attends to the corresponding VLM layer's h_t (gated) and h_a (+ proprio),
fusing raw VLM features into action latents. Output: an (B, 8, 7) action chunk; trained with L1.
"""
import math

import torch
import torch.nn as nn

from constants import ACTION_DIM, LLM_DIM, NUM_ACTIONS_CHUNK, NUM_LLM_LAYERS, NUM_TASK_TOKENS


def learnable_random_perturbations(seq_len, dim, device, dtype):
    """Small Gaussian perturbation added to the (zero) action seed during training."""
    p = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(p, mean=0.0, std=0.02)
    return p


class MLPResNetBlock(nn.Module):
    """One Bridge block: self-attention over the action-chunk tokens (x) + cross-attention to
    [h_a, proprio] (ungated) and h_t (gated by tanh(g)), then a residual FFN."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

        # NOTE: q/k/v/o are SHARED across the self / task / adapter streams (exactly as the repo).
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))  # tanh -> starts at 0 (h_t contributes nothing at init)

    def forward(self, x, h_t=None, h_a=None, p=None):
        # x: (B, T=8, C). h_t: (B, K, C) vision/task tokens. h_a: (B, 64, C) action-query tokens. p: (B, 1, C) proprio.
        ratio_g = torch.tanh(self.gating_factor)

        conditions = [h_a]
        if p is not None:
            conditions.append(p)
        h = torch.cat(conditions, dim=1)  # (B, K_t, C)  -- "task_k/v" in the repo (action + proprio)

        B, T, C = x.shape
        K_t = h.size(1)
        K = h_t.size(1)

        q = self.q_proj(x)
        k_tokens, v_tokens = self.k_proj(x), self.v_proj(x)
        k_task, v_task = self.k_proj(h), self.v_proj(h)
        k_adapter, v_adapter = self.k_proj(h_t), self.v_proj(h_t)

        def heads(t, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = heads(q, T)
        k_tokens, v_tokens = heads(k_tokens, T), heads(v_tokens, T)
        k_task, v_task = heads(k_task, K_t), heads(v_task, K_t)
        k_adapter, v_adapter = heads(k_adapter, K), heads(v_adapter, K)

        s_tokens = torch.matmul(q, k_tokens.transpose(-2, -1))                # self-attn
        s_task = torch.matmul(q, k_task.transpose(-2, -1))                    # -> [h_a, proprio]
        s_adapter = torch.matmul(q, k_adapter.transpose(-2, -1)) * ratio_g    # -> vision tokens (gated)

        scores = torch.cat([s_tokens, s_task, s_adapter], dim=-1) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        v = torch.cat([v_tokens, v_task, v_adapter], dim=2)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        return self.ffn(out + x)


class MLPResNet(nn.Module):
    """MLP-ResNet with one cross-attention Bridge block per VLM layer."""

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_a=None, h_t=None, p=None):
        # x: (B, 8, input_dim=7*896). h_a/h_t: (B, num_layers+1, tokens, 896).
        x = self.relu(self.fc1(self.layer_norm1(x)))     # (B, 8, hidden)
        for i, block in enumerate(self.mlp_resnet_blocks):
            # block i reads VLM layer i+1 (index 0 is the embedding layer, skipped).
            x = block(x, h_t=h_t[:, i + 1, :], h_a=h_a[:, i + 1, :], p=p)
        return self.fc2(self.layer_norm2(x))             # (B, 8, action_dim)


class L1RegressionActionHead(nn.Module):
    """The VLA-Adapter Policy: predicts a continuous (B, NUM_ACTIONS_CHUNK, ACTION_DIM) action chunk."""

    def __init__(self, input_dim=LLM_DIM, hidden_dim=LLM_DIM, action_dim=ACTION_DIM,
                 num_task_tokens=NUM_TASK_TOKENS, num_blocks=NUM_LLM_LAYERS):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim * ACTION_DIM,   # action seed is flattened to (chunk, action_dim*hidden)
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states, proprio=None, proprio_projector=None, phase="Inference"):
        # actions_hidden_states: (B, num_layers+1, NUM_TASK_TOKENS + NUM_TOKENS, hidden)
        B = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        p = None
        if proprio is not None and proprio_projector is not None:
            proprio = proprio.reshape(B, -1).to(torch.bfloat16)
            p = proprio_projector(proprio).unsqueeze(1)   # (B, 1, hidden)

        h_t = actions_hidden_states[:, :, : self.num_task_tokens, :]   # vision/task tokens per layer
        h_a = actions_hidden_states[:, :, self.num_task_tokens :, :]   # action-query tokens per layer

        # Zero action seed (B, NUM_ACTIONS_CHUNK, action_dim*hidden); + learnable perturbation while training.
        seed = torch.zeros((B, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
                           device=device, dtype=h_a.dtype).detach()
        seed = seed.reshape(B, NUM_ACTIONS_CHUNK, -1)                  # (B, 8, action_dim*hidden)
        if phase == "Training":
            seed = seed + learnable_random_perturbations(seed.shape[1], seed.shape[2], device, seed.dtype)

        return self.model(seed, h_a=h_a, p=p, h_t=h_t)                 # (B, 8, action_dim)


class ProprioProjector(nn.Module):
    """Projects the proprio vector (PROPRIO_DIM) up to the LLM hidden dim (one Bridge conditioning token)."""

    def __init__(self, llm_dim, proprio_dim):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act_fn1(self.fc1(x)))
