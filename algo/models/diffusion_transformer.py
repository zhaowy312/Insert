import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionTransformer(nn.Module):
    """
    输入: 
        noisy_action: [B, action_dim] (加了噪的动作)
        timestep: [B] (当前扩散步数)
        cond: [B, cond_dim] (视觉+触觉+本体感知的特征向量)
    输出:
        pred_noise: [B, action_dim] (预测的噪声)
    """
    def __init__(self, action_dim, cond_dim, embed_dim=256, n_heads=4, depth=4):
        super().__init__()
        
        self.action_embed = nn.Linear(action_dim, embed_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cond_embed = nn.Linear(cond_dim, embed_dim)

        # Transformer Encoder Layer
        # 这里把 cond 和 action 作为序列输入给 Transformer
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, activation='gelu', batch_first=True)
            for _ in range(depth)
        ])

        self.final_head = nn.Linear(embed_dim, action_dim)

    def forward(self, noisy_action, timestep, cond):
        # 1. Embedding
        x = self.action_embed(noisy_action) # [B, embed_dim]
        t = self.time_embed(timestep)       # [B, embed_dim]
        c = self.cond_embed(cond)           # [B, embed_dim]

        # 2. Construct Sequence for Transformer
        # 简单的做法：直接相加 (ResNet style) 或者 concat (Sequence style)
        # ReDi-LPD 建议：将 Condition 作为 token，Action 作为 token
        # 这里为了简单高效，采用 Feature-wise Condition (相加)
        
        token = x + t + c # [B, embed_dim]
        
        # 为了使用 Transformer，我们需要增加一个 seq_len 维度
        token = token.unsqueeze(1) # [B, 1, embed_dim]

        # 3. Transformer Processing
        for block in self.blocks:
            token = block(token)

        # 4. Output
        token = token.squeeze(1)
        pred_noise = self.final_head(token)
        
        return pred_noise