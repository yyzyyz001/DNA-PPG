import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
from .pos_embed import get_1d_sincos_pos_embed_from_grid


# ----------------------------
# Patch Embedding for 1D signal (input: B, 1, T)
# ----------------------------
class PatchEmbed1D(nn.Module):
    """1D signal to patch embeddings using Conv1d.
       Input:  (B, 1, T)
       Output: (B, P, E)
    """
    def __init__(self, ts_len=1250, patch_size=10, embed_dim=768, stride=None):
        super().__init__()
        self.ts_len = int(ts_len)
        self.patch_size = int(patch_size)
        self.stride = int(stride) if stride is not None else int(patch_size)

        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride
        )

        with torch.no_grad():
            dummy = torch.randn(1, 1, ts_len)
            _, E, P = self.proj(dummy).shape  # (1, E, P)
        self.num_patches = P
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        # x: (B, 1, T)
        x = self.proj(x)                    # (B, E, P)
        x = x.permute(0, 2, 1).contiguous() # (B, P, E)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 norm_layer=nn.LayerNorm, drop=0.0):
        super().__init__()
        self.block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            proj_drop=drop,
            attn_drop=drop
        )

    def forward(self, x):
        # x: (B, P+1, E)
        return self.block(x)


class Vit1DEncoder(nn.Module):
    def __init__(self,
                 ts_len=1250,
                 patch_size=10,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 stride=None,
                 pool_type: str = "cls"   # "cls" or "mean"
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pool_type = pool_type

        # 1) patch embedding
        self.patch_embed = PatchEmbed1D(
            ts_len=ts_len, patch_size=patch_size, embed_dim=embed_dim, stride=stride
        )
        num_patches = self.patch_embed.num_patches

        # 2) position & cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)

        # 3) encoder blocks
        self.blocks = nn.ModuleList([
            EncoderLayer(embed_dim=embed_dim,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    # -------- init --------
    def initialize_weights(self):
        with torch.no_grad():
            grid = np.arange(self.pos_embed.shape[-2], dtype=np.float32)
            pe = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid)
            self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    # -------- core encode --------
    def _encode(self, x: torch.Tensor):
        """
        x: (B, 1, T) -> (B, P+1, E)
        """
        x = self.patch_embed(x)                 # (B, P, E)
        x = x + self.pos_embed[:, 1:, :]       # (B, P, E)

        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)   # (B, 1, E)
        x = torch.cat([cls, x], dim=1)         # (B, P+1, E)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, pool_type= None):
        tokens = self._encode(x)
        pool = pool_type or self.pool_type
        if pool == "cls":
            return tokens[:, 0, :]
        if pool == "mean":
            return tokens[:, 1:, :].mean(dim=1)
        raise ValueError(...)

    # -------- public APIs --------
    @torch.no_grad()
    def forward_tokens(self, x: torch.Tensor):
        return self._encode(x)

    @torch.no_grad()
    def forward_pooled(self, x: torch.Tensor, pool_type: str = "cls"):
        pool = pool_type or self.pool_type
        tokens = self._encode(x)               # (B, P+1, E)
        if pool == "cls":
            return tokens[:, 0, :]             # (B, E)
        elif pool == "mean":
            return tokens[:, 1:, :].mean(dim=1)


if __name__ == "__main__":
    B, T = 8, 1250
    x = torch.randn(B, 1, T)

    encoder = Vit1DEncoder(
        ts_len=T,
        patch_size=10,
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=3.0,
        pool_type="cls"  # or "mean"
    )

    tokens = encoder.forward_tokens(x)   # (B, P+1, 768)
    feats  = encoder.forward_pooled(x)   # (B, 768)
    print(tokens.shape, feats.shape)
