from typing import Optional, Tuple
import torch
import torch.nn as nn

C_CH = 26
G_CH = 7


class Backbone(nn.Module):
    """48枚のカード特徴と1個のグローバル特徴を統合するEncoder"""

    def __init__(
        self,
        dim: int = 64,
        n_head: int = 4,
        d_ff: int = 128,
        n_lyr: int = 2,
        drop: float = 0.05,
    ):
        super().__init__()
        # Design Intent: 異質な2つの特徴量を共通のdim空間へ写像する
        self.c_fc = nn.Linear(C_CH, dim)
        self.g_fc = nn.Linear(G_CH, dim)

        # Design Intent: カード固有概念を全ヘッドで共有表現させるための単一Embedding
        self.c_emb = nn.Embedding(48, dim)
        # Design Intent: 学習初期のAttention発散を防止する微小分散初期化
        nn.init.normal_(self.c_emb.weight, mean=0.0, std=0.02)

        lyr = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(lyr, num_layers=n_lyr, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, c_x: torch.Tensor, g_x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():  # type: ignore
            if c_x.shape[1:] != (C_CH, 48):
                raise ValueError(f"Expected c_x shape [B, {C_CH}, {48}], got {list(c_x.shape)}")
            if g_x.shape[1] != G_CH:
                raise ValueError(f"Expected g_x shape [B, {G_CH}], got {list(g_x.shape)}")

        # [B, 26, 48] -> [B, 48, 26] -> [B, 48, dim]
        c_tok = self.c_fc(c_x.permute(0, 2, 1))
        ids = torch.arange(48, device=c_x.device)
        c_tok = c_tok + self.c_emb(ids)

        # [B, 7] -> [B, 1, dim]
        g_tok = self.g_fc(g_x).unsqueeze(1)

        # 先頭(index 0)をGlobalとして結合 [B, 49, dim]
        x = torch.cat([g_tok, c_tok], dim=1)
        return self.norm(self.tr(x))


class NeuRD(nn.Module):
    """NeuRDに基づくこいこい方策・価値モデル（4Head構成）"""

    def __init__(
        self,
        dim: int = 64,
        n_head: int = 4,
        d_ff: int = 128,
        n_lyr: int = 2,
        drop: float = 0.05,
    ):
        super().__init__()
        self.net = Backbone(dim, n_head, d_ff, n_lyr, drop)

        # Design Intent: 48枚のカードトークン専用ヘッド(Pick / Discard共有)
        self.card_pi = nn.Linear(dim, 1)
        self.card_q = nn.Linear(dim, 1)

        # Design Intent: グローバルトークン専用ヘッド(Continue / Stop)
        self.koi_pi = nn.Linear(dim, 2)
        self.koi_q = nn.Linear(dim, 2)

    def forward(
        self,
        c_x: torch.Tensor,
        g_x: torch.Tensor,
        card_mask: Optional[torch.Tensor] = None,
        koi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(c_x, g_x)

        g_h = h[:, 0]   # [B, dim]
        c_h = h[:, 1:]  # [B, 48, dim]

        c_pi = self.card_pi(c_h).squeeze(-1)  # [B, 48]
        c_q = self.card_q(c_h).squeeze(-1)    # [B, 48]

        k_pi = self.koi_pi(g_h)               # [B, 2]
        k_q = self.koi_q(g_h)                 # [B, 2]

        if not torch.jit.is_tracing():  # type: ignore
            if card_mask is not None and card_mask.shape != c_pi.shape:
                raise ValueError(f"Expected card_mask shape {list(c_pi.shape)}, got {list(card_mask.shape)}")
            if koi_mask is not None and koi_mask.shape != k_pi.shape:
                raise ValueError(f"Expected koi_mask shape {list(k_pi.shape)}, got {list(koi_mask.shape)}")

        # Design Intent: Policyのみマスクし、NeuRDのアドバンテージ計算を破綻させない
        if card_mask is not None:
            c_pi = c_pi.masked_fill(~card_mask, -1e9)

        if koi_mask is not None:
            k_pi = k_pi.masked_fill(~koi_mask, -1e9)

        return c_pi, c_q, k_pi, k_q