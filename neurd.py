from typing import Optional, Tuple
import torch
import torch.nn as nn

C_CH = 43
G_CH = 7


class Backbone(nn.Module):
    """48枚のカード特徴と1個のグローバル特徴を統合するEncoder"""

    # 型チェッカー（Pylance等）の推論エラーを回避するための型定義
    card_ids: torch.Tensor
    month_ids: torch.Tensor
    kind_ids: torch.Tensor
    month_kind_ids: torch.Tensor
    role_ids: torch.Tensor

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

        # Design Intent: カード番号だけでは「同じ月」「同じ札種」「同じ役候補」という構造を
        # Transformerがすべてデータから再発見する必要がある。
        # Embeddingを構造化することで、Attentionが意味的に近いカード同士へ早期から
        # 注意を向けられるようにし、学習効率と汎化性能の向上を狙う。
        self.card_emb = nn.Embedding(48, dim)
        self.month_emb = nn.Embedding(12, dim)
        self.kind_emb = nn.Embedding(4, dim)
        self.month_kind_emb = nn.Embedding(48, dim)
        num_roles = 10
        self.role_emb = nn.Embedding(num_roles, dim)

        # Design Intent: 学習初期のAttention発散を防止する微小分散初期化
        nn.init.normal_(self.card_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.month_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.kind_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.month_kind_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.role_emb.weight, mean=0.0, std=0.02)

        # --- 固定のカード属性テンソルをBufferとして登録 ---
        self.register_buffer("card_ids", torch.arange(48, dtype=torch.long))
        
        month_ids = torch.arange(48, dtype=torch.long) // 4
        self.register_buffer("month_ids", month_ids)

        # 札種 (0: Light, 1: Animal/Tane, 2: Ribbon, 3: Kasu)
        kind = torch.full((48,), 3, dtype=torch.long)
        kind[[0, 8, 28, 40, 44]] = 0
        kind[[4, 12, 16, 20, 24, 29, 32, 36, 41]] = 1
        kind[[1, 5, 9, 13, 17, 21, 25, 33, 37, 42]] = 2
        self.register_buffer("kind_ids", kind)

        # Design Intent: (month_id * 4 + kind_id) によって、「同じ月の同じ札種(例: 1月のカス)」を共通のカテゴリとして表現する
        self.register_buffer("month_kind_ids", month_ids * 4 + kind)

        # 役属性 (1カードが複数所属可能) はマルチホット表現 [48, 10] とし、行列積で加算合成できるようにする
        roles = torch.zeros(48, num_roles, dtype=torch.float32)
        roles[[0, 8, 28, 40, 44], 0] = 1.0 # Light
        roles[[1, 5, 9, 13, 17, 21, 25, 33, 37, 42], 1] = 1.0 # Ribbon
        roles[[4, 12, 16, 20, 24, 29, 32, 36, 41], 2] = 1.0 # Animal
        roles[[2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 32, 34, 35, 38, 39, 43, 45, 46, 47], 3] = 1.0 # Kasu
        roles[[40], 4] = 1.0 # Rain
        roles[[20, 24, 36], 5] = 1.0 # BoarDeerButterfly
        roles[[1, 5, 9], 6] = 1.0 # RedRibbon
        roles[[21, 33, 37], 7] = 1.0 # BlueRibbon
        roles[[8, 32], 8] = 1.0 # FlowerSake
        roles[[28, 32], 9] = 1.0 # MoonSake
        self.register_buffer("role_ids", roles)

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
        
        # Design Intent: 役属性はマルチホット表現との行列積により、該当する複数ロールのEmbeddingの和(sum)を一度に取得する
        role_embeddings = torch.matmul(self.role_ids, self.role_emb.weight)

        c_tok = (
            c_tok 
            + self.card_emb(self.card_ids)
            + self.month_emb(self.month_ids)
            + self.kind_emb(self.kind_ids)
            + self.month_kind_emb(self.month_kind_ids)
            + role_embeddings
        )

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