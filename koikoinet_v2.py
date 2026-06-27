# author: shguan3
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_MONTHS = 12
CARDS_PER_MONTH = 4
TOTAL_CARDS = NUM_MONTHS * CARDS_PER_MONTH  # 48


class KoiKoiEncoderBlockV2(nn.Module):
    """花札48枚を「12ヶ月×4枚」の空間構造として捉え直す特徴量抽出器"""

    def __init__(
        self,
        n_input: int = 24,
        n_emb: int = 128,
        n_fw: int = 256,
        n_attn_head: int = 4,
        n_layer: int = 3,
    ):
        super().__init__()
        self.n_input = n_input

        # Design Intent: 月単位(4枚)の空間相関を圧縮しn_emb次元へ写像する
        self.month_conv = nn.Conv2d(n_input, n_emb, kernel_size=(1, CARDS_PER_MONTH))

        # Design Intent: 12ヶ月間の出来役シーケンス相関を学習する
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=n_attn_head, dim_feedforward=n_fw, batch_first=True
        )
        self.attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # JITトレースコンパイル中はこの動的チェックをバイパスさせる
        if not torch.jit.is_tracing(): # type: ignore
            if x.dim() != 3 or x.size(1) != self.n_input or x.size(2) != TOTAL_CARDS:
                raise ValueError(
                    f"Expected tensor shape [Batch, {self.n_input}, {TOTAL_CARDS}], got {list(x.shape)}"
                )

        b_size = x.size(0)

        # [B, 24, 48] -> [B, 24, 12, 4]
        x = x.view(b_size, self.n_input, NUM_MONTHS, CARDS_PER_MONTH)

        # [B, 24, 12, 4] -> [B, nEmb, 12, 1] -> [B, nEmb, 12]
        x = F.relu(self.month_conv(x).squeeze(-1))

        # [B, nEmb, 12] -> [B, 12, nEmb] -> Transformer -> [B, 12, nEmb]
        x = x.permute(0, 2, 1)
        x = F.layer_norm(x, [x.size(-1)])
        x = self.attn_encoder(x)

        # [B, 12, nEmb] -> [B, nEmb, 12] -> [B, nEmb, 48]
        x = x.permute(0, 2, 1)
        return x.repeat_interleave(CARDS_PER_MONTH, dim=2)


class NeuRDModel(nn.Module):
    def __init__(
        self, is_koikoi: bool = False, n_input: int = 24, n_emb: int = 128
    ):
        super().__init__()
        self.is_koikoi = is_koikoi
        self.encoder_block = KoiKoiEncoderBlockV2(n_input=n_input, n_emb=n_emb)

        self.policy_out = nn.Conv1d(n_emb, 1, kernel_size=1)
        self.value_out = nn.Linear(n_emb, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder_block(x)

        # こいこい判断時は最初の2トークン[こいこい, 勝負]のみを抽出
        if self.is_koikoi:
            logits = self.policy_out(encoded[:, :, [0, 1]]).squeeze(1)
        else:
            logits = self.policy_out(encoded).squeeze(1)

        # 状態価値は0番トークンをグローバル表現として利用
        value = self.value_out(encoded[:, :, 0]).squeeze(-1)
        return logits, value