# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:34:35 2021

@author: shguan3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# length: 48(discard/pick), 50(koikoi)
# origin input size: (BATCH_SIZE, nFeature, length)
# conv1d input size: (BATCH_SIZE, nFeature, length)
# [UPDATED] multiheadattention input size: (BATCH_SIZE, length, nEmb) -> batch_first=True

NetParameter = {
    'nInput': 300,
    'nEmb': 256,
    'nFw': 512,
    'nAttnHead': 4,
    'nLayer': 2
}

class KoiKoiEncoderBlock(nn.Module):
    def __init__(self, nInput, nEmb, nFw, nAttnHead, nLayer):
        super(KoiKoiEncoderBlock, self).__init__()
        self.f1 = nn.Conv1d(nInput, nFw, 1)
        self.f2 = nn.Conv1d(nFw, nEmb, 1)
        
        # 【変更点】batch_first=True を指定
        # これにより、PyTorch 2.0以降で最速のエフェクティブなSDPAカーネル（FlashAttention等）が自動選択されます
        attn_layer = nn.TransformerEncoderLayer(nEmb, nAttnHead, nFw, batch_first=True)
        self.attn_encoder = nn.TransformerEncoder(attn_layer, nLayer)
        
    def forward(self, x): 
        x = self.f2(F.relu(self.f1(x)))
        x = F.layer_norm(x, [x.size(-1)])  # 元のLayerNormの挙動（L次元の正規化）を維持
        
        # 【変更点】(BATCH_SIZE, nEmb, length) -> (BATCH_SIZE, length, nEmb)
        # バッチ次元（0番目）を固定したまま転置するため、メモリ効率が大幅に向上します
        x = x.permute(0, 2, 1)
        
        x = self.attn_encoder(x)
        
        # 【変更点】Conv1dの入力形式 (BATCH_SIZE, nEmb, length) に戻す
        x = x.permute(0, 2, 1)  
        return x


class DiscardModel(nn.Module):
    def __init__(self):
        super(DiscardModel, self).__init__()       
        self.encoder_block = KoiKoiEncoderBlock(**NetParameter)
        self.out = nn.Conv1d(NetParameter['nEmb'], 1, 1)
        
    def forward(self, x):       
        x = self.encoder_block(x)
        x = self.out(x).squeeze(1)
        return x


class PickModel(nn.Module):
    def __init__(self):
        super(PickModel, self).__init__()
        self.encoder_block = KoiKoiEncoderBlock(**NetParameter)
        self.out = nn.Conv1d(NetParameter['nEmb'], 1, 1)
        
    def forward(self, x):
        x = self.encoder_block(x)
        x = self.out(x).squeeze(1)
        return x
    

class KoiKoiModel(nn.Module):
    def __init__(self):
        super(KoiKoiModel, self).__init__()
        self.encoder_block = KoiKoiEncoderBlock(**NetParameter)
        self.out = nn.Conv1d(NetParameter['nEmb'], 1, 1)
        
    def forward(self, x):
        x = self.encoder_block(x)
        x = self.out(x[:, :, [0, 1]]).squeeze(1)
        return x


class TargetQNet(nn.Module):
    def __init__(self):
        super(TargetQNet, self).__init__()       
        self.encoder_block = KoiKoiEncoderBlock(**NetParameter)
        self.out = nn.Conv1d(NetParameter['nEmb'], 1, 1)
        
    def forward(self, x):       
        x = self.encoder_block(x)
        x = self.out(x[:, :, 0].unsqueeze(2)).squeeze(1)
        return x