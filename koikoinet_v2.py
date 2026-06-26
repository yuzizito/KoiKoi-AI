import torch
import torch.nn as nn
import torch.nn.functional as F

NetParameterV2 = {
    'nInput': 24,
    'nEmb': 128,
    'nFw': 256,
    'nAttnHead': 4,
    'nLayer': 3
}

class KoiKoiEncoderBlockV2(nn.Module):
    def __init__(self, nInput, nEmb, nFw, nAttnHead, nLayer):
        super(KoiKoiEncoderBlockV2, self).__init__()
        
        # Design Intent: Extract spatial features per month (4 cards) and aggregate into nEmb dimensions
        self.month_conv = nn.Conv2d(nInput, nEmb, kernel_size=(1, 4))
        
        # Design Intent: Learn sequence correlations between the 12 months (e.g., Yaku compositions)
        attn_layer = nn.TransformerEncoderLayer(nEmb, nAttnHead, nFw, batch_first=True)
        self.attn_encoder = nn.TransformerEncoder(attn_layer, nLayer)
        
    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor [Batch, Channels, SeqLen], got {x.dim()}D tensor.")
            
        B = x.size(0)
        
        # Design Intent: Reshape to treat the 48 cards as 12 distinct groups of 4 (months)
        x = x.view(B, -1, 12, 4)
        
        # Feature extraction per month: [B, 24, 12, 4] -> [B, nEmb, 12, 1] -> [B, nEmb, 12]
        x = self.month_conv(x).squeeze(3)
        x = F.relu(x)
        
        # Sequence correlation: [B, nEmb, 12] -> [B, 12, nEmb] -> Transformer -> [B, 12, nEmb]
        x = x.permute(0, 2, 1)
        x = F.layer_norm(x, [x.size(-1)])
        x = self.attn_encoder(x)
        
        # Spatial expansion back to card level: [B, 12, nEmb] -> [B, nEmb, 12] -> [B, nEmb, 48]
        x = x.permute(0, 2, 1)
        x = x.repeat_interleave(4, dim=2)
        
        return x

class DiscardModel(nn.Module):
    def __init__(self):
        super(DiscardModel, self).__init__()
        self.encoder_block = KoiKoiEncoderBlockV2(**NetParameterV2)
        self.out = nn.Conv1d(NetParameterV2['nEmb'], 1, 1)
        
    def forward(self, x):
        x = self.encoder_block(x)
        x = self.out(x).squeeze(1)
        return x

class PickModel(nn.Module):
    def __init__(self):
        super(PickModel, self).__init__()
        self.encoder_block = KoiKoiEncoderBlockV2(**NetParameterV2)
        self.out = nn.Conv1d(NetParameterV2['nEmb'], 1, 1)
        
    def forward(self, x):
        x = self.encoder_block(x)
        x = self.out(x).squeeze(1)
        return x

class KoiKoiModel(nn.Module):
    def __init__(self):
        super(KoiKoiModel, self).__init__()
        self.encoder_block = KoiKoiEncoderBlockV2(**NetParameterV2)
        self.out = nn.Conv1d(NetParameterV2['nEmb'], 1, 1)
        
    def forward(self, x):
        encoded = self.encoder_block(x)
        
        x_out = self.out(encoded[:, :, [0, 1]]).squeeze(1)
        return x_out

class TargetQNet(nn.Module):
    def __init__(self):
        super(TargetQNet, self).__init__()
        self.encoder_block = KoiKoiEncoderBlockV2(**NetParameterV2)
        self.out = nn.Conv1d(NetParameterV2['nEmb'], 1, 1)
        
    def forward(self, x):
        encoded = self.encoder_block(x)
        
        x_out = self.out(encoded[:, :, 0].unsqueeze(2)).squeeze(1)
        return x_out

class NeuRDModel(nn.Module):
    def __init__(self, is_koikoi=False):
        super(NeuRDModel, self).__init__()
        self.is_koikoi = is_koikoi
        self.encoder_block = KoiKoiEncoderBlockV2(**NetParameterV2)
        
        # Actor (Policy Head)
        self.policy_out = nn.Conv1d(NetParameterV2['nEmb'], 1, 1)
        # Critic (Value Head)
        self.value_out = nn.Linear(NetParameterV2['nEmb'], 1)
        
    def forward(self, x):
        encoded = self.encoder_block(x)
        
        if self.is_koikoi:
            logits = self.policy_out(encoded[:, :, [0, 1]]).squeeze(1)
        else:
            logits = self.policy_out(encoded).squeeze(1)
            
        # Value head expects [B, nEmb]. Use the 0-th token for global state value.
        value = self.value_out(encoded[:, :, 0]).squeeze(-1)
        
        return logits, value