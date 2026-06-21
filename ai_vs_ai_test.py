#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn.modules.linear as my_linear
setattr(my_linear, '_LinearWithBias', my_linear.Linear)

import os
import torch
import koikoilearn
from koikoinet2L import DiscardModel, PickModel, KoiKoiModel

RL_DISCARD_PATH = 'model_rl/discard_state.pt'
RL_PICK_PATH    = 'model_rl/pick_state.pt'
RL_KOIKOI_PATH  = 'model_rl/koikoi_state.pt'
SL_DISCARD_PATH = 'model_agent/discard_sl.pt'
SL_PICK_PATH    = 'model_agent/pick_sl.pt'
SL_KOIKOI_PATH  = 'model_agent/koikoi_sl.pt'
RECORD_PATH = 'gamerecords_agents/'

os.makedirs(RECORD_PATH, exist_ok=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Transformerの互換性維持とBfloat16化を行うための共通関数
def load_model(path, model_class):
    model = model_class().to(device)
    
    loaded_data = torch.load(path, map_location=device, weights_only=False)
    if isinstance(loaded_data, torch.nn.Module):
        model.load_state_dict(loaded_data.state_dict())
    else:
        model.load_state_dict(loaded_data)
        
    for module in model.modules():
        if type(module).__name__ in ['MultiheadAttention', 'TransformerEncoderLayer']:
            if not hasattr(module, 'batch_first'): module.batch_first = False
            if not hasattr(module, 'norm_first'): module.norm_first = False
    return model.bfloat16().eval()

agent_rl = koikoilearn.Agent(
    load_model(RL_DISCARD_PATH, DiscardModel),
    load_model(RL_PICK_PATH, PickModel),
    load_model(RL_KOIKOI_PATH, KoiKoiModel)
)

agent_sl = koikoilearn.Agent(
    load_model(SL_DISCARD_PATH, DiscardModel),
    load_model(SL_PICK_PATH, PickModel),
    load_model(SL_KOIKOI_PATH, KoiKoiModel)
)

arena = koikoilearn.Arena(
    agent_rl, 
    agent_sl, 
    game_state_kwargs={'player_name': ['RL', 'SL'], 'record_path': RECORD_PATH, 'save_record': True}
)

print("[User vs Guan] AI同士の対戦を開始します...")
arena.multi_game_test(100)

print(f"\n対戦結果 -> User の勝利数: {arena.test_win_num[1]} / Guan の勝利数: {arena.test_win_num[2]} / 引き分け: {arena.test_win_num[0]}")