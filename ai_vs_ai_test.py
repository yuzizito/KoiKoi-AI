#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:42:30 2021

@author: guansanghai
"""

# ==========================================
# 【追加】旧SLモデル読み込み用のPyTorchバージョン互換性パッチ
# ==========================================
import torch.nn.modules.linear as my_linear
my_linear._LinearWithBias = my_linear.Linear

import os
import torch
import koikoilearn

# 最新の強化学習モデル(RL-PBRS) と 古いベースライン(SL) を対戦させる設定
ai_name_pair = ['RL-PBRS', 'SL'] 
record_path = 'gamerecords_agents/'
game_state_kwargs={'player_name':ai_name_pair,
                   'record_path':record_path,
                   'save_record':True}

if not os.path.isdir(record_path):
    os.mkdir(record_path)

# デバイスの自動判別（play_vs_ai.py と統一）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ai_agent = {}
for ii, ai_name in enumerate(ai_name_pair):
    assert ai_name in ['RL-Point', 'RL-WP', 'SL', 'RL-PBRS']
    
    if ai_name == 'SL':
        discard_model_path = 'model_agent/discard_sl.pt'
        pick_model_path = 'model_agent/pick_sl.pt'
        koikoi_model_path = 'model_agent/koikoi_sl.pt'
    elif ai_name == 'RL-Point':
        discard_model_path = 'model_agent/discard_rl_point.pt'
        pick_model_path = 'model_agent/pick_rl_point.pt'
        koikoi_model_path = 'model_agent/koikoi_rl_point.pt'
    elif ai_name == 'RL-WP':
        discard_model_path = 'model_agent/discard_rl_wp.pt'
        pick_model_path = 'model_agent/pick_rl_wp.pt'
        koikoi_model_path = 'model_agent/koikoi_rl_wp.pt'
    elif ai_name == 'RL-PBRS':
        # 最新の学習ループが出力しているJITモデルを直接指定
        discard_model_path = 'traced_discard.pt'
        pick_model_path = 'traced_pick.pt'
        koikoi_model_path = 'traced_koikoi.pt'
    
    # ---------------------------------------------------------
    # 【JIT/FP32対応】互換性のあるロード処理
    # ---------------------------------------------------------
    if ai_name == 'RL-PBRS':
        # JITモデルは torch.jit.load で読み込み、即座に .float() 化
        discard_model = torch.jit.load(discard_model_path, map_location=device).float()
        pick_model    = torch.jit.load(pick_model_path, map_location=device).float()
        koikoi_model  = torch.jit.load(koikoi_model_path, map_location=device).float()
    else:
        # 古いネイティブモデルは weights_only=False で読み込み
        discard_model = torch.load(discard_model_path, map_location=device, weights_only=False)
        pick_model    = torch.load(pick_model_path, map_location=device, weights_only=False)
        koikoi_model  = torch.load(koikoi_model_path, map_location=device, weights_only=False)
        
        # Transformerの互換性パッチを一括適用
        for model in [discard_model, pick_model, koikoi_model]:
            for module in model.modules():
                if type(module).__name__ in ['MultiheadAttention', 'TransformerEncoderLayer']:
                    if not hasattr(module, 'batch_first'): module.batch_first = False
                    if not hasattr(module, 'norm_first'): module.norm_first = False
        
        discard_model = discard_model.float()
        pick_model    = pick_model.float()
        koikoi_model  = koikoi_model.float()
    
    discard_model.eval()
    pick_model.eval()
    koikoi_model.eval()
    
    ai_agent[ii+1] = koikoilearn.Agent(discard_model, pick_model, koikoi_model)

arena = koikoilearn.Arena(ai_agent[1], ai_agent[2], game_state_kwargs=game_state_kwargs)

print(f"[{ai_name_pair[0]} vs {ai_name_pair[1]}] AI同士の対戦を開始します...")

# 1局だと結果のブレが大きいため、実力を確認したい場合はここの数値を 100 などに増やしてください
arena.multi_game_test(1)

# 対局結果の表示を追加
print(f"対戦結果 -> {ai_name_pair[0]} の勝利数: {arena.test_win_num[0]} / {ai_name_pair[1]} の勝利数: {arena.test_win_num[1]} / 引き分け: {arena.test_win_num[2]}")
print('Over')