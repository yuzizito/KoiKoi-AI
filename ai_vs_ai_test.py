#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn.modules.linear as my_linear
my_linear._LinearWithBias = my_linear.Linear

import os
import glob
import torch
import koikoilearn

ai_name_pair = ['RL-PBRS', 'SL']
record_path = 'gamerecords_agents/'
game_state_kwargs={'player_name':ai_name_pair, 'record_path':record_path, 'save_record':True}

if not os.path.isdir(record_path):
    os.mkdir(record_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# GUI連携エラー回避ラッパー（型変換をより安全に強化）
class NativeBF16Wrapper(torch.nn.Module):
    def __init__(self, native_model):
        super().__init__()
        self.model = native_model
    def forward(self, x):
        if x.dtype != torch.bfloat16:
            x = x.bfloat16()
        return self.model(x).float()

def get_best_model_path(key):
    # アリーナで高スコアを出した瞬間のスナップショットを検索
    files = glob.glob(f'model_rl_pbrs/{key}_*_*.pt')
    
    # final_stop.pt などの数値ではないファイルを除外する
    valid_files = []
    for f in files:
        if "final_stop" in f:
            continue
        try:
            # 末尾のスコア部分（例: 92）を抽出して数値化できるかチェック
            score_part = f.split('_')[-1].split('.')[0]
            int(score_part)
            valid_files.append(f)
        except ValueError:
            continue

    if not valid_files:
        raise FileNotFoundError(f"【エラー】model_rl_pbrs フォルダに {key} の高スコア保存データ（数値付き）が見つかりません。")
    
    # 末尾のスコア数値が最も高いものを選択
    best_file = max(valid_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return best_file

ai_agent = {}
for ii, ai_name in enumerate(ai_name_pair):
    if ai_name == 'SL':
        d_net = torch.load('model_agent/discard_sl.pt', map_location=device, weights_only=False)
        p_net = torch.load('model_agent/pick_sl.pt', map_location=device, weights_only=False)
        k_net = torch.load('model_agent/koikoi_sl.pt', map_location=device, weights_only=False)

        for model in [d_net, p_net, k_net]:
            for module in model.modules():
                if type(module).__name__ == 'MultiheadAttention':
                    if not hasattr(module, 'batch_first'): module.batch_first = False
                elif type(module).__name__ == 'TransformerEncoderLayer':
                    if not hasattr(module, 'norm_first'): module.norm_first = False

        discard_model = d_net.float().eval()
        pick_model    = p_net.float().eval()
        koikoi_model  = k_net.float().eval()

    elif ai_name == 'RL-PBRS':
        d_path = get_best_model_path('discard')
        p_path = get_best_model_path('pick')
        k_path = get_best_model_path('koikoi')
        
        print(f"【チャンピオンモデル読込】\n{d_path}\n{p_path}\n{k_path}")

        # 1. アリーナで92%を叩き出した瞬間のネイティブモデルを直接ロード (JITトレースは不使用)
        d_net = torch.load(d_path, map_location=device, weights_only=False)
        p_net = torch.load(p_path, map_location=device, weights_only=False)
        k_net = torch.load(k_path, map_location=device, weights_only=False)

        # 2. rl_train.py と全く同じTransformerパッチ
        for model in [d_net, p_net, k_net]:
            for module in model.modules():
                if type(module).__name__ == 'MultiheadAttention':
                    if not hasattr(module, 'batch_first'): module.batch_first = False
                elif type(module).__name__ == 'TransformerEncoderLayer':
                    if not hasattr(module, 'norm_first'): module.norm_first = False

        # 3. アリーナと完全に同じ BFloat16 にキャスト
        d_net = d_net.bfloat16().eval()
        p_net = p_net.bfloat16().eval()
        k_net = k_net.bfloat16().eval()

        # 4. GUI連携ラッパー
        discard_model = NativeBF16Wrapper(d_net)
        pick_model    = NativeBF16Wrapper(p_net)
        koikoi_model  = NativeBF16Wrapper(k_net)

    ai_agent[ii+1] = koikoilearn.Agent(discard_model, pick_model, koikoi_model)

arena = koikoilearn.Arena(ai_agent[1], ai_agent[2], game_state_kwargs=game_state_kwargs)

print(f"[{ai_name_pair[0]} vs {ai_name_pair[1]}] AI同士の対戦を開始します...")
arena.multi_game_test(100)

print(f"\n対戦結果 -> {ai_name_pair[0]} の勝利数: {arena.test_win_num[0]} / {ai_name_pair[1]} の勝利数: {arena.test_win_num[1]} / 引き分け: {arena.test_win_num[2]}")
print('Over')