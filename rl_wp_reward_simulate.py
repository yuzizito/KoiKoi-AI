#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import multiprocessing
import torch
import koikoilearn
from koikoinet2L import DiscardModel, PickModel, KoiKoiModel

RL_FOLDER = 'model_rl_wp'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model_clean(path, model_class):
    model = model_class().to(DEVICE)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(state_dict, torch.nn.Module):
            state_dict = state_dict.state_dict()
        model.load_state_dict(state_dict)
        
    for module in model.modules():
        if type(module).__name__ == 'MultiheadAttention':
            if not hasattr(module, 'batch_first'): module.batch_first = False
        elif type(module).__name__ == 'TransformerEncoderLayer':
            if not hasattr(module, 'norm_first'): module.norm_first = False
    return model.eval()

def make_win_prob_dict(agent, round_num, point, dealer, n_test):
    """アリーナ対戦による特定条件下での勝率計測"""
    game_state_kwargs = {'round_num': round_num, 'init_point': [point, 60 - point], 'init_dealer': dealer}
    arena = koikoilearn.Arena(agent, agent, game_state_kwargs)
    arena.multi_game_test(n_test)
    result = (round_num, point, dealer, arena.test_win_num)
    return result

if __name__ == '__main__':
    discard_model = load_model_clean(f'{RL_FOLDER}/discard_0_0.pt', DiscardModel)
    pick_model = load_model_clean(f'{RL_FOLDER}/pick_0_0.pt', PickModel)
    koikoi_model = load_model_clean(f'{RL_FOLDER}/koikoi_0_0.pt', KoiKoiModel)
    ai_agent = koikoilearn.Agent(discard_model, pick_model, koikoi_model)

    dealer = 1
    n_test = 200
    results = []

    print("勝率テーブルシミュレーションを開始します...")
    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 2)) as pool:
        async_tasks = []
        for round_num in range(8, 0, -1):
            for point in range(1, 60):
                args = (ai_agent, round_num, point, dealer, n_test)
                async_tasks.append(pool.apply_async(make_win_prob_dict, args=args))
        
        for task in async_tasks:
            results.append(task.get())

    results.sort()
    
    with open('result_wp_sim.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("シミュレーション完了: result_wp_sim.pkl を保存しました。")