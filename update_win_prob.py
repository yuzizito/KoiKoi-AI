#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Windows環境での競合回避

import pickle
import multiprocessing
import numpy as np
import torch

torch.set_num_threads(1)

from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import koikoilearn
from koikoinet_v2 import DiscardModel, PickModel, KoiKoiModel

# パラメータ設定
MODEL_DIR = 'model_v2'
V2_PATHS = {
    'discard': f'{MODEL_DIR}/discard.pt',
    'pick': f'{MODEL_DIR}/pick.pt',
    'koikoi': f'{MODEL_DIR}/koikoi.pt'
}
SAVE_PATH = 'win_prob_mat.pkl'
N_TEST = 100

# バッチサイズ1の大量推論はCPUの方が圧倒的に効率的
DEVICE = torch.device('cpu') 

# 各子プロセスごとにメモリに常駐するエージェント
global_agent = None

def load_and_trace_model(path, model_class, dummy_input):
    """モデルをロードし、TorchScript (JIT) にコンパイルして推論を高速化する"""
    model = model_class().to(DEVICE)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
        if isinstance(state_dict, torch.nn.Module):
            state_dict = state_dict.state_dict()
        model.load_state_dict(state_dict)
    
    model.eval()
    
    # JITコンパイルによりPythonオーバーヘッドを完全に除去
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    return traced_model

def init_worker():
    """マルチプロセスの各ワーカー初期化時に1度だけ呼ばれる（メモリ効率化・通信コスト削減）"""
    global global_agent
    
    # ワーカーごとに乱数シードを分散させて対戦の多様性を担保
    import random
    import time
    seed = (os.getpid() * int(time.time() * 1000)) % 123456789
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # V2仕様 [Batch, 24, 48] のダミーテンソル
    dummy_input = torch.zeros(1, 24, 48, dtype=torch.float32, device=DEVICE)
    
    discard_model = load_and_trace_model(V2_PATHS['discard'], DiscardModel, dummy_input)
    pick_model = load_and_trace_model(V2_PATHS['pick'], PickModel, dummy_input)
    koikoi_model = load_and_trace_model(V2_PATHS['koikoi'], KoiKoiModel, dummy_input)
    
    global_agent = koikoilearn.Agent(discard_model, pick_model, koikoi_model)

def make_win_prob_dict_task(args):
    """ワーカー内で実行される単一のシミュレーションタスク"""
    round_num, point, dealer, n_test = args
    game_state_kwargs = {
        'round_num': round_num, 
        'round_total': 8,
        'init_point': [point, 60 - point], 
        'init_dealer': dealer
    }
    
    # 事前ロードされたJIT化エージェントを使用
    arena = koikoilearn.Arena(global_agent, global_agent, game_state_kwargs)
    arena.multi_game_test(n_test)
    
    return (round_num, point, dealer, arena.test_win_num)

if __name__ == '__main__':
    # 安全なプロセス起動モデル
    multiprocessing.set_start_method('spawn', force=True)
    
    print("--- 報酬関数(勝率テーブル) 高速更新スクリプト (V2/JIT対応) ---")
    
    # 1. タスクリストの生成
    dealer = 1
    tasks = []
    for round_num in range(8, 0, -1):
        for point in range(1, 60, 2):
            tasks.append((round_num, point, dealer, N_TEST))
            
    total_tasks = len(tasks)
    results = []

    # 2. 並列シミュレーションの実行 (ワーカー初期化を利用)
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"\n[Step 1] {n_workers} ワーカーでアリーナ対戦シミュレーションを開始します...")
    
    with multiprocessing.Pool(processes=n_workers, initializer=init_worker) as pool:
        # ★ ここを修正：tqdmでラップするだけでプログレスバーになります
        for result in tqdm(pool.imap_unordered(make_win_prob_dict_task, tasks), total=total_tasks, desc="シミュレーション進捗"):
            results.append(result)

    results.sort()

    # 3. ロジスティック回帰による曲線の平滑化
    print("\n[Step 2] ロジスティック回帰による勝率曲線の平滑化を実行しています...")
    win_prob_mat = np.zeros([2, 9, 61])
    
    for round_num in range(1, 9):
        round_results = [item for item in results if item[0] == round_num]
        
        x_list, y_list = [], []
        for _, pt, dlr, win_num in round_results:
            # 引き分けは両者に分配
            draw_half = win_num[0] // 2
            n_win = win_num[1] + draw_half
            n_lose = win_num[2] + draw_half
            
            # Numpyを使った高速な配列展開
            if n_win + n_lose > 0:
                x_list.append(np.full(n_win + n_lose, pt))
                y_list.append(np.concatenate([np.ones(n_win), np.zeros(n_lose)]))
                
        if x_list and y_list:
            x = np.concatenate(x_list).reshape(-1, 1)
            y = np.concatenate(y_list)
            
            if len(np.unique(y)) > 1:
                win_prob_model = LogisticRegression(n_jobs=-1)
                win_prob_model.fit(x, y)
                p = np.arange(61)
                w = win_prob_model.predict_proba(p.reshape(-1, 1))[:, 1]
            else:
                w = np.full(61, float(y[0]))
        else:
            w = np.full(61, 0.5)
            
        win_prob_mat[1, round_num, :] = w
        win_prob_mat[0, round_num, :] = 1 - w[::-1]

    # 4. 保存
    print(f"\n[Step 3] 更新された勝率テーブルを '{SAVE_PATH}' に保存します...")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(win_prob_mat, f)
        
    print("\nすべての処理が完了しました！")