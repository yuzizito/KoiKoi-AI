#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import pickle
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

import core
from neurd import NeuRD, C_CH, G_CH

# ==========================================
# 設定
# ==========================================
MODEL_PATH = 'model/model.pt'
SAVE_PATH = 'win_prob_mat.pkl'
N_TEST = 8000 
MAX_RND = 6  # ラウンド数の上限を6に変更
DEVICE_STR = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


def load_and_cache_model():
    model = NeuRD().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        elif isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        elif isinstance(state_dict, torch.nn.Module):
            model.load_state_dict(state_dict.state_dict())
    else:
        print("Warning: Model file not found. Using randomly initialized weights.")
    
    model.eval()
    
    dummy_c = torch.zeros(1, C_CH, 48, dtype=torch.float32, device=DEVICE)
    dummy_g = torch.zeros(1, G_CH, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (dummy_c, dummy_g), check_trace=False)
    
    buf = io.BytesIO()
    torch.jit.save(traced_model, buf)
    core.reload_model(buf.getvalue())


def analyze_win_prob(mat):
    """生成された勝率テーブルの品質をテストする"""
    print(f"\n[Step 3] 生成されたテーブルの品質テストを開始します...")
    print(f"データ形状: {mat.shape} (期待値: [2, 9, 61])")
    
    print("\n【1. 境界値のチェック】")
    # MAX_RND=6 に合わせてテスト対象のラウンドを変更
    for r in [1, 3, 6]:
        p_minus30 = mat[1, r, 0]   # インデックス 0  (-30点差)
        p_0       = mat[1, r, 30]  # インデックス 30 (同点)
        p_plus30  = mat[1, r, 60]  # インデックス 60 (+30点差)
        print(f"ラウンド {r} (親) -> -30点差: {p_minus30:.4f} | 0点差: {p_0:.4f} | +30点差: {p_plus30:.4f}")
    
    print("\n【2. 単調増加性 (右肩上がりか) のチェック】")
    monotonic_errors = 0
    for d in [0, 1]:
        for r in range(1, MAX_RND + 1):
            diff = np.diff(mat[d, r, :])
            if np.any(diff < -1e-5):
                monotonic_errors += 1
    if monotonic_errors == 0:
        print("✅ パス: すべての条件で勝率が単調増加しています。")
    else:
        print(f"❌ 警告: 勝率が途中で減少している箇所が {monotonic_errors} 箇所あります。")


def main():
    print("--- 報酬関数(勝率テーブル) 高速更新・テストスクリプト (NeuRD/C++統合版) ---")
    
    load_and_cache_model()
    
    print(f"\n[Step 1] C++エンジンでのアリーナ対戦シミュレーションを開始します... (N_TEST={N_TEST})")
    results = []
    
    # MAX_RND=6 でシミュレーションを回す
    for round_num in tqdm(range(MAX_RND, 0, -1), desc="シミュレーション進捗"):
        for pt1 in range(15, 46):
            pt2 = 60 - pt1
            diff = pt1 - pt2  # -30, -28, ..., 0, ..., 28, 30 (必ず偶数、2点刻み)
            
            # 親 (P1=Dealer)
            res_d1 = core.run_arena(N_TEST, b"", b"", DEVICE_STR, MAX_RND, round_num, pt1, pt2, 1)
            results.append((round_num, diff, 1, res_d1))
            
            # 子 (P1=Non-Dealer)
            res_d2 = core.run_arena(N_TEST, b"", b"", DEVICE_STR, MAX_RND, round_num, pt1, pt2, 2)
            results.append((round_num, diff, 0, res_d2))

    print("\n[Step 2] 単調回帰とPCHIP補間による勝率曲線のノイズ除去を実行しています...")
    # core.cpp の配列想定 [2, 9, 61] に合わせる (インデックス 1~6 を使用)
    win_prob_mat = np.zeros([2, 9, 61])
    
    # 最終的にマトリクスにマッピングする全点差（-30 から 30 までの 61 個の整数）
    target_diffs = np.arange(-30, 31)
    
    for d in [0, 1]:
        for round_num in range(1, MAX_RND + 1):
            round_results = [item for item in results if item[0] == round_num and item[2] == d]
            
            x_data = []
            y_data = []
            for _, diff, _, res in round_results:
                win_rate = (res['win'] + res['draw'] * 0.5) / N_TEST
                x_data.append(diff)
                y_data.append(win_rate)
                
            if x_data:
                # 昇順ソートを保証
                sort_idx = np.argsort(x_data)
                x_data = np.array(x_data)[sort_idx]
                y_data = np.array(y_data)[sort_idx]
                
                # 1. 単調回帰 (2点刻みの測定ノイズを滑らかにする)
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                y_iso = iso_reg.fit_transform(x_data, y_data)
                
                # 2. PCHIP補間 (偶数点差のデータを繋ぎ、奇数点差インデックスも自然に埋める)
                pchip = PchipInterpolator(x_data, y_iso)
                smoothed_y = pchip(target_diffs)
                
                # 0〜60のインデックス（diff + 30）に格納
                win_prob_mat[d, round_num, :61] = np.clip(smoothed_y, 0.0, 1.0)
            else:
                win_prob_mat[d, round_num, :] = 0.5

    print(f"\n更新された勝率テーブルを '{SAVE_PATH}' に保存します...")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(win_prob_mat, f)
        
    analyze_win_prob(win_prob_mat)
    
    print("\n[Step 4] ランダムな条件での勝率確認 (10サンプル)")
    import random
    for i in range(10):
        d = random.choice([0, 1])
        r = random.randint(1, MAX_RND)
        diff = random.randint(-30, 30)
        d_str = "親" if d == 1 else "子"
        idx = diff + 30
        prob = win_prob_mat[d, r, idx]
        print(f"  サンプル {i+1:2d} | ラウンド {r} | 役職: {d_str} | 点差: {diff:3d}点 -> 勝率: {prob:.4f}")
    
    print("\nすべての処理が完了しました！")


if __name__ == '__main__':
    main()