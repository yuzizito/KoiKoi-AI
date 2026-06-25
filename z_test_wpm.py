import pickle
import numpy as np

def analyze_win_prob(path='win_prob_mat.pkl'):
    with open(path, 'rb') as f:
        mat = pickle.load(f)
    
    print(f"データ形状: {mat.shape} (期待値: [2, 9, 61])")
    
    # --- 1. 境界値チェック ---
    print("\n【1. 境界値のチェック】")
    for r in [1, 4, 8]:
        p_0 = mat[1, r, 0]   # 親・0点
        p_30 = mat[1, r, 30] # 親・30点 (基準点)
        p_60 = mat[1, r, 60] # 親/*60点
        print(f"ラウンド {r} (親) -> 0点: {p_0:.4f} | 30点: {p_30:.4f} | 60点: {p_60:.4f}")
    
    # --- 2. 単調増加性のチェック ---
    print("\n【2. 単調増加性 (右肩上がりか) のチェック】")
    monotonic_errors = 0
    for d in [0, 1]:
        for r in range(1, 9):
            # 前の点数より勝率が下がっている箇所がないか検証
            diff = np.diff(mat[d, r, :])
            if np.any(diff < -1e-5):  # 浮動小数点の誤差を考慮
                monotonic_errors += 1
    if monotonic_errors == 0:
        print("✅ パス: すべての条件で点数とともに勝率が滑らかに増加しています。")
    else:
        print(f"❌ 警告: 勝率が途中で減少している箇所が {monotonic_errors} 箇所あります。")

    # --- 3. 親の優位性チェック ---
    print("\n【3. 親（ディーラー）優位性のチェック】")
    dealer_errors = 0
    for r in range(1, 9):
        # 1〜59点の間で、親(1)の勝率が子(0)の勝率以上であることを確認
        if np.any(mat[1, r, 1:60] < mat[0, r, 1:60]):
            dealer_errors += 1
    if dealer_errors == 0:
        print("✅ パス: すべてのラウンドで「親の勝率 ＞ 子の勝率」が成立しています。")
    else:
        print(f"❌ 警告: 子の勝率の方が高くなっているラウンドが {dealer_errors} つあります。")

    # --- 4. 残り時間（ラウンド）による感度チェック ---
    print("\n【4. ラウンド進行による逃げ切り難易度のチェック】")
    # 40点（リード時）の勝率がラウンドとともに上がっているか
    p_40_r1 = mat[1, 1, 40]
    p_40_r8 = mat[1, 8, 40]
    print(f"40点保有時の勝率 -> 第1ラウンド: {p_40_r1:.4f} ➔ 第8ラウンド: {p_40_r8:.4f}")
    if p_40_r8 > p_40_r1:
        print("✅ パス: 終盤になるほどリードしている側の勝率が正しく1に収束しています。")
    else:
        print("❌ 警告: ラウンド進行による勝率の収束が正しく行われていません。")

if __name__ == '__main__':
    analyze_win_prob()