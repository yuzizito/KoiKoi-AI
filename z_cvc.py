import os

# --- OpenMPのDLL衝突回避とスレッド数制御 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '15' # PCの環境に合わせて調整してください
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import koikoicore
from neurd import NeuRD

import torch.nn.modules.linear as my_linear
setattr(my_linear, '_LinearWithBias', my_linear.Linear)

# 以前のパス設定を保持しています。必要に応じて 'model_v2/...' 等に変更してください。
A_PATHS = {'discard': 'model_v2/discard.pt', 'pick': 'model_v2/pick.pt', 'koikoi': 'model_v2/koikoi.pt'}
B_PATHS = {'discard': 'model_v2/arena/discard.pt', 'pick': 'model_v2/arena/pick.pt', 'koikoi': 'model_v2/arena/koikoi.pt'}

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE_STR = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_ROUND = 6

def load_and_trace_models(paths):
    """指定されたパスからNeuRDを読み込み、C++推論用のJITトレースモデルを構築する"""
    traced_models = {}
    example_input_normal = torch.zeros((1, 24, 48), dtype=torch.float32, device=DEVICE)
    example_input_koikoi = torch.zeros((1, 24, 48), dtype=torch.float32, device=DEVICE)
    
    for key in ['discard', 'pick', 'koikoi']:
        is_kk = (key == 'koikoi')
        model = NeuRD(is_koikoi=is_kk).to(DEVICE)
        
        # Validation: モデルファイルが存在しない場合のフォールバック
        if os.path.exists(paths[key]):
            state_dict = torch.load(paths[key], map_location=DEVICE, weights_only=True)
            if isinstance(state_dict, torch.nn.Module):
                state_dict = state_dict.state_dict()
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: モデルが見つからないため、初期状態を使用します: {paths[key]}")
            
        # Design Intent: setattrを使用することで静的型チェッカーの誤認警告を安全に回避する
        for module in model.modules():
            if type(module).__name__ == 'MultiheadAttention':
                setattr(module, 'batch_first', False)
            elif type(module).__name__ == 'TransformerEncoderLayer':
                setattr(module, 'norm_first', False)
                
        model = model.float().eval()
        
        # C++バックエンド(BatchSimulator)へ渡すためのJITトレース
        ex_input = example_input_koikoi if is_kk else example_input_normal
        traced_models[key] = torch.jit.trace(model, ex_input, check_trace=False)
        
    return traced_models

print("[A vs B] モデルをロード・コンパイル中...")
agent_a_traced = load_and_trace_models(A_PATHS)
agent_b_traced = load_and_trace_models(B_PATHS)

total_games = 512
# Validation: 先手後手を平等にするDuplicate Matchのために、必ず偶数(ペア)へ切り上げる
if total_games % 2 != 0:
    total_games += 1

print(f"\n[A vs B] C++バックエンドによる高速対戦を開始します ({total_games} games)...")
# Design Intent: 巨大バッチでGPUを最大活用するため、1手ごとのJSON棋譜書き出しは行いません

arena_res = koikoicore.run_arena_batch_simulations(
    total_games,
    agent_a_traced['discard']._c, agent_a_traced['pick']._c, agent_a_traced['koikoi']._c,
    agent_b_traced['discard']._c, agent_b_traced['pick']._c, agent_b_traced['koikoi']._c,
    DEVICE_STR, MAX_ROUND
)

w, l, d = arena_res["win"], arena_res["lose"], arena_res["draw"]
avg_pt = arena_res["score"] / total_games if total_games > 0 else 0

print(f"\n対戦結果 -> A の勝利数: {w} / B の勝利数: {l} / 引き分け: {d} / Aの平均スコア: {avg_pt:.1f}pt")