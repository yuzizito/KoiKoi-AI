import torch.nn.modules.linear as my_linear
setattr(my_linear, '_LinearWithBias', my_linear.Linear)

import torch
import numpy as np
import random
from collections import namedtuple

import os
import time
import pickle
import torch.multiprocessing as mp
import threading
import multiprocessing
import concurrent.futures

import koikoigame
import koikoilearn
import koikoicore
from koikoinet2L import DiscardModel, PickModel, KoiKoiModel, TargetQNet

# --- 環境設定・スレッド制御 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
torch.set_num_threads(1)
import torch.nn.modules.linear as my_linear

# --- 定数定義（ファイル先頭へ集約） ---
LOG_PATH = 'log_rl.txt'
RL_FOLDER = 'model_rl'
START_LOOP_NUM = 0
LEARNING_RATE = 1e-4
BATCH_SIZE = 4096
CPU_COUNT = 2
LOOP_GAMES = 1024
N_CORE_GAMES = LOOP_GAMES // CPU_COUNT
CAP_D = LOOP_GAMES // CPU_COUNT * 72
CAP_P = LOOP_GAMES // CPU_COUNT * 12
CAP_K = LOOP_GAMES // CPU_COUNT * 8
N_LOOP_ACTION_NET_UPDATE = 10
N_LOOP_ARENA_TEST = 50
ARENA_WORKERS = 4

SAVED_MODEL_PATH = {
    'discard': f'{RL_FOLDER}/discard_state.pt', 
    'pick': f'{RL_FOLDER}/pick_state.pt',
    'koikoi': f'{RL_FOLDER}/koikoi_state.pt'
}

ARENA_OPPONENT_PATHS = {
    'discard': 'model_agent/discard.pt',
    'pick': 'model_agent/pick.pt',
    'koikoi': 'model_agent/koikoi.pt'
}

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE_STR = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- グローバル状態 ---
stop_event = threading.Event()
win_prob_mat = None

TraceSlot = namedtuple('TraceSlot', ['key','state','action'])
Transition = namedtuple('Transition', ['state','action','reward'])


# --------------------------------------------------------------　検証用　要削除

def get_card_domain_info(card_idx):
    """カードインデックスからドメイン情報(月と札種別)を返す"""
    suit = card_idx // 4 + 1
    brights = {0, 8, 28, 40, 44}
    seeds = {4, 12, 16, 20, 24, 29, 32, 36, 41}
    ribbons = {1, 5, 9, 13, 17, 21, 25, 33, 37, 42}
    
    if card_idx in brights: type_str = "BRIGHTS"
    elif card_idx in seeds: type_str = "SEEDS  "
    elif card_idx in ribbons: type_str = "RIBBONS"
    else: type_str = "WASTE  "
    
    return f"{suit:02d}M-{type_str}"

# --------------------------------------------------------------


# --- 共通ユーティリティ関数 ---
def time_str():
    return time.strftime("%y%m%d %H%M%S", time.localtime())

def print_log(log_str, log_path=LOG_PATH):
    with open(log_path, 'a', encoding='utf-8') as f:
        print(log_str)
        print(log_str, file=f)
    return

def _patch_model_attributes(model):
    """古いバージョンのPyTorchモデルとの互換性を維持するための属性パッチ"""
    for module in model.modules():
        if type(module).__name__ == 'MultiheadAttention':
            if not hasattr(module, 'batch_first'):
                module.batch_first = False
        elif type(module).__name__ == 'TransformerEncoderLayer':
            if not hasattr(module, 'norm_first'):
                module.norm_first = False
    return model

def get_master_net():
    """アリーナ評価のベンチマーク（対戦相手）モデルをロードする"""
    map_location = torch.device('cpu')
    
    models = {
        'discard': DiscardModel().cpu(),
        'pick': PickModel().cpu(),
        'koikoi': KoiKoiModel().cpu()
    }
    
    for key, model in models.items():
        path = ARENA_OPPONENT_PATHS[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"アリーナ対戦相手用のモデルが見つかりません: {path}")
            
        loaded_data = torch.load(path, map_location=map_location, weights_only=False)
        
        if isinstance(loaded_data, torch.nn.Module):
            model.load_state_dict(loaded_data.state_dict())
        else:
            model.load_state_dict(loaded_data)
            
        _patch_model_attributes(model)
        
    return models['discard'], models['pick'], models['koikoi']

def get_value_action_net(action_net_path, value_net, action_model_class):
    action_net = action_model_class().cpu() 
    
    if os.path.exists(action_net_path):
        try:
            # 永続化されたネイティブの state_dict をロード
            state_dict = torch.load(action_net_path, map_location='cpu', weights_only=True)
            # もしモデル丸ごと保存されていた場合へのフォールバック
            if isinstance(state_dict, torch.nn.Module):
                state_dict = state_dict.state_dict()
                
            action_net.load_state_dict(state_dict)
            value_net.load_state_dict(state_dict)
            print(f"[Loadeed] ネイティブモデルの重みを復元しました: {action_net_path}")
        except Exception as e:
            print(f"Warning: 重みの読み込みに失敗しました ({e})。初期状態から開始します。")
    else:
        print(f"[Init] チェックポイントが存在しないため、新規に初期化します: {action_net_path}")

    _patch_model_attributes(action_net)
    _patch_model_attributes(value_net)
                    
    return value_net, action_net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_worker():
    seed = (os.getpid() * int(time.time() * 1000)) % 123456789
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    global master_agent
    master_discard_net, master_pick_net, master_koikoi_net = get_master_net()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    master_discard_net = master_discard_net.to(device).float().eval()
    master_pick_net = master_pick_net.to(device).float().eval()
    master_koikoi_net = master_koikoi_net.to(device).float().eval()
    
    master_agent = koikoilearn.Agent(master_discard_net, master_pick_net, master_koikoi_net)

def wait_for_exit_key():
    print("\n[システム] 画面上で 'q' キーを押して Enter を入力すると停止します。\n")
    while True:
        try:
            user_input = input()
            if user_input.strip().lower() == 'q':
                print("\n[停止シグナル検知] モデルを保存して終了します。")
                stop_event.set()
                break
        except Exception:
            break

def parallel_arena_test(agent, n_games):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for key in agent.model.keys():
        agent.model[key] = agent.model[key].to(device)
        agent.model[key].eval()
        
    arena = koikoilearn.Arena(agent, master_agent)
    arena.multi_game_test(n_games)
    result = arena.test_win_num
    result.append(np.mean(arena.test_point[1]))
    
    return result

def test_result_analysis(result,loop):
    result = np.array(result)
    win_num = np.sum(result[:,[0,1,2]],0)
    win_rate = win_num / np.sum(win_num)
    score = win_rate[0]*0.5 + win_rate[1]
    point = np.mean(result[:,3])
    print_log(f'■■■  arena {loop:05}   win {int(win_num[1])}   lose {int(win_num[2])}  draw {int(win_num[0])}   score {point:.1f}pt  ■■■', LOG_PATH)
    return score

if __name__ == '__main__':
    if not os.path.isdir(RL_FOLDER):
        os.mkdir(RL_FOLDER)
        
    exit_thread = threading.Thread(target=wait_for_exit_key, daemon=True)
    exit_thread.start()
        
    with open('win_prob_mat.pkl', 'rb') as f:
        win_prob_mat = pickle.load(f)

    master_discard_net, master_pick_net, master_koikoi_net = get_master_net()
    master_agent = koikoilearn.Agent(master_discard_net, master_pick_net, master_koikoi_net)


    # === ①起動時: 固定の100局面（discard）を抽出・保存（修正版） ===
    print_log(">>> 固定局面セット（100局面）を生成中...", LOG_PATH)
    fixed_discard_states = []
    dummy_game = koikoigame.KoiKoiGameState()
    while len(fixed_discard_states) < 100:
        if dummy_game.game_over:
            dummy_game = koikoigame.KoiKoiGameState()
            continue
        if dummy_game.round_state.round_over:
            dummy_game.new_round()
            continue
            
        # アクション待ち状態の時だけ、かつ 'discard' の時だけ局面を記録
        if dummy_game.round_state.wait_action:
            if dummy_game.round_state.state == 'discard':
                feat_cpu = dummy_game.feature_tensor.unsqueeze(0).clone()
                mask_np = dummy_game.round_state.action_mask.copy()
                fixed_discard_states.append((feat_cpu, mask_np))
            
            # 有効なランダム行動を取得
            action = master_agent.auto_random_action(dummy_game)
        else:
            # アクション待ちでないフェーズ（山札処理など）は None で進める
            action = None
            
        # 常にステップを進める
        dummy_game.round_state.step(action)
            
    prev_fixed_actions = []
    # ======================================================================


    # ==========================================
    # 1. モデルの準備 (ValueNetとActionNetの両方を保持)
    # ==========================================
    value_net, action_net = {}, {}
    # 拡張子を .pt (JIT用) から、ネイティブ用の .pth もしくは _state.pt に変更して管理
    value_net['discard'], action_net['discard'] = get_value_action_net(f'{RL_FOLDER}/discard_state.pt', TargetQNet().cpu(), DiscardModel)
    value_net['pick'], action_net['pick'] = get_value_action_net(f'{RL_FOLDER}/pick_state.pt', TargetQNet().cpu(), PickModel)
    value_net['koikoi'], action_net['koikoi'] = get_value_action_net(f'{RL_FOLDER}/koikoi_state.pt', TargetQNet().cpu(), KoiKoiModel)
    
    example_input_normal = torch.zeros((1, 300, 48), dtype=torch.float32, device=DEVICE)
    example_input_koikoi = torch.zeros((1, 300, 50), dtype=torch.float32, device=DEVICE) 

    # ==========================================
    # 2. 【追加】C++エンジンに渡すための一時JITモデルの初期エクスポート
    # ==========================================
    def export_temporary_jit_models():
        for key in ['discard', 'pick', 'koikoi']:
            action_net[key].to(DEVICE).float().eval()
            value_net[key].to(DEVICE).float().train()
            
        with torch.inference_mode():
            torch.jit.trace(action_net['discard'], example_input_normal, check_trace=False).save("traced_discard.pt") # type: ignore
            torch.jit.trace(action_net['pick'], example_input_normal, check_trace=False).save("traced_pick.pt") # type: ignore
            torch.jit.trace(action_net['koikoi'], example_input_koikoi, check_trace=False).save("traced_koikoi.pt") # type: ignore
            
        torch.jit.trace(value_net['discard'], example_input_normal, check_trace=False).save("traced_value_discard.pt") # type: ignore
        torch.jit.trace(value_net['pick'], example_input_normal, check_trace=False).save("traced_value_pick.pt") # type: ignore
        torch.jit.trace(value_net['koikoi'], example_input_koikoi, check_trace=False).save("traced_value_koikoi.pt") # type: ignore
        
        for key in ['discard', 'pick', 'koikoi']:
            action_net[key].cpu()
            value_net[key].cpu()

    export_temporary_jit_models()
    
    # ==========================================
    # 3. 学習用モデル (ValueNet) のトレース (C++トレーナー用)
    # ==========================================
    
    for key in ['discard', 'pick', 'koikoi']:
        value_net[key] = value_net[key].to(device).float().train()

    traced_val_discard = torch.jit.trace(value_net['discard'], example_input_normal, check_trace=False)
    traced_val_pick    = torch.jit.trace(value_net['pick'], example_input_normal, check_trace=False)
    traced_val_koikoi  = torch.jit.trace(value_net['koikoi'], example_input_koikoi, check_trace=False)
    
    traced_val_discard.save("traced_value_discard.pt") # type: ignore
    traced_val_pick.save("traced_value_pick.pt")       # type: ignore
    traced_val_koikoi.save("traced_value_koikoi.pt")   # type: ignore  
    
    # ==========================================
    # 4. C++ トレーナーの初期化
    # ==========================================
    trainer = {
        key: koikoicore.KoiKoiTrainer(f"traced_value_{key}.pt", LEARNING_RATE, DEVICE_STR)
        for key in ['discard', 'pick', 'koikoi']
    }

    score = [0.0]
    
    # ==========================================
    # 5. 強化学習 メインループ (非同期パイプライン版)
    # ==========================================
    
    # 最初の1回だけシミュレーションを事前に行い、初期データを生成（パイプラインの準備）
    print_log(">>> パイプラインの準備中（初期データの生成）...", LOG_PATH)
    wp_mat_np = win_prob_mat.astype(np.float32) if win_prob_mat is not None else np.zeros((2, 9, 61), dtype=np.float32)
    
    results = koikoicore.run_parallel_simulations(
        CPU_COUNT, N_CORE_GAMES, N_CORE_GAMES,          
        CAP_D, CAP_P, CAP_K, 1.0, wp_mat_np,
        "traced_discard.pt", "traced_pick.pt", "traced_koikoi.pt", DEVICE_STR
    )

    # 非同期実行用のスレッドプールを用意
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def background_training(data):
        """バックグラウンドで学習を実行し、サンプル数とロスを計算して返す"""
        l_d = trainer['discard'].train_from_results(data, 'discard', BATCH_SIZE)
        l_p = trainer['pick'].train_from_results(data, 'pick', BATCH_SIZE)
        l_k = trainer['koikoi'].train_from_results(data, 'koikoi', BATCH_SIZE)
        
        # ★修正: Pythonの辞書/リスト展開負荷を減らし、テンソルのサイズ(shape)から直接取得する
        s_d = sum(d.get('discard', {}).get('actions', torch.empty(0)).shape[0] for d in data if isinstance(d, dict))
        s_p = sum(d.get('pick', {}).get('actions', torch.empty(0)).shape[0] for d in data if isinstance(d, dict))
        s_k = sum(d.get('koikoi', {}).get('actions', torch.empty(0)).shape[0] for d in data if isinstance(d, dict))
        return (s_d, l_d), (s_p, l_p), (s_k, l_k)
        
    for loop in range(START_LOOP_NUM, 10000):
        loop_start_time = time.perf_counter()
        
        wp_mat_np = win_prob_mat.astype(np.float32) if win_prob_mat is not None else np.zeros((2, 9, 61), dtype=np.float32)
        sync_models = (loop % N_LOOP_ACTION_NET_UPDATE == 0)
        
        # 1. 【非同期】前回のデータを使って、バックグラウンドで学習（GPU）を開始！
        future = executor.submit(background_training, results)
        
        # 2. 【同期】同時に、フォアグラウンドで次回のシミュレーション（CPU+GPU推論）を実行！
        next_results = koikoicore.run_parallel_simulations(
            CPU_COUNT, N_CORE_GAMES, N_CORE_GAMES,          
            CAP_D, CAP_P, CAP_K, 1.0, wp_mat_np,
            "traced_discard.pt", "traced_pick.pt", "traced_koikoi.pt", DEVICE_STR
        )
        
        # 3. 学習スレッドの完了を待ち、結果（ロス等）を受け取る
        (s_d, l_d), (s_p, l_p), (s_k, l_k) = future.result()
        
        # 次のループのためにデータを入れ替え
        results = next_results
        
        # 4. モデルの同期処理（シミュレーションも学習も終わった安全なタイミングで実行）
        if sync_models:
            for key in ['discard', 'pick', 'koikoi']:
                # ★CPU負荷削減の究極手: ファイル保存・ロードを完全に廃止し、
                # C++内でGPUメモリ上の重みを推論用モデルへ直接コピー(0.01秒以下で完了)
                trainer[key].sync_to_inference_model(key)
        
        elapsed_time = time.perf_counter() - loop_start_time
        
        print_log(f'loop {loop:05}  time {elapsed_time:02.1f}s    sample:loss  discard {s_d:05}:{l_d:.2f}  pick {s_p:05}:{l_p:.2f}  koikoi {s_k:05}:{l_k:.2f}', LOG_PATH)
        
        # 5. アリーナ評価
        if loop % N_LOOP_ARENA_TEST == 0:
            
            for key in ['discard', 'pick', 'koikoi']:
                trainer[key].save_model(f"traced_{key}.pt")
                updated_jit = torch.jit.load(f"traced_{key}.pt", map_location='cpu')
                action_net[key].load_state_dict(updated_jit.state_dict())
            
            test_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'])
            

            # === ★検証用ログ出力の開始 ===
            for key in test_agent.model.keys():
                test_agent.model[key] = test_agent.model[key].to(DEVICE).float()

            current_top1_actions = []
            q_diffs = []
            
            print_log(f"\n--- [Loop {loop:05}] Action Analysis ---", LOG_PATH)
            
            for i, (feat_cpu, mask_np) in enumerate(fixed_discard_states):
                feat_gpu = test_agent._move_to_gpu(feat_cpu)
                q_details = test_agent.get_q_details('discard', feat_gpu, mask_np)
                
                top1_action = q_details[0]['action']
                top1_q = q_details[0]['q']
                current_top1_actions.append(top1_action)
                
                if len(q_details) > 1:
                    top2_q = q_details[1]['q']
                    diff = top1_q - top2_q
                else:
                    diff = 0.0
                q_diffs.append(diff)
                
                # 代表局面(最初の15局)の詳細出力
                if i < 15:
                    log_str = f" 局{i+1:02d}: "
                    for rank, d in enumerate(q_details):
                        act = d['action']
                        q = d['q']
                        info = get_card_domain_info(act)
                        log_str += f"[T{rank+1}] 札{act:02d} {info} (Q={q:6.3f}) | "
                    if len(q_details) > 1:
                        log_str += f"差={diff:6.3f}"
                    print_log(log_str, LOG_PATH)
            
            # 統計情報の出力
            if len(prev_fixed_actions) > 0:
                changed = sum(1 for a, b in zip(current_top1_actions, prev_fixed_actions) if a != b)
                churn_rate = (changed / len(fixed_discard_states)) * 100
                print_log(f"★ Policy Churn (行動変化率): {churn_rate:.1f}%", LOG_PATH)
            else:
                print_log("★ Policy Churn: 初期記録のため算出なし", LOG_PATH)
                
            prev_fixed_actions = current_top1_actions
            
            if len(q_diffs) > 0:
                mean_d = np.mean(q_diffs)
                med_d = np.median(q_diffs)
                p90_d = np.percentile(q_diffs, 90)
                max_d = np.max(q_diffs)
                print_log(f"★ Top1-Top2差分: Mean={mean_d:.4f} | Median={med_d:.4f} | P90={p90_d:.4f} | Max={max_d:.4f}", LOG_PATH)
            
            for key in test_agent.model.keys():
                test_agent.model[key] = test_agent.model[key].cpu()
            # ============================================
            
            
            result = []
            pool = mp.Pool(ARENA_WORKERS, initializer=init_worker)
            for _ in range(ARENA_WORKERS):
                result.append(pool.apply_async(parallel_arena_test, args=(test_agent, 200//ARENA_WORKERS)))
            pool.close()
            pool.join()
            
            torch.cuda.empty_cache()
            del pool
            
            s = test_result_analysis([res.get() for res in result], loop)
            score.append(s)
            
        # 終了シグナル検知時の処理
        if stop_event.is_set():
            print_log(f'\n{time_str()} チェックポイントを保存中...', LOG_PATH)
            
            for key in ['discard', 'pick', 'koikoi']:
                trainer[key].save_model(f"traced_value_{key}.pt")
                updated_jit = torch.jit.load(f"traced_value_{key}.pt", map_location='cpu')
                action_net[key].load_state_dict(updated_jit.state_dict())
                
                torch.save(action_net[key].state_dict(), f'{RL_FOLDER}/{key}_state.pt')
                
            koikoicore.destroy_sim_manager()
            print_log(f'{time_str()} プログラムを安全に終了しました。', LOG_PATH)
            break