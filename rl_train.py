import os
import time
import pickle
import random
import threading
import concurrent.futures

import torch
import torch.multiprocessing as mp
import numpy as np

import koikoigame
from koikoigame import MAX_ROUND
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
setattr(my_linear, '_LinearWithBias', my_linear.Linear)

# --- 定数定義 ---
LOG_PATH = 'log.txt'
RL_FOLDER = 'model_agent'
START_LOOP_NUM = 0
LEARNING_RATE = 1e-4
BATCH_SIZE = 4096

CPU_COUNT = 2
LOOP_GAMES = 1024
N_CORE_GAMES = LOOP_GAMES // CPU_COUNT
CAP_D = LOOP_GAMES // CPU_COUNT * 68
CAP_P = LOOP_GAMES // CPU_COUNT * 9
CAP_K = LOOP_GAMES // CPU_COUNT * 8

N_LOOP_ACTION_NET_UPDATE = 10
N_LOOP_ARENA_TEST = 50
ARENA_WORKERS = 4

ARENA_OPPONENT_PATHS = {
    'discard': 'model_agent/discard_arena.pt',
    'pick': 'model_agent/pick_arena.pt',
    'koikoi': 'model_agent/koikoi_arena.pt'
}
PHASES = ['discard', 'pick', 'koikoi']

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE_STR = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- グローバル状態 ---
stop_event = threading.Event()

# --- 共通ユーティリティ関数 ---
def time_str():
    return time.strftime("%y%m%d %H%M%S", time.localtime())

def print_log(log_str, log_path=LOG_PATH):
    with open(log_path, 'a', encoding='utf-8') as f:
        print(log_str)
        print(log_str, file=f)

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
    models = {
        'discard': DiscardModel().cpu(),
        'pick': PickModel().cpu(),
        'koikoi': KoiKoiModel().cpu()
    }
    
    for key, model in models.items():
        path = ARENA_OPPONENT_PATHS[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"アリーナ対戦相手用のモデルが見つかりません: {path}")
            
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, torch.nn.Module):
            state_dict = state_dict.state_dict()
            
        model.load_state_dict(state_dict)
        _patch_model_attributes(model)
        
    return models['discard'], models['pick'], models['koikoi']

def get_value_action_net(action_net_path, value_net, action_model_class):
    """強化学習用のValueNetとActionNetを初期化・ロードする"""
    action_net = action_model_class().cpu() 
    
    if os.path.exists(action_net_path):
        try:
            # 永続化されたネイティブの state_dict をロード
            state_dict = torch.load(action_net_path, map_location='cpu', weights_only=True)
            if isinstance(state_dict, torch.nn.Module):
                state_dict = state_dict.state_dict()
                
            action_net.load_state_dict(state_dict)
            value_net.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: 重みの読み込みに失敗しました ({e})。初期状態から開始します。")
    else:
        print(f"[Init] チェックポイントが存在しないため、新規に初期化します: {action_net_path}")

    _patch_model_attributes(action_net)
    _patch_model_attributes(value_net)
                    
    return value_net, action_net

def init_worker():
    """アリーナテスト用の子プロセスの初期化"""
    seed = (os.getpid() * int(time.time() * 1000)) % 123456789
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    global master_agent
    m_discard, m_pick, m_koikoi = get_master_net()
    
    m_discard = m_discard.to(DEVICE).float().eval()
    m_pick = m_pick.to(DEVICE).float().eval()
    m_koikoi = m_koikoi.to(DEVICE).float().eval()
    
    master_agent = koikoilearn.Agent(m_discard, m_pick, m_koikoi)

def wait_for_exit_key():
    """バックグラウンドで終了コマンドを待機するスレッド用関数"""
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

def parallel_arena_test(state_dicts, n_games):
    """単一プロセスでのアリーナテスト実行"""
    # 子プロセス内でネイティブモデルをインスタンス化して重みをロード
    models = {
        'discard': DiscardModel().cpu(),
        'pick': PickModel().cpu(),
        'koikoi': KoiKoiModel().cpu()
    }
    
    for key in PHASES:
        models[key].load_state_dict(state_dicts[key])
        _patch_model_attributes(models[key])
        models[key] = models[key].to(DEVICE).eval()
        
    agent = koikoilearn.Agent(models['discard'], models['pick'], models['koikoi'])
        
    arena = koikoilearn.Arena(agent, master_agent)
    arena.multi_game_test(n_games)
    
    result = arena.test_win_num
    result.append(np.mean(arena.test_point[1])) # type: ignore
    return result

def test_result_analysis(result, loop):
    """アリーナテストの集計と結果出力"""
    result = np.array(result)
    win_num = np.sum(result[:, [0, 1, 2]], axis=0)
    win_rate = win_num / np.sum(win_num)
    score = win_rate[0] * 0.5 + win_rate[1]
    point = np.mean(result[:, 3])
    
    print_log(f'■■■  arena {loop:05}   win {int(win_num[1])}   lose {int(win_num[2])}  draw {int(win_num[0])}   score {point:.1f}pt  ■■■')
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

    # 1. ネイティブモデル（通常の nn.Module）の準備
    value_net, action_net = {}, {}
    value_net['discard'], action_net['discard'] = get_value_action_net(f'{RL_FOLDER}/discard_state.pt', TargetQNet().cpu(), DiscardModel)
    value_net['pick'], action_net['pick'] = get_value_action_net(f'{RL_FOLDER}/pick_state.pt', TargetQNet().cpu(), PickModel)
    value_net['koikoi'], action_net['koikoi'] = get_value_action_net(f'{RL_FOLDER}/koikoi_state.pt', TargetQNet().cpu(), KoiKoiModel)
    
    example_input_normal = torch.zeros((1, 300, 48), dtype=torch.float32, device=DEVICE)
    example_input_koikoi = torch.zeros((1, 300, 50), dtype=torch.float32, device=DEVICE) 

    # 2. メモリ上での JIT トレースオブジェクトの生成
    traced_action_net = {}
    traced_value_net = {}
    
    for key in PHASES:
        ex_input = example_input_koikoi if key == 'koikoi' else example_input_normal
        
        action_net[key].to(DEVICE).float().eval()
        traced_action_net[key] = torch.jit.trace(action_net[key], ex_input, check_trace=False)
        
        value_net[key].to(DEVICE).float().train()
        traced_value_net[key] = torch.jit.trace(value_net[key], ex_input, check_trace=False)

    # 3. C++ トレーナーの初期化に、メモリ上の JIT モデルを直接渡す
    trainer = {
        key: koikoicore.KoiKoiTrainer(traced_value_net[key]._c, LEARNING_RATE, DEVICE_STR)
        for key in PHASES
    }

    score = [0.0]
    wp_mat_np = win_prob_mat.astype(np.float32) if win_prob_mat is not None else np.zeros((2, 9, 61), dtype=np.float32)
    
    # 4. 初回の並列シミュレーション開始
    results = koikoicore.run_parallel_simulations(
        CPU_COUNT, N_CORE_GAMES, N_CORE_GAMES,          
        CAP_D, CAP_P, CAP_K, 1.0, wp_mat_np,
        traced_action_net['discard']._c, traced_action_net['pick']._c, traced_action_net['koikoi']._c,
        DEVICE_STR, MAX_ROUND
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def background_training(data):
        losses, samples = {}, {}
        for key in PHASES:
            losses[key] = trainer[key].train_from_results(data, key, BATCH_SIZE)
            samples[key] = sum(d.get(key, {}).get('actions', torch.empty(0)).shape[0] for d in data if isinstance(d, dict))
        return samples, losses
    
    # --- データを安全にコピーする関数 ---
    def clone_sim_results(results_list):
        cloned = []
        for r_dict in results_list:
            c_dict = {}
            for k, v in r_dict.items():
                if v['states'].numel() > 0:
                    c_dict[k] = {
                        'states': v['states'].clone(),
                        'actions': v['actions'].clone(),
                        'rewards': v['rewards'].clone()
                    }
                else:
                    c_dict[k] = v
            cloned.append(c_dict)
        return cloned
        
    for loop in range(START_LOOP_NUM, 10000):
        loop_start_time = time.perf_counter()
        sync_models = (loop % N_LOOP_ACTION_NET_UPDATE == 0)
        
        # バックグラウンド学習
        safe_results = clone_sim_results(results)
        future = executor.submit(background_training, safe_results)
        
        # フォアグラウンドシミュレーション
        next_results = koikoicore.run_parallel_simulations(
            CPU_COUNT, N_CORE_GAMES, N_CORE_GAMES,          
            CAP_D, CAP_P, CAP_K, 1.0, wp_mat_np,
            traced_action_net['discard']._c, traced_action_net['pick']._c, traced_action_net['koikoi']._c,
            DEVICE_STR, MAX_ROUND
        )
        
        samples, losses = future.result()
        results = next_results
        
        # モデルの同期処理 (C++の重み更新。Python側の traced_action_net にも即時自動反映される)
        if sync_models:
            for key in PHASES:
                trainer[key].sync_to_inference_model(key)
        
        elapsed_time = time.perf_counter() - loop_start_time
        print_log(f'loop {loop:05}  time {elapsed_time:04.1f}s    loss  discard {losses["discard"]:.2f}  pick {losses["pick"]:.2f}  koikoi {losses["koikoi"]:.2f}')
        
        # アリーナ評価
        if loop % N_LOOP_ARENA_TEST == 0:
            # JITモデルからテンソル(state_dict)のみを抽出・CPU化し、プロセス間通信のPickleエラーを回避
            current_state_dicts = {
                key: {k: v.cpu() for k, v in traced_action_net[key].state_dict().items()}
                for key in PHASES
            }
            
            with mp.Pool(ARENA_WORKERS, initializer=init_worker) as pool:
                result_async = [
                    pool.apply_async(parallel_arena_test, args=(current_state_dicts, 200 // ARENA_WORKERS)) 
                    for _ in range(ARENA_WORKERS)
                ]
                arena_results = [res.get() for res in result_async]
            
            torch.cuda.empty_cache()
            
            s = test_result_analysis(arena_results, loop)
            score.append(s)
            
        # 終了シグナル検知時の処理
        if stop_event.is_set():
            print_log(f'\n{time_str()} チェックポイントを保存中...')
            for key in PHASES:
                # C++の学習結果は traced_value_net にオンメモリで反映済みなので
                # 直接 state_dict を取り出して保存するだけでOK
                torch.save(traced_value_net[key].state_dict(), f'{RL_FOLDER}/{key}_state.pt')
                
            koikoicore.destroy_sim_manager()
            print_log(f'{time_str()} プログラムを安全に終了しました。')
            break