import os
import time
import pickle
import random
import threading
import concurrent.futures

import torch
import torch.multiprocessing as mp
import numpy as np

from koikoigame import MAX_ROUND
import koikoilearn
import koikoicore

from koikoinet_v2 import DiscardModel, PickModel, KoiKoiModel, TargetQNet

# --- 環境設定・スレッド制御 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '15'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
torch.set_num_threads(1)

import torch.nn.modules.linear as my_linear
setattr(my_linear, '_LinearWithBias', my_linear.Linear)

# --- 定数定義 ---
START_LOOP_NUM = 0
LEARNING_RATE = 1e-5
BATCH_SIZE = 4096

CPU_COUNT = 1
LOOP_GAMES = 256
N_CORE_GAMES = LOOP_GAMES // CPU_COUNT
CAP_D = LOOP_GAMES // CPU_COUNT * 180
CAP_P = LOOP_GAMES // CPU_COUNT * 70
CAP_K = LOOP_GAMES // CPU_COUNT * 70

N_LOOP_ACTION_NET_UPDATE = 20
N_LOOP_ARENA_TEST = 50
ARENA_WORKERS = 4

MAX_POOL_SIZE = 16
SAMPLE_PER_THREAD = 800

MODEL_FOLDER = 'model_v2'
OPPONENT_PATHS = {
    'discard': 'model_v2/arena/discard.pt',
    'pick': 'model_v2/arena/pick.pt',
    'koikoi': 'model_v2/arena/koikoi.pt'
}
LOG_PATH = 'log.txt'
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
        path = OPPONENT_PATHS[key]
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
    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
        
    exit_thread = threading.Thread(target=wait_for_exit_key, daemon=True)
    exit_thread.start()
        
    with open('win_prob_mat.pkl', 'rb') as f:
        win_prob_mat = pickle.load(f)

    master_discard_net, master_pick_net, master_koikoi_net = get_master_net()
    master_agent = koikoilearn.Agent(master_discard_net, master_pick_net, master_koikoi_net)

    value_net, action_net = {}, {}
    value_net['discard'], action_net['discard'] = get_value_action_net(f'{MODEL_FOLDER}/discard.pt', TargetQNet().cpu(), DiscardModel)
    value_net['pick'], action_net['pick'] = get_value_action_net(f'{MODEL_FOLDER}/pick.pt', TargetQNet().cpu(), PickModel)
    value_net['koikoi'], action_net['koikoi'] = get_value_action_net(f'{MODEL_FOLDER}/koikoi.pt', TargetQNet().cpu(), KoiKoiModel)
    
    example_input_normal = torch.zeros((1, 24, 48), dtype=torch.float32, device=DEVICE)
    example_input_koikoi = torch.zeros((1, 24, 48), dtype=torch.float32, device=DEVICE)

    traced_action_net = {}
    traced_value_net = {}
    
    for key in PHASES:
        ex_input = example_input_koikoi if key == 'koikoi' else example_input_normal
        
        action_net[key].to(DEVICE).float().eval()
        traced_action_net[key] = torch.jit.trace(action_net[key], ex_input, check_trace=False)
        
        value_net[key].to(DEVICE).float().train()
        traced_value_net[key] = torch.jit.trace(value_net[key], ex_input, check_trace=False)

    trainer = {
        key: koikoicore.KoiKoiTrainer(traced_value_net[key]._c, LEARNING_RATE, DEVICE_STR)
        for key in PHASES
    }

    score = [0.0]
    wp_mat_np = win_prob_mat.astype(np.float32) if win_prob_mat is not None else np.zeros((2, 9, 61), dtype=np.float32)
    
    results = koikoicore.run_parallel_simulations(
        CPU_COUNT, N_CORE_GAMES, N_CORE_GAMES,          
        CAP_D, CAP_P, CAP_K, 1.0, wp_mat_np,
        traced_action_net['discard']._c, traced_action_net['pick']._c, traced_action_net['koikoi']._c,
        DEVICE_STR, MAX_ROUND, 0
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def background_training(data):
        losses, samples = {}, {}
        for key in PHASES:
            state_list = []
            reward_list = []
            
            # Design Intent: Python側でデータを抽出し、PyTorchの高速な結合処理を利用する
            for d in data:
                if isinstance(d, dict) and key in d:
                    s = d[key].get('states')
                    r = d[key].get('rewards')
                    
                    # Validation: テンソルが存在し、かつ空でないことを確認
                    if s is not None and s.numel() > 0:
                        state_list.append(s)
                        reward_list.append(r)
            
            if state_list:
                all_states = torch.cat(state_list, dim=0)
                all_rewards = torch.cat(reward_list, dim=0)
                
                # Design Intent: 純粋なテンソルのみを渡すことで、C++側は即座にGILを解放して計算に専念できる
                losses[key] = trainer[key].train_epoch(all_states, all_rewards, BATCH_SIZE)
                samples[key] = all_states.shape[0]
            else:
                losses[key] = 0.0
                samples[key] = 0
                
        return samples, losses
    
    replay_buffer_pool = []

    def extract_sampled_dict(r_dict, sample_size):
        sampled = {}
        for k in PHASES:
            if k not in r_dict: continue
            states = r_dict[k]['states']
            actions = r_dict[k]['actions']
            rewards = r_dict[k]['rewards']
            total_n = states.shape[0]
            if total_n == 0:
                sampled[k] = r_dict[k]
                continue
            
            n_select = min(sample_size, total_n)
            perm = torch.randperm(total_n)[:n_select]
            sampled[k] = {
                'states': states[perm].clone(),
                'actions': actions[perm].clone(),
                'rewards': rewards[perm].clone()
            }
        return sampled
        
    for loop in range(START_LOOP_NUM, 10000):
        loop_start_time = time.perf_counter()
        sync_models = (loop % N_LOOP_ACTION_NET_UPDATE == 0)
        
        training_data = list(results)
        for pool_data in replay_buffer_pool:
            training_data.extend(pool_data)
        
        future = executor.submit(background_training, training_data)
        
        next_results = koikoicore.run_parallel_simulations(
            CPU_COUNT, N_CORE_GAMES, N_CORE_GAMES,          
            CAP_D, CAP_P, CAP_K, 1.0, wp_mat_np,
            traced_action_net['discard']._c, traced_action_net['pick']._c, traced_action_net['koikoi']._c,
            DEVICE_STR, MAX_ROUND, loop
        )
        
        sampled_results = [extract_sampled_dict(r, SAMPLE_PER_THREAD) for r in results]
        replay_buffer_pool.append(sampled_results)
        if len(replay_buffer_pool) > MAX_POOL_SIZE:
            replay_buffer_pool.pop(0)
            
        samples, losses = future.result()
        results = next_results
        
        if sync_models:
            for key in PHASES:
                trainer[key].sync_to_inference_model(key)
        
        elapsed_time = time.perf_counter() - loop_start_time
        print_log(f'loop {loop:05}  time {elapsed_time:04.1f}s    loss:sample  discard {losses["discard"]:.2f}:{samples["discard"]:05}  pick {losses["pick"]:.2f}:{samples["pick"]:05}  koikoi {losses["koikoi"]:.2f}:{samples["koikoi"]:05}')
        
        if loop % N_LOOP_ARENA_TEST == 0:
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
            
        if stop_event.is_set():
            print_log(f'\n{time_str()} チェックポイントを保存中...')
            for key in PHASES:
                torch.save(traced_value_net[key].state_dict(), f'{MODEL_FOLDER}/{key}.pt')
                
            koikoicore.destroy_sim_manager()
            print_log(f'{time_str()} プログラムを安全に終了しました。')
            break