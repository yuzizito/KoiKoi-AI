import torch.nn.modules.linear as my_linear; my_linear._LinearWithBias = my_linear.Linear
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:06:35 2021

@author: guansanghai
"""

import torch
torch.set_float32_matmul_precision('high')
import numpy as np
import random
from collections import namedtuple

import os
import time
import pickle
import torch.multiprocessing as mp
import threading
from torch.utils.data import TensorDataset, DataLoader

import koikoigame
import koikoilearn
import koikoicore
from koikoinet2L import DiscardModel, PickModel, KoiKoiModel, TargetQNet

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
torch.set_num_threads(1)

stop_event = threading.Event()

log_path = 'log_rl_pbrs.txt'
rl_folder = 'model_rl'

# continue training with trained models
start_loop_num = 1
saved_model_path = {'discard':f'{rl_folder}/discard_final_stop.pt', 
                    'pick':f'{rl_folder}/pick_final_stop.pt',
                    'koikoi':f'{rl_folder}/koikoi_final_stop.pt'}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
win_prob_mat = None

TraceSlot = namedtuple('TraceSlot', ['key','state','action'])
Transition = namedtuple('Transition', ['state','action','reward'])

def time_str():
    return time.strftime("%y%m%d %H%M%S",time.localtime())

def print_log(log_str, log_path):
    with open(log_path, 'a', encoding='utf-8') as f:
        print(log_str)
        print(log_str, file=f)
    return

class Buffer():
    def __init__(self):
        self.memory = {'discard': {'states': [], 'actions': [], 'rewards': []}, 
                       'pick': {'states': [], 'actions': [], 'rewards': []}, 
                       'koikoi': {'states': [], 'actions': [], 'rewards': []}}
        self.sizes = {'discard': 0, 'pick': 0, 'koikoi': 0}
    
    def extend(self, data_dict):
        for key in ['discard', 'pick', 'koikoi']:
            if data_dict.get(key) is not None:
                self.memory[key]['states'].append(data_dict[key]['states'])
                self.memory[key]['actions'].append(data_dict[key]['actions'])
                self.memory[key]['rewards'].append(data_dict[key]['rewards'])
                self.sizes[key] += len(data_dict[key]['rewards'])
        return
    
    def clear(self):
        self.__init__()
        return

def get_master_net():
    map_location = torch.device('cpu')
    discard_model_path = 'model_agent/discard_sl.pt'
    pick_model_path = 'model_agent/pick_sl.pt'
    koikoi_model_path = 'model_agent/koikoi_sl.pt'

    discard_model = torch.load(discard_model_path, map_location, weights_only=False)
    for module in discard_model.modules():
        if type(module).__name__ == 'MultiheadAttention':
            if not hasattr(module, 'batch_first'):
                module.batch_first = False
        elif type(module).__name__ == 'TransformerEncoderLayer':
            if not hasattr(module, 'norm_first'):
                module.norm_first = False

    pick_model = torch.load(pick_model_path, map_location, weights_only=False)
    for module in pick_model.modules():
        if type(module).__name__ == 'MultiheadAttention':
            if not hasattr(module, 'batch_first'):
                module.batch_first = False
        elif type(module).__name__ == 'TransformerEncoderLayer':
            if not hasattr(module, 'norm_first'):
                module.norm_first = False

    koikoi_model = torch.load(koikoi_model_path, map_location, weights_only=False)  
    for module in koikoi_model.modules():
        if type(module).__name__ == 'MultiheadAttention':
            if not hasattr(module, 'batch_first'):
                module.batch_first = False
        elif type(module).__name__ == 'TransformerEncoderLayer':
            if not hasattr(module, 'norm_first'):
                module.norm_first = False
    return discard_model, pick_model, koikoi_model


def get_value_action_net(action_net_path, value_net, action_model_class):
    map_location = torch.device('cpu')
    action_net = action_model_class().cpu() 
    
    try:
        loaded_model = torch.load(action_net_path, map_location=map_location, weights_only=False)
        is_jit = isinstance(loaded_model, torch.jit.ScriptModule)
    except RuntimeError:
        try:
            loaded_model = torch.jit.load(action_net_path, map_location=map_location)
            is_jit = True
        except Exception:
            print(f"Warning: Could not load {action_net_path}. Initializing from scratch.")
            return value_net, action_net

    if is_jit:
        state_dict = loaded_model.state_dict()
        act_sd = action_net.state_dict()
        for k in act_sd.keys():
            if k in state_dict:
                act_sd[k].copy_(state_dict[k])
        action_net.load_state_dict(act_sd)
        
        val_sd = value_net.state_dict()
        for k in val_sd.keys():
            if k in state_dict:
                val_sd[k].copy_(state_dict[k])
        value_net.load_state_dict(val_sd)
    else:
        action_net = loaded_model
        value_net.load_state_dict(action_net.state_dict())

    # ★ [重要] batch_first と norm_first の整合性を確保（推論高速化・正常動作に必須）
    for net in [action_net, value_net]:
        for module in net.modules():
            if type(module).__name__ == 'MultiheadAttention':
                if not hasattr(module, 'batch_first'):
                    module.batch_first = False
            elif type(module).__name__ == 'TransformerEncoderLayer':
                if not hasattr(module, 'norm_first'):
                    module.norm_first = False
                    
    return value_net, action_net

def parallel_sampling(n_games, global_win_prob_mat):
    try:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        wp_mat_np = global_win_prob_mat.astype(np.float32) if global_win_prob_mat is not None else np.zeros((2,9,61), dtype=np.float32)

        # 超高速C++シミュレータの起動 (内部でPyTorchモデルを読み込み、自己対局を完走させる)
        sim = koikoicore.BatchSimulator(
            min(128, n_games), n_games, 
            n_games * 150, n_games * 150, n_games * 50,
            1.0, wp_mat_np,
            "traced_discard.pt", "traced_pick.pt", "traced_koikoi.pt", device_str
        )
        
        t_start = time.perf_counter()
        sim.play_games()  # ← ここでGPU推論を含めた全ゲームがC++ネイティブで実行される
        t_play = time.perf_counter() - t_start
        
        buffers = sim.finalize_buffers()
        
        clean_buffer = {'discard': None, 'pick': None, 'koikoi': None}
        for key in ['discard', 'pick', 'koikoi']:
            if buffers[key]['states'].shape[0] > 0:
                states = torch.from_numpy(buffers[key]['states'].astype(np.float16)).share_memory_()
                actions = torch.from_numpy(buffers[key]['actions']).share_memory_()
                rewards = torch.from_numpy(buffers[key]['rewards']).share_memory_()
                clean_buffer[key] = {'states': states, 'actions': actions, 'rewards': rewards}

        # ログ用のダミー時間データ
        clean_buffer['time_stats'] = [(t_play, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        return clean_buffer

    except Exception as e:
        raise e

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_worker():
    seed = (os.getpid() * int(time.time() * 1000)) % 123456789
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    global master_agent
    master_discard_net, master_pick_net, master_koikoi_net = get_master_net()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    master_discard_net = master_discard_net.to(device).bfloat16().eval()
    master_pick_net = master_pick_net.to(device).bfloat16().eval()
    master_koikoi_net = master_koikoi_net.to(device).bfloat16().eval()
    
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
    s = f'{time_str()} {sum(win_num)} games tested, '
    s += f'{win_num[1]} wins, {win_num[2]} loses, {win_num[0]} draws '
    s += f'({win_rate[1]:.2f}, {win_rate[2]:.2f}, {win_rate[0]:.2f}), '
    s += f'{point:.1f} points'
    print_log(s, log_path)
    print_log(f'Record,loop:{loop}, score:{score:.3f}, point:{point:.2f}', log_path)
    return score

def random_action_prob_scheduler(score):
    if score < 0.10: p = [0.25] * 4
    elif score < 0.20: p = [0.20] * 4
    elif score < 0.30: p = [0.15] * 4
    elif score < 0.40: p = [0.125] * 4
    elif score < 0.50: p = [0.10] * 4
    elif score < 0.55: p = [0.075] * 4
    else: p = [0.05] * 4
    return p

from collections import deque

if __name__ == '__main__':
    if not os.path.isdir(rl_folder):
        os.mkdir(rl_folder)
        
    exit_thread = threading.Thread(target=wait_for_exit_key, daemon=True)
    exit_thread.start()
        
    with open('win_prob_mat.pkl','rb') as f:
        win_prob_mat = pickle.load(f)

    master_discard_net, master_pick_net, master_koikoi_net = get_master_net()
    master_agent = koikoilearn.Agent(master_discard_net, master_pick_net, master_koikoi_net)
    
    cpu_count = 8
    loop_games = 480
    n_core_games = loop_games // cpu_count
    batch_size = 128
    
    n_loop_action_net_update = 5
    n_loop_arena_test = 20

    # ==========================================
    # 1. モデルの準備 (ValueNetとActionNetの両方を保持)
    # ==========================================
    
    value_net, action_net = {}, {}
    value_net['discard'], action_net['discard'] = get_value_action_net(saved_model_path['discard'], TargetQNet().cpu(), DiscardModel)
    value_net['pick'], action_net['pick'] = get_value_action_net(saved_model_path['pick'], TargetQNet().cpu(), PickModel)
    value_net['koikoi'], action_net['koikoi'] = get_value_action_net(saved_model_path['koikoi'], TargetQNet().cpu(), KoiKoiModel)
    example_input_normal = torch.zeros((1, 300, 48), dtype=torch.bfloat16, device=device) # ダミーデータ
    example_input_koikoi = torch.zeros((1, 300, 50), dtype=torch.bfloat16, device=device) # ダミーデータ
    
    # ==========================================
    # 2. 推論用モデル (ActionNet) のトレース (C++シミュレータ用)
    # ==========================================
    
    for key in ['discard', 'pick', 'koikoi']:
        action_net[key] = action_net[key].to(device).bfloat16().eval()

    with torch.inference_mode():
        traced_discard = torch.jit.trace(action_net['discard'], example_input_normal, check_trace=False)
        traced_pick    = torch.jit.trace(action_net['pick'], example_input_normal, check_trace=False)
        traced_koikoi  = torch.jit.trace(action_net['koikoi'], example_input_koikoi, check_trace=False)
        
        traced_discard.save("traced_discard.pt")
        traced_pick.save("traced_pick.pt")
        traced_koikoi.save("traced_koikoi.pt")

    for key in ['discard', 'pick', 'koikoi']:
        action_net[key] = action_net[key].cpu()
    
    # ==========================================
    # 3. 学習用モデル (ValueNet) のトレース (C++トレーナー用)
    # ==========================================
    
    for key in ['discard', 'pick', 'koikoi']:
        value_net[key] = value_net[key].to(device).bfloat16().train()

    traced_val_discard = torch.jit.trace(value_net['discard'], example_input_normal, check_trace=False)
    traced_val_pick    = torch.jit.trace(value_net['pick'], example_input_normal, check_trace=False)
    traced_val_koikoi  = torch.jit.trace(value_net['koikoi'], example_input_koikoi, check_trace=False)
    
    traced_val_discard.save("traced_value_discard.pt")
    traced_val_pick.save("traced_value_pick.pt")
    traced_val_koikoi.save("traced_value_koikoi.pt")

    play_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'],
                                   random_action_prob=[0.1, 0.1, 0.1, 0.1])    
    
    # ==========================================
    # 4. C++ トレーナーの初期化 (学習用モデルを読み込む)
    # ==========================================
    
    lr = 1e-5
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainer = {
        'discard': koikoicore.KoiKoiTrainer("traced_value_discard.pt", lr, device_str),
        'pick': koikoicore.KoiKoiTrainer("traced_value_pick.pt", lr, device_str),
        'koikoi': koikoicore.KoiKoiTrainer("traced_value_koikoi.pt", lr, device_str)
    }

    score = [0.0]
    print_log(f'\n{time_str()} start training on device: {device}', log_path)
    
    # ==========================================
    # 5. 強化学習 メインループ
    # ==========================================
    
    for loop in range(start_loop_num, 100000):
        loop_start_time = time.perf_counter()
        
        wp_mat_np = win_prob_mat.astype(np.float32) if win_prob_mat is not None else np.zeros((2,9,61), dtype=np.float32)
        sync_models = (loop % n_loop_action_net_update == 0)
        
        losses = koikoicore.run_simulation_and_train(
            cpu_count, n_core_games, n_core_games,          
            11000,  # cap_d (discard)
            1400,   # cap_p (pick)
            1600,   # cap_k (koikoi)
            1.0, wp_mat_np,
            "traced_discard.pt", "traced_pick.pt", "traced_koikoi.pt", device_str,
            trainer['discard'], trainer['pick'], trainer['koikoi'],
            batch_size, sync_models
        )
        
        elapsed_time = time.perf_counter() - loop_start_time
        
        s_d = losses["samples_discard"]
        l_d = losses["loss_discard"]
        s_p = losses["samples_pick"]
        l_p = losses["loss_pick"]
        s_k = losses["samples_koikoi"]
        l_k = losses["loss_koikoi"]
        
        print_log(f'{time_str()} ★ loop:{loop} (学習時間: {elapsed_time:.2f}s)', log_path)
        print_log(f'[sample,loss] : discard [{s_d},{l_d:.2f}], pick [{s_p},{l_p:.2f}], koikoi [{s_k},{l_k:.2f}]', log_path)
        
        if sync_models:
            print_log(f'{time_str()} Updated JIT models synced and saved natively.', log_path)
        
        # 4. アリーナ評価 (従来通り)
        if loop % n_loop_arena_test == 0:
            
            for key in ['discard', 'pick', 'koikoi']:
                action_net[key].load_state_dict(torch.jit.load(f"traced_{key}.pt", map_location='cpu').state_dict())
            
            test_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'])

            arena_workers = 4 
            
            result = []
            pool = mp.Pool(arena_workers, initializer=init_worker)
            for _ in range(arena_workers):
                result.append(pool.apply_async(parallel_arena_test, args=(test_agent, 400//arena_workers)))
            pool.close()
            pool.join()
            
            torch.cuda.empty_cache()
            del pool
            
            s = test_result_analysis([res.get() for res in result], loop)
            score.append(s)
            
            if s == max(score[-20:]):
                for key in ['discard', 'pick', 'koikoi']:
                    torch.save(action_net[key], f'{rl_folder}/{key}_{loop}_{round(s*100)}.pt')
                print_log(f'{time_str()} 最新モデル保存完了', log_path)
            
        # 5. 終了処理
        if stop_event.is_set():
            print_log(f'\n{time_str()} 最新モデル保存中', log_path)
            for key in ['discard', 'pick', 'koikoi']:
                trainer[key].save_model(f'{rl_folder}/{key}_final_stop.pt')
                
            koikoicore.destroy_sim_manager()
            print_log(f'{time_str()} プログラム終了', log_path)
            break