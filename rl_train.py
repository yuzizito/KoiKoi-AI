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
import multiprocessing
import threading
from torch.utils.data import TensorDataset, DataLoader

import koikoigame
import koikoilearn
from koikoinet2L import DiscardModel, PickModel, KoiKoiModel, TargetQNet

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# 終了を検知するためのシグナルオブジェクト
stop_event = threading.Event()

def wait_for_exit_key():
    print("\n[システム] 画面上で 'q' キーを押して Enter を入力すると、現在のループ終了時に安全に停止します。\n")
    while True:
        try:
            user_input = input()
            if user_input.strip().lower() == 'q':
                print("\n[停止シグナル検知] 現在のイテレーションが終了次第、モデルを保存して安全に終了します。お待ちください...")
                stop_event.set()
                break
        except Exception:
            break

# training settings
task_name = 'point' # wp, point
log_path = f'log_rl_{task_name}.txt'
rl_folder = f'model_rl_{task_name}'

# continue training with trained models
start_loop_num = 1
saved_model_path = {'discard':f'{rl_folder}/discard_final_stop.pt', 
                    'pick':f'{rl_folder}/pick_final_stop.pt',
                    'koikoi':f'{rl_folder}/koikoi_final_stop.pt'}

assert task_name in ['point', 'wp']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
win_prob_mat = None

TraceSlot = namedtuple(
    'TraceSlot', ['key','state','action'])

Transition = namedtuple(
    'Transition', ['state','action','reward'])

def time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

def print_log(log_str, log_path):
    with open(log_path, 'a') as f:
        print(log_str)
        print(log_str, file=f)
    return


class Buffer():
    def __init__(self):
        #self.memory = {'discard':[], 'pick':[], 'koikoi':[]}
        # ★ SoA (Struct of Arrays) 形式でデータを保持するように変更
        self.memory = {'discard': {'states': [], 'actions': [], 'rewards': []}, 
                       'pick': {'states': [], 'actions': [], 'rewards': []}, 
                       'koikoi': {'states': [], 'actions': [], 'rewards': []}}
        self.sizes = {'discard': 0, 'pick': 0, 'koikoi': 0}
    
    def extend(self, data_dict):
        #for key in data_dict.keys():
        #    self.memory[key].extend(data_dict[key])
        for key in ['discard', 'pick', 'koikoi']:
            if data_dict.get(key) is not None:
                self.memory[key]['states'].append(data_dict[key]['states'])
                self.memory[key]['actions'].append(data_dict[key]['actions'])
                self.memory[key]['rewards'].append(data_dict[key]['rewards'])
                self.sizes[key] += len(data_dict[key]['rewards'])
        return
    
    #def get_batch(self, key, batch_size):
    #    n_batch = len(self.memory[key]) // batch_size
    #    ind_list = [ii for ii in range(n_batch)]
    #    random.shuffle(ind_list)
    #    for ii in ind_list:
    #        yield self.memory[key][ii:n_batch * batch_size:n_batch]
    
    def clear(self):
        self.__init__()
        return


class TraceSimulator():
    def __init__(self, agent, global_win_prob_mat,
                 record_state=['discard','discard-pick','draw-pick','koikoi'], 
                 discount=1):
        self.agent = {1:agent, 2:agent}
        self.record_state = record_state
        self.discount = discount
        self.win_prob_mat = global_win_prob_mat
        self.buffer = {'discard':[], 'pick':[], 'koikoi':[]}
    
    def __reward_wp(self, player):
        round_num = self.game_state.round + 1
        point = self.game_state.point[player] + self.game_state.round_state.round_point[player]
        is_dealer = int(self.game_state.round_state.winner == player)
        if round_num <= 8 and (0 < point < 60):
            win_prob = self.win_prob_mat[is_dealer, round_num, point]  
        else:
            win_prob = 0.5 if point == 30 else float(point > 30)        
        return win_prob * 10.0
    
    def __reward_point(self, player):
        round_point = self.game_state.round_state.round_point[player]
        return float(round_point)
    
    def random_make_games(self, n_games):
        self.t_action = 0.0
        self.t_step = 0.0
        self.t_clone = 0.0
        self.t_other = 0.0
        
        # バッファを最初からSoA（リスト）形式で保持
        self.buffer = {
            'discard': {'states': [], 'actions': [], 'rewards': []}, 
            'pick': {'states': [], 'actions': [], 'rewards': []}, 
            'koikoi': {'states': [], 'actions': [], 'rewards': []}
        }
        
        self.agent[1].t_feat_gen = 0.0
        self.agent[1].t_cpu_to_gpu = 0.0
        self.agent[1].t_forward = 0.0
        self.agent[1].t_post_proc = 0.0

        local_envs_count = min(128, n_games)
        
        envs = [koikoigame.KoiKoiGameState() for _ in range(local_envs_count)]
        env_traces = [{1: [], 2: []} for _ in range(local_envs_count)]
        
        env_last_features = [{
            'discard': None, 
            'discard-pick': None, 
            'draw-pick': None, 
            'koikoi': None
        } for _ in range(local_envs_count)]
        
        finished_games = 0
        type_dict = {'discard': 'discard', 'discard-pick': 'pick', 'draw-pick': 'pick', 'koikoi': 'koikoi'}
        
        def action_to_index(action):
            if action in [False, True]: 
                return int(action)
            elif action is not None: 
                return 4 * (action[0] - 1) + (action[1] - 1)
            return None

        # ★ PyTorchを排除し、NumPyネイティブの配列スライスに変更
        def adjust_card_order_np(feature, index):
            ind_list = [index] + [ii for ii in range(feature.shape[1]) if ii != index]
            return feature[:, ind_list]

        while finished_games < n_games:
            requests = {'discard': [], 'discard-pick': [], 'draw-pick': [], 'koikoi': []}
            
            t_scan = time.perf_counter()
            for i, game_state in enumerate(envs):
                if game_state.game_over:
                    continue
                    
                if not game_state.round_state.wait_action:
                    t_s = time.perf_counter()
                    game_state.round_state.step(None)
                    self.t_step += (time.perf_counter() - t_s)
                    continue
                
                state = game_state.round_state.state
                player = game_state.round_state.turn_player
                mask = game_state.round_state.action_mask
                
                # ★ PyTorchを介さず、NumPy配列をそのままコピーしてキャッシュする
                feat_np = game_state.feature_tensor.numpy()
                env_last_features[i][state] = feat_np.copy()
                
                # PyTorchへの変換はバッチ推論の入り口だけで行う
                requests[state].append((i, player, feat_np, mask))
            self.t_other += (time.perf_counter() - t_scan)

            for state_name, req_list in requests.items():
                if len(req_list) == 0:
                    continue
                
                t_stack = time.perf_counter()
                # バッチ推論のためにここで初めてTensor化
                batched_features_cpu = torch.from_numpy(np.stack([r[2] for r in req_list]))
                masks = [r[3] for r in req_list]
                self.t_other += (time.perf_counter() - t_stack)
                
                t_action_start = time.perf_counter()
                actions = self.agent[1].predict_batch(state_name, batched_features_cpu, masks)
                self.t_action += (time.perf_counter() - t_action_start)
                
                for idx, (env_idx, player, _, _) in enumerate(req_list):
                    action = actions[idx]
                    
                    if player in [1, 2] and (state_name in self.record_state) and (action is not None):
                        t_c = time.perf_counter()
                        # NumPy配列のまま履歴に保存
                        feat = env_last_features[env_idx][state_name]
                        self.t_clone += (time.perf_counter() - t_c)
                        
                        env_traces[env_idx][player].append(TraceSlot(
                            key = type_dict[state_name], 
                            state = feat, 
                            action = action_to_index(action)
                        ))
                    
                    t_s = time.perf_counter()
                    envs[env_idx].round_state.step(action)
                    self.t_step += (time.perf_counter() - t_s)

            for i in range(len(envs)):
                if envs[i].round_state.round_over and not envs[i].game_over:
                    t_reward = time.perf_counter()
                    if task_name == 'wp':
                        reward = {}
                        for p in [1, 2]:
                            round_num = envs[i].round + 1
                            point = envs[i].point[p] + envs[i].round_state.round_point[p]
                            is_dealer = int(envs[i].round_state.winner == p)
                            if round_num <= 8 and (0 < point < 60):
                                win_prob = self.win_prob_mat[is_dealer, round_num, point]  
                            else:
                                win_prob = 0.5 if point == 30 else float(point > 30)        
                            reward[p] = win_prob * 10.0
                    elif task_name == 'point':
                        reward = {1: float(envs[i].round_state.round_point[1]), 2: float(envs[i].round_state.round_point[2])}
                    
                    for player in [1, 2]:
                        for rev_step in range(len(env_traces[i][player])):
                            slot = env_traces[i][player][-rev_step-1]
                            # ★ PyTorch変換を排除し、NumPyのまま処理（Transition生成も廃止）
                            state_np = adjust_card_order_np(slot.state, slot.action).astype(np.float16)
                            self.buffer[slot.key]['states'].append(state_np)
                            self.buffer[slot.key]['actions'].append(slot.action)
                            self.buffer[slot.key]['rewards'].append(reward[player] * (self.discount ** rev_step))
                            
                    self.t_other += (time.perf_counter() - t_reward)
                    envs[i].new_round()
                    
                    env_traces[i] = {1: [], 2: []}
                    env_last_features[i] = {'discard': None, 'discard-pick': None, 'draw-pick': None, 'koikoi': None}
                    
                if envs[i].game_over:
                    t_reward = time.perf_counter()
                    if task_name == 'wp':
                        reward = {}
                        for p in [1, 2]:
                            round_num = envs[i].round + 1
                            point = envs[i].point[p] + envs[i].round_state.round_point[p]
                            is_dealer = int(envs[i].round_state.winner == p)
                            if round_num <= 8 and (0 < point < 60):
                                win_prob = self.win_prob_mat[is_dealer, round_num, point]  
                            else:
                                win_prob = 0.5 if point == 30 else float(point > 30)        
                            reward[p] = win_prob * 10.0
                    elif task_name == 'point':
                        reward = {1: float(envs[i].round_state.round_point[1]), 2: float(envs[i].round_state.round_point[2])}
                    
                    for player in [1, 2]:
                        for rev_step in range(len(env_traces[i][player])):
                            slot = env_traces[i][player][-rev_step-1]
                            # ★ PyTorch変換を排除し、NumPyのまま処理
                            state_np = adjust_card_order_np(slot.state, slot.action).astype(np.float16)
                            self.buffer[slot.key]['states'].append(state_np)
                            self.buffer[slot.key]['actions'].append(slot.action)
                            self.buffer[slot.key]['rewards'].append(reward[player] * (self.discount ** rev_step))
                            
                    self.t_other += (time.perf_counter() - t_reward)
                    finished_games += 1
                    
                    if finished_games < n_games:
                        envs[i] = koikoigame.KoiKoiGameState()
                        env_traces[i] = {1: [], 2: []}
                        env_last_features[i] = {'discard': None, 'discard-pick': None, 'draw-pick': None, 'koikoi': None}

        t_feat_gen = self.agent[1].t_feat_gen
        t_forward = self.agent[1].t_forward
        t_post_proc = self.agent[1].t_post_proc
        t_cpu_to_gpu = self.agent[1].t_cpu_to_gpu
        
        self.buffer['time_stats'] = [(
            self.t_action, 
            self.t_step, 
            self.t_clone, 
            self.t_other, 
            t_feat_gen, 
            t_forward, 
            t_post_proc, 
            t_cpu_to_gpu
        )]
        
        # ★ AoSからSoAへ変換（巨大なNumPy配列化してシリアライズを超高速化）
        clean_buffer = {'discard': None, 'pick': None, 'koikoi': None, 'time_stats': self.buffer['time_stats']}
        for key in ['discard', 'pick', 'koikoi']:
            if len(self.buffer[key]['rewards']) > 0:
                # 既にリスト形式になっているので、最後に一発で配列化
                states = np.array(self.buffer[key]['states'])
                actions = np.array(self.buffer[key]['actions'], dtype=np.int32)
                rewards = np.array(self.buffer[key]['rewards'], dtype=np.float32)
                clean_buffer[key] = {'states': states, 'actions': actions, 'rewards': rewards}
        return clean_buffer
    
    def make_game_trace(self):
        def action_to_index(action):
            if action in [False, True]:
                index = int(action)
            elif action != None:
                index = 4*(action[0]-1) + (action[1]-1)
            else:
                index = None
            return index
        
        def adjust_card_order(feature, index):
            ind_list = [index] + [ii for ii in range(feature.size(1)) if ii!=index]
            return feature[:,ind_list]
        
        type_dict = {'discard':'discard', 'discard-pick':'pick', 'draw-pick':'pick', 'koikoi':'koikoi'}
        
        t_start = time.perf_counter()
        self.game_state = koikoigame.KoiKoiGameState()
        self.t_other += (time.perf_counter() - t_start)
        
        while True:
            if self.game_state.game_over == True:
                break
            # play a round
            trace = {1:[], 2:[]}
            while not self.game_state.round_state.round_over:
                player = self.game_state.round_state.turn_player
                state = self.game_state.round_state.state
                
                t0 = time.perf_counter()
                action = self.agent[player].auto_action(self.game_state, use_mask=True)            
                self.t_action += (time.perf_counter() - t0)
                
                if player in [1,2] and (state in self.record_state) and (action is not None):
                    t_feat_start = time.perf_counter()
                    raw_feat = self.agent[player].last_feature[state]
                    t_feat_end = time.perf_counter()
                    
                    t_clone_start = time.perf_counter()
                    
                    if raw_feat is None:
                        feat = self.game_state.feature_tensor.clone()
                    else:
                        feat = raw_feat.clone()
                        
                    t_clone_end = time.perf_counter()
                    
                    self.t_other += (t_feat_end - t_feat_start)
                    self.t_clone += (t_clone_end - t_clone_start)
                    
                    trace[player].append(TraceSlot(
                        key = type_dict[state],
                        state = feat, 
                        action = action_to_index(action)))
                
                t2 = time.perf_counter()
                self.game_state.round_state.step(action) 
                self.t_step += (time.perf_counter() - t2)
            
            t_rec = time.perf_counter()
            if task_name == 'wp':
                reward = {1:self.__reward_wp(1), 2:self.__reward_wp(2)}
            elif task_name == 'point':
                reward = {1:self.__reward_point(1), 2:self.__reward_point(2)}
            
            for player in [1,2]:
                for rev_step in range(len(trace[player])):
                    key = trace[player][-rev_step-1].key
                    action = trace[player][-rev_step-1].action
                    state_tensor = adjust_card_order(trace[player][-rev_step-1].state.clone(), action).half()
                    self.buffer[key].append(Transition(
                        state = state_tensor.numpy(), 
                        action = action, 
                        reward = reward[player] * (self.discount ** rev_step)))
            self.game_state.new_round()
            self.t_other += (time.perf_counter() - t_rec)
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


def get_value_action_net(action_net_path, value_net):
    map_location = torch.device('cpu')
    action_net = torch.load(action_net_path, map_location, weights_only=False)
    
    for module in action_net.modules():
        if type(module).__name__ == 'MultiheadAttention':
            if not hasattr(module, 'batch_first'):
                module.batch_first = False
        elif type(module).__name__ == 'TransformerEncoderLayer':
            if not hasattr(module, 'norm_first'):
                module.norm_first = False
                
    value_net.load_state_dict(action_net.state_dict())
    return value_net, action_net


def parallel_sampling(agent, n_games, global_win_prob_mat):
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        for key in agent.model.keys():
            agent.model[key] = agent.model[key].to(device)
            agent.model[key].eval()
        
        trace_simulator = TraceSimulator(agent, global_win_prob_mat)
        trace_simulator.agent[1].model = agent.model
        trace_simulator.agent[2].model = agent.model
        
        sample_dict = trace_simulator.random_make_games(n_games)
        return sample_dict

    except Exception as e:
        raise e

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_worker():
    
    # 乱数シードの再設定
    seed = (os.getpid() * int(time.time() * 1000)) % 123456789
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    global master_agent
    master_discard_net, master_pick_net, master_koikoi_net = get_master_net()
    
    master_discard_net = master_discard_net.to(device)
    master_pick_net = master_pick_net.to(device)
    master_koikoi_net = master_koikoi_net.to(device)
    
    master_discard_net.eval()
    master_pick_net.eval()
    master_koikoi_net.eval()
    
    master_agent = koikoilearn.Agent(master_discard_net, master_pick_net, master_koikoi_net)

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
    print_log(f'Record,{loop},{score},{point}', log_path)
    return score


def random_action_prob_scheduler(score):
    if score < 0.10:
        p = [0.25] * 4
    elif score < 0.20:
        p = [0.20] * 4
    elif score < 0.30:
        p = [0.15] * 4
    elif score < 0.40:
        p = [0.125] * 4
    elif score < 0.50:
        p = [0.10] * 4
    elif score < 0.55:
        p = [0.075] * 4
    else:
        p = [0.05] * 4
    return p


if __name__ == '__main__':
    if not os.path.isdir(rl_folder):
        os.mkdir(rl_folder)
        
    exit_thread = threading.Thread(target=wait_for_exit_key, daemon=True)
    exit_thread.start()
        
    with open('win_prob_mat.pkl','rb') as f:
        win_prob_mat = pickle.load(f)

    criterion = torch.nn.SmoothL1Loss(beta=30.0).to(device)
    master_discard_net, master_pick_net, master_koikoi_net = get_master_net()
    master_agent = koikoilearn.Agent(master_discard_net, master_pick_net, master_koikoi_net)
    
    cpu_count = 8
    loop_games = 480
    n_core_games = loop_games // cpu_count
    
    batch_size = 256
    
    n_loop_action_net_update = 5
    n_loop_arena_test = 10
    
    buffer = Buffer()
    
    value_net, action_net = {}, {}
    value_net['discard'] = TargetQNet().cpu()
    value_net['pick'] = TargetQNet().cpu()
    value_net['koikoi'] = TargetQNet().cpu()
    
    action_net['discard'] = DiscardModel().cpu()
    action_net['pick'] = PickModel().cpu()
    action_net['koikoi'] = KoiKoiModel().cpu()
    for key in ['discard', 'pick', 'koikoi']:
        torch.save(action_net[key], f'{rl_folder}/{key}_0_0.pt')
    
    value_net['discard'], raw_discard = get_value_action_net(saved_model_path['discard'], value_net['discard'])
    value_net['pick'], raw_pick = get_value_action_net(saved_model_path['pick'], value_net['pick'])
    value_net['koikoi'], raw_koikoi = get_value_action_net(saved_model_path['koikoi'], value_net['koikoi'])    

    example_input_normal = torch.zeros((1, 300, 48), dtype=torch.float32)
    example_input_koikoi = torch.zeros((1, 300, 50), dtype=torch.float32)
    
    with torch.inference_mode():
        traced_discard = torch.jit.trace(raw_discard.eval(), example_input_normal)
        traced_pick    = torch.jit.trace(raw_pick.eval(), example_input_normal)
        traced_koikoi  = torch.jit.trace(raw_koikoi.eval(), example_input_koikoi)

    action_net = {
        'discard': raw_discard.eval(),
        'pick': raw_pick.eval(),
        'draw-pick': traced_pick,
        'koikoi': raw_koikoi.eval()
    }
    
    for key in ['discard', 'pick', 'koikoi']:
        value_net[key].to(device)
            
    play_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'],
                                   random_action_prob=[0.1, 0.1, 0.1, 0.1])    
    
    optimizer = {'discard': torch.optim.Adam(value_net['discard'].parameters(), lr=0.0001),
                 'pick': torch.optim.Adam(value_net['pick'].parameters(), lr=0.0001),
                 'koikoi': torch.optim.Adam(value_net['koikoi'].parameters(), lr=0.0001)}
    scaler = torch.amp.GradScaler('cuda')

    score = [0.0]
    print_log(f'\n{time_str()} start training on device: {device}', log_path)
    
    pool = multiprocessing.Pool(cpu_count, initializer=init_worker)
    
    for loop in range(start_loop_num, 100000):
        loop_start_time = time.perf_counter()
        
        async_results = []
        for _ in range(cpu_count):
            res = pool.apply_async(parallel_sampling, 
                                   args=(play_agent, n_core_games, win_prob_mat))
            async_results.append(res)
        
        total_action = 0.0
        total_step = 0.0
        total_clone = 0.0
        total_other = 0.0
        total_feat_gen = 0.0
        total_forward = 0.0
        total_post_proc = 0.0
        total_cpu_to_gpu = 0.0
        
        buffer = Buffer() 
        
        for res in async_results:
            res_dict = res.get()
            if 'time_stats' in res_dict:
                stats = res_dict.pop('time_stats')[0]
                total_action += stats[0]
                total_step += stats[1]
                total_clone += stats[2]
                total_other += stats[3]
                total_feat_gen += stats[4]
                total_forward += stats[5]
                total_post_proc += stats[6]
                total_cpu_to_gpu += stats[7]
            
            buffer.extend(res_dict)
            
        n_sample = [buffer.sizes[key] for key in ['discard', 'pick', 'koikoi']]
        elapsed_time = time.perf_counter() - loop_start_time
        
        print_log(f'{time_str()} {loop} loops, {tuple(n_sample)} samples generated. (Total Loop Time: {elapsed_time:.2f}s)', log_path)
        
        sum_w_time = total_action + total_step + total_clone + total_other
        if sum_w_time > 0:
            print(f"   [Profile] NN 推論 (auto_action) : {total_action:.2f}s ({total_action/sum_w_time*100:.1f}%)")
            if total_action > 0:
                print(f"      └─ 特徴量変換 (feat_tensor) : {total_feat_gen:.2f}s ({total_feat_gen/total_action*100:.1f}% of action)")
                print(f"      └─ ★CPU->GPU転送 (CUDA.to)  : {total_cpu_to_gpu:.2f}s ({total_cpu_to_gpu/total_action*100:.1f}% of action)") # 新設
                print(f"      └─ 純粋な推論 (model.forward): {total_forward:.2f}s ({total_forward/total_action*100:.1f}% of action)")
                print(f"      └─ 後処理 (numpy/exp/argmax) : {total_post_proc:.2f}s ({total_post_proc/total_action*100:.1f}% of action)")
            print(f"   [Profile] ゲーム更新 (step)     : {total_step:.2f}s ({total_step/sum_w_time*100:.1f}%)")
            print(f"   [Profile] 純粋なメモリ複製 (clone): {total_clone:.2f}s ({total_clone/sum_w_time*100:.1f}%)") # ラベル変更
            print(f"   [Profile] その他 (特徴量生成含む) : {total_other:.2f}s ({total_other/sum_w_time*100:.1f}%)") # ラベル変更
            print(f"   [IPC Overhead] 通信・シリアライズ推定 : {max(0.0, elapsed_time - (sum_w_time/cpu_count)):.2f}s")
        
        for key in ['discard', 'pick', 'koikoi']:
            value_net[key].train()
            train_loss = []
            
            if len(buffer.memory[key]) == 0:
                continue
                
            #all_states = np.stack([t.state for t in buffer.memory[key]])
            #all_rewards = np.array([t.reward for t in buffer.memory[key]], dtype=np.float32)
            
            all_states = np.concatenate(buffer.memory[key]['states'], axis=0)
            all_rewards = np.concatenate(buffer.memory[key]['rewards'], axis=0)
            
            dataset = TensorDataset(torch.from_numpy(all_states), torch.from_numpy(all_rewards))
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                pin_memory=True,
                num_workers=0
            )
            
            for step, (state_batch, reward_batch) in enumerate(data_loader):
                state_batch = state_batch.to(device, non_blocking=True).float()
                reward_batch = reward_batch.to(device, non_blocking=True)
                
                optimizer[key].zero_grad()
                
                with torch.amp.autocast(device_type='cuda'):
                    q_values = value_net[key](state_batch).squeeze(1)
                    loss = criterion(q_values, reward_batch)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer[key])
                scaler.update()
                
                train_loss.append(loss.cpu().data.item())
            print_log(f'{time_str()} {key} net, {step+1} steps, loss = {np.mean(train_loss)}', log_path)
        
        buffer = Buffer()
        
        if loop % n_loop_action_net_update == 0:
            type_dict = {'discard':'discard', 'discard-pick':'pick', 'draw-pick':'pick', 'koikoi':'koikoi'}
            
            for model_key, net_key in type_dict.items():
                raw_net = getattr(value_net[net_key], '_orig_mod', value_net[net_key])
                action_net[net_key].load_state_dict({k: v.cpu() for k, v in raw_net.state_dict().items()})
                action_net[net_key].eval()
                
            play_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'],
                                           random_action_prob=random_action_prob_scheduler(score[-1]))
            test_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'])
        
        if loop % n_loop_arena_test == 0:
            result = []
            async_results_test = []
            
            for _ in range(cpu_count):
                res = pool.apply_async(parallel_arena_test, 
                                       args=(test_agent, 400//cpu_count))
                async_results_test.append(res)
                
            for res in async_results_test:
                result.append(res.get())
                
            s = test_result_analysis(result,loop)
            score.append(s)
            if s == max(score[-20:]) or (loop%50==0):
                for key in ['discard', 'pick', 'koikoi']:
                    path = f'{rl_folder}/{key}_{loop}_{round(s*100)}.pt'
                    torch.save(action_net[key], path)
                with open(f'{rl_folder}/optimizer.pickle','wb') as f:
                    pickle.dump(optimizer, f)
                print_log(f'{time_str()}   New model saved.', log_path)
            play_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'],
                                           random_action_prob=random_action_prob_scheduler(score[-1]))
            
        if stop_event.is_set():
            print_log(f'\n{time_str()} [安全停止] 最新のモデルを保存します。', log_path)
            for key in ['discard', 'pick', 'koikoi']:
                final_path = f'{rl_folder}/{key}_final_stop.pt'
                torch.save(action_net[key], final_path)
            print_log(f'{time_str()} 全てのモデルの安全保存が完了しました。プログラムを終了します。', log_path)
            break