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

import koikoigame
import koikoilearn
from koikoinet2L import DiscardModel, PickModel, KoiKoiModel, TargetQNet

# 終了を検知するためのシグナルオブジェクト
stop_event = threading.Event()

def wait_for_exit_key():
    """裏でキー入力を監視するサブスレッド用の関数"""
    print("\n[システム] 画面上で 'q' キーを押して Enter を入力すると、現在のループ終了時に安全に停止します。\n")
    while True:
        try:
            # 入力を待つ（Enterが必要ですが、標準機能だけで速度低下なく実現できます）
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
start_loop_num = 227
saved_model_path = {'discard':f'{rl_folder}/discard_final_stop.pt', 
                    'pick':f'{rl_folder}/pick_final_stop.pt',
                    'koikoi':f'{rl_folder}/koikoi_final_stop.pt'}

assert task_name in ['point', 'wp']

# GPUが利用可能ならcuda、なければcpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 【メモリ対策】子プロセスで二重に読み込まれないよう、グローバルでの読み込みを廃止（__main__内で読み込む）
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
        self.memory = {'discard':[], 'pick':[], 'koikoi':[]}
    
    def extend(self, data_dict):
        for key in data_dict.keys():
            self.memory[key].extend(data_dict[key])
        return
    
    def get_batch(self, key, batch_size):
        n_batch = len(self.memory[key]) // batch_size
        ind_list = [ii for ii in range(n_batch)]
        random.shuffle(ind_list)
        for ii in ind_list:
            yield self.memory[key][ii:n_batch * batch_size:n_batch]
    
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
        self.buffer = {'discard':[], 'pick':[], 'koikoi':[]}
        for _ in range(n_games):
            self.make_game_trace()
        return self.buffer
    
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
        
        self.game_state = koikoigame.KoiKoiGameState()
        
        while True:
            if self.game_state.game_over == True:
                break
            # play a round
            trace = {1:[], 2:[]}
            while not self.game_state.round_state.round_over:
                player = self.game_state.round_state.turn_player
                state = self.game_state.round_state.state
                action = self.agent[player].auto_action(self.game_state, use_mask=True)            
                if player in [1,2] and (state in self.record_state) and (action is not None):
                    trace[player].append(TraceSlot(
                        key = type_dict[state],
                        state = self.game_state.feature_tensor.clone(), 
                        action = action_to_index(action)))
                self.game_state.round_state.step(action) 
            
            # record
            if task_name == 'wp':
                reward = {1:self.__reward_wp(1), 2:self.__reward_wp(2)}
            elif task_name == 'point':
                reward = {1:self.__reward_point(1), 2:self.__reward_point(2)}
            
            for player in [1,2]:
                for rev_step in range(len(trace[player])):
                    key = trace[player][-rev_step-1].key
                    action = trace[player][-rev_step-1].action
                    self.buffer[key].append(Transition(
                        state = adjust_card_order(trace[player][-rev_step-1].state.clone(), action), 
                        action = action, 
                        reward = reward[player] * (self.discount ** rev_step)))
            # next round
            self.game_state.new_round()
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
    trace_simulator = TraceSimulator(agent, global_win_prob_mat)
    sample_dict = trace_simulator.random_make_games(n_games)
    return sample_dict


def parallel_arena_test(agent, n_games, master_agent):
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


# Monte-Carlo learning with self-play
if __name__ == '__main__':
    if not os.path.isdir(rl_folder):
        os.mkdir(rl_folder)
        
    exit_thread = threading.Thread(target=wait_for_exit_key, daemon=True)
    exit_thread.start()
        
    # 【メモリ対策】win_prob_mat や master_agent の初期化を __main__ 内部に移動
    with open('win_prob_mat.pkl','rb') as f:
        win_prob_mat = pickle.load(f)

    criterion = torch.nn.SmoothL1Loss(beta=30.0).to(device)
    master_discard_net, master_pick_net, master_koikoi_net = get_master_net()
    master_agent = koikoilearn.Agent(master_discard_net, master_pick_net, master_koikoi_net)
    
    # 【メモリ対策】同時に立ち上げるプロセス数を「48」から「4」などの安全な値に制限
    # ※マシンの物理コア数に合わせて 4〜8 程度にしておくのがメモリ・CPU的に最も安全です。
    cpu_count = 4
    loop_games = 480
    n_core_games = loop_games // cpu_count
    
    batch_size = 256
    
    n_loop_action_net_update = 5
    n_loop_arena_test = 5
    
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
    
    value_net['discard'], action_net['discard'] = get_value_action_net(
        saved_model_path['discard'], value_net['discard'])
    value_net['pick'], action_net['pick'] = get_value_action_net(
        saved_model_path['pick'], value_net['pick'])
    value_net['koikoi'], action_net['koikoi'] = get_value_action_net(
        saved_model_path['koikoi'], value_net['koikoi'])    
    
    for key in ['discard', 'pick', 'koikoi']:
        value_net[key].to(device)
            
    play_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'],
                                   random_action_prob=[0.1, 0.1, 0.1, 0.1])    
    
    optimizer = {'discard': torch.optim.Adam(value_net['discard'].parameters(), lr=0.0001),
                 'pick': torch.optim.Adam(value_net['pick'].parameters(), lr=0.0001),
                 'koikoi': torch.optim.Adam(value_net['koikoi'].parameters(), lr=0.0001)}
    scaler = scaler = torch.amp.GradScaler(device_type='cuda')

    score = [0.0]
    print_log(f'\n{time_str()} start training on device: {device}', log_path)
    for loop in range(start_loop_num, 100000):
        pool = multiprocessing.Pool(cpu_count)
        for _ in range(cpu_count):
            pool.apply_async(parallel_sampling, 
                             args=(play_agent, n_core_games, win_prob_mat), 
                             callback=buffer.extend)
        pool.close()
        pool.join()

        n_sample = [len(buffer.memory[key]) for key in ['discard', 'pick', 'koikoi']]
        print_log(f'{time_str()} {loop} loops, {tuple(n_sample)} samples generated.', log_path)
        
        # optimize value net
        for key in ['discard', 'pick', 'koikoi']:
            value_net[key].train()
            train_loss = []
            for step, transitions in enumerate(buffer.get_batch(key, batch_size)):
                transitions = Transition(*zip(*transitions))
                
                state_batch = torch.stack(transitions.state).to(device, non_blocking=True)
                reward_batch = torch.Tensor(transitions.reward).to(device, non_blocking=True) 
                
                # 勾配の初期化
                optimizer[key].zero_grad()
                
                # --- 追加：自動混合精度（AMP）コンテキスト ---
                with torch.amp.autocast(device_type='cuda'):
                    q_values = value_net[key](state_batch).squeeze(1)
                    loss = criterion(q_values, reward_batch)
                
                # --- 変更：スケーラーを経由した逆伝播とパラメータ更新 ---
                scaler.scale(loss).backward()
                scaler.step(optimizer[key])
                scaler.update()
                
                train_loss.append(loss.cpu().data.item())
            print_log(f'{time_str()} {key} net, {step+1} steps, loss = {np.mean(train_loss)}', log_path)
        
        del transitions, state_batch, reward_batch, q_values, loss
        buffer.clear()
        
        # update action net and agent
        if loop % n_loop_action_net_update == 0:
            type_dict = {'discard':'discard', 'discard-pick':'pick', 'draw-pick':'pick', 'koikoi':'koikoi'}
            for model_key, net_key in type_dict.items():
                #action_net[net_key].load_state_dict(value_net[net_key].to('cpu').state_dict())
                #value_net[net_key].to(device)
                raw_net = getattr(value_net[net_key], '_orig_mod', value_net[net_key])
                cpu_state_dict = {k: v.cpu() for k, v in raw_net.state_dict().items()}
                action_net[net_key].load_state_dict(cpu_state_dict)
                
            play_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'],
                                           random_action_prob=random_action_prob_scheduler(score[-1]))
            test_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'])  
        
        # arena test
        if loop % n_loop_arena_test == 0:
            result = []
            pool = multiprocessing.Pool(cpu_count)
            for _ in range(cpu_count):
                pool.apply_async(parallel_arena_test, 
                                 args=(test_agent, 400//cpu_count, master_agent), 
                                 callback=result.append)
            pool.close()
            pool.join()
            s = test_result_analysis(result,loop)
            score.append(s)
            if s == max(score[-20:]) or (loop%50==0):
                for key in ['discard', 'pick', 'koikoi']:
                    path = f'{rl_folder}/{key}_{loop}_{round(s*100)}.pt'
                    torch.save(action_net[key], path)
                with open(f'{rl_folder}/optimizer.pickle','wb') as f:
                    pickle.dump(optimizer, f)
                print_log(f'{time_str()}  New model saved.', log_path)
            play_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'],
                                           random_action_prob=random_action_prob_scheduler(score[-1]))
            
        if stop_event.is_set():
            print_log(f'\n{time_str()} [安全停止] 最新のモデルを保存します。', log_path)
            # 終了直前に強制的に最新モデルを確定保存する
            for key in ['discard', 'pick', 'koikoi']:
                final_path = f'{rl_folder}/{key}_final_stop.pt'
                torch.save(action_net[key], final_path)
            print_log(f'{time_str()} 全てのモデルの安全保存が完了しました。プログラムを終了します。', log_path)
            break # 学習ループを脱出して終了