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
start_loop_num = 517
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
        
        self.t_action = 0.0
        self.t_step = 0.0
        self.t_clone = 0.0
        self.t_other = 0.0
        
        # エージェント側の内訳タイマーをリセット
        for p in [1, 2]:
            self.agent[p].t_feat_gen = 0.0
            self.agent[p].t_forward = 0.0
            self.agent[p].t_post_proc = 0.0
        
        for _ in range(n_games):
            self.make_game_trace()
            
        # プレイヤー1のエージェントから内訳タイマーを取得（自己対局で共通のインスタンス）
        t_feat_gen = self.agent[1].t_feat_gen
        t_forward = self.agent[1].t_forward
        t_post_proc = self.agent[1].t_post_proc
            
        # 拡張版 time_stats を送信 (従来の4つに加えて、内訳3つを追加)
        self.buffer['time_stats'] = [(self.t_action, self.t_step, self.t_clone, self.t_other, t_feat_gen, t_forward, t_post_proc)]
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
                
                # ① NN推論 (auto_action) の時間を計測
                t0 = time.perf_counter()
                action = self.agent[player].auto_action(self.game_state, use_mask=True)            
                self.t_action += (time.perf_counter() - t0)
                
                if player in [1,2] and (state in self.record_state) and (action is not None):
                    t_feat_start = time.perf_counter()
                    # 1. まずキャッシュから特徴量を取り出す
                    raw_feat = self.agent[player].last_feature[state]
                    t_feat_end = time.perf_counter()
                    
                    t_clone_start = time.perf_counter()
                    
                    # ★【追加・安全弁】もし自動フェーズなどでキャッシュが空(None)だった場合は、
                    # 例外的にその場で安全に新規生成させる（フォールバック）
                    if raw_feat is None:
                        feat = self.game_state.feature_tensor.clone()
                    else:
                        feat = raw_feat.clone() # キャッシュがあればそれを利用
                        
                    t_clone_end = time.perf_counter()
                    
                    self.t_other += (t_feat_end - t_feat_start)
                    self.t_clone += (t_clone_end - t_clone_start)
                    
                    trace[player].append(TraceSlot(
                        key = type_dict[state],
                        state = feat, 
                        action = action_to_index(action)))
                
                # ③ ゲーム状態の更新 (step) の時間を計測
                t2 = time.perf_counter()
                self.game_state.round_state.step(action) 
                self.t_step += (time.perf_counter() - t2)
            
            # record
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
            # next round
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
    # 【追加】子プロセス側のメモリ空間上で、安全かつ確実に TorchScript トレース（高速化）をかける
    example_input = torch.zeros((1, 300, 48), dtype=torch.float32)
    
    with torch.inference_mode():
        # 送信されてきた生のモデルから、最速の JITコンパイル版 を作成して差し替える
        traced_discard = torch.jit.trace(agent.model['discard'], example_input)
        traced_pick    = torch.jit.trace(agent.model['discard-pick'], example_input)
        traced_koikoi  = torch.jit.trace(agent.model['koikoi'], example_input)
        
        # エージェントが内部で参照しているモデルマップを高速版に上書き
        agent.model = {
            'discard': traced_discard,
            'discard-pick': traced_pick,
            'draw-pick': traced_pick,
            'koikoi': traced_koikoi
        }
    
    # あとは完全に最適化されたランタイムでゲームをシミュレーション
    trace_simulator = TraceSimulator(agent, global_win_prob_mat)
    sample_dict = trace_simulator.random_make_games(n_games)
    return sample_dict

def init_worker():
    global master_agent
    master_discard_net, master_pick_net, master_koikoi_net = get_master_net()
    master_agent = koikoilearn.Agent(master_discard_net, master_pick_net, master_koikoi_net)

def parallel_arena_test(agent, n_games):
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

    # 【変更】親プロセス側での torch.jit.trace は行わず、生の eval() モデルをセットする
    # これにより、multiprocessing の引数として正常に送信（シリアライズ）できるようになります
    action_net = {
        'discard': raw_discard.eval(),
        'pick': raw_pick.eval(),
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
        # ループ開始時の高精度タイムスタンプを取得
        loop_start_time = time.perf_counter()
        
        async_results = []
        for _ in range(cpu_count):
            res = pool.apply_async(parallel_sampling, 
                                   args=(play_agent, n_core_games, win_prob_mat))
            async_results.append(res)
        
        # 各子プロセスのタイマーを集計するための変数を初期化
        total_action = 0.0
        total_step = 0.0
        total_clone = 0.0
        total_other = 0.0
        total_feat_gen = 0.0
        total_forward = 0.0
        total_post_proc = 0.0
        
        # get()で全ワーカーの終了を待機しつつ、時間統計を回収
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
            buffer.extend(res_dict)
            
        n_sample = [len(buffer.memory[key]) for key in ['discard', 'pick', 'koikoi']]
        elapsed_time = time.perf_counter() - loop_start_time
        
        # ログ出力
        print_log(f'{time_str()} {loop} loops, {tuple(n_sample)} samples generated. (Total Loop Time: {elapsed_time:.2f}s)', log_path)
        
        sum_w_time = total_action + total_step + total_clone + total_other
        if sum_w_time > 0:
            print(f"   [Profile] NN 推論 (auto_action) : {total_action:.2f}s ({total_action/sum_w_time*100:.1f}%)")
            if total_action > 0:
                print(f"      └─ 特徴量変換 (feat_tensor) : {total_feat_gen:.2f}s ({total_feat_gen/total_action*100:.1f}% of action)")
                print(f"      └─ 純粋な推論 (model.forward): {total_forward:.2f}s ({total_forward/total_action*100:.1f}% of action)")
                print(f"      └─ 後処理 (numpy/exp/argmax) : {total_post_proc:.2f}s ({total_post_proc/total_action*100:.1f}% of action)")
            print(f"   [Profile] ゲーム更新 (step)     : {total_step:.2f}s ({total_step/sum_w_time*100:.1f}%)")
            print(f"   [Profile] 純粋なメモリ複製 (clone): {total_clone:.2f}s ({total_clone/sum_w_time*100:.1f}%)") # ラベル変更
            print(f"   [Profile] その他 (特徴量生成含む) : {total_other:.2f}s ({total_other/sum_w_time*100:.1f}%)") # ラベル変更
            print(f"   [IPC Overhead] 通信・シリアライズ推定 : {max(0.0, elapsed_time - (sum_w_time/cpu_count)):.2f}s")
        
        # optimize value net
        for key in ['discard', 'pick', 'koikoi']:
            value_net[key].train()
            train_loss = []
            
            if len(buffer.memory[key]) == 0:
                continue
                
            all_states = np.stack([t.state for t in buffer.memory[key]])
            all_rewards = np.array([t.reward for t in buffer.memory[key]], dtype=np.float32)
            
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
        
        buffer.clear()
        
        # update action net and agent
        if loop % n_loop_action_net_update == 0:
            type_dict = {'discard':'discard', 'discard-pick':'pick', 'draw-pick':'pick', 'koikoi':'koikoi'}
            
            action_net = {}
            for model_key, net_key in type_dict.items():
                raw_net = getattr(value_net[net_key], '_orig_mod', value_net[net_key])
                
                if net_key == 'discard':
                    cpu_model = DiscardModel().cpu()
                elif net_key == 'pick':
                    cpu_model = PickModel().cpu()
                elif net_key == 'koikoi':
                    cpu_model = KoiKoiModel().cpu()
                
                cpu_model.load_state_dict({k: v.cpu() for k, v in raw_net.state_dict().items()})
                # evalモードの生モデルを保持（ここではトレースしない）
                action_net[net_key] = cpu_model.eval()
                
            play_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi'],
                                           random_action_prob=random_action_prob_scheduler(score[-1]))
            test_agent = koikoilearn.Agent(action_net['discard'], action_net['pick'], action_net['koikoi']) 
        
        # arena test
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