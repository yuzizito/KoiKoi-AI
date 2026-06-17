#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:11:56 2021

@author: guansanghai
"""

import random
import numpy as np
import time
import torch
import koikoigame

class Agent():
    def __init__(self, discard_model, pick_model, koikoi_model, random_action_prob=[0.,0.,0.,0.]):
        self.model = {'discard':discard_model, 'discard-pick':pick_model, 
                      'draw-pick':pick_model, 'koikoi':koikoi_model}
        for key in self.model.keys():
            self.model[key].eval()

        card_list = [[i+1,j+1] for i in range(12) for j in range(4)]
        self.action_dict = {'discard':card_list, 'discard-pick':card_list, 
                            'draw-pick':card_list, 'koikoi':(False, True)}
        
        self.random_action_prob = {'discard':random_action_prob[0],
                                   'discard-pick':random_action_prob[1],
                                   'draw':0,
                                   'draw-pick':random_action_prob[2],
                                   'koikoi':random_action_prob[3]}

        # 計測タイマー
        self.t_feat_gen = 0.0     # CPU上での特徴量組み立て時間
        self.t_cpu_to_gpu = 0.0   # CPUからGPUへのテンソル転送時間
        self.t_forward = 0.0      # GPU上での純粋なモデル順伝播時間
        self.t_post_proc = 0.0    # GPU上でのマスク処理・argmax・.item()引き戻し時間

    def _move_to_gpu(self, feature_cpu):
        t_start = time.perf_counter()
        feature_gpu = feature_cpu.to('cuda:0', non_blocking=True)
        self.t_cpu_to_gpu += (time.perf_counter() - t_start)
        return feature_gpu

    def __predict(self, state, feature_gpu, mask):
        """【単発推論用】"""
        device = feature_gpu.device
        
        # 1. GPU上での純粋な forward 時間を計測
        t1 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_tensor = self.model[state](feature_gpu)
        self.t_forward += (time.perf_counter() - t1)
        
        # 2. 後処理時間を計測 (マスク、argmax、.item() 以外はCPUに戻さない)
        t2 = time.perf_counter()
        output = output_tensor.squeeze(0)
        
        mask_np = np.array(mask, dtype=np.bool_)
        mask_tensor = torch.from_numpy(mask_np).to(device, non_blocking=True)
        min_val = torch.finfo(output.dtype).min
        output = output.masked_fill(~mask_tensor, min_val)
        
        best_index = output.argmax().item()
        action_output = self.action_dict[state][best_index]
        
        self.t_post_proc += (time.perf_counter() - t2)
        return action_output
    
    def predict_batch(self, state, features_cpu, masks_np):
        if features_cpu is None or features_cpu.size(0) == 0:
            return []
            
        # 1. 結合されたバッチを 1回 でGPUへ転送
        feature_gpu = self._move_to_gpu(features_cpu)
        device = feature_gpu.device
        
        # 2. GPU上での純粋な一括 forward（JITモデルが呼ばれます）
        t1 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_tensor = self.model[state](feature_gpu)
        self.t_forward += (time.perf_counter() - t1)
        
        # 3. GPU完結型のバッチ後処理
        t2 = time.perf_counter()
        
        masks_bool = masks_np.astype(np.bool_)
        masks_tensor = torch.from_numpy(masks_bool).to(device, non_blocking=True)
        
        min_val = torch.finfo(output_tensor.dtype).min
        output_tensor = output_tensor.masked_fill(~masks_tensor, min_val)
        
        best_indices = output_tensor.argmax(dim=1)
        best_indices_cpu = best_indices.cpu().tolist()
        actions = [self.action_dict[state][idx] for idx in best_indices_cpu]
            
        self.t_post_proc += (time.perf_counter() - t2)
        return actions
    
    def auto_action(self, game_state, use_mask=True, for_test=False):
        p = random.random()
        if game_state.round_state.wait_action == False:
            return None
        if (for_test == True) and (game_state.round_state.state == 'koikoi'):
            turn_player = game_state.round_state.turn_player
            end_point = game_state.point[turn_player] \
                + game_state.round_state.yaku_point(turn_player)
            if game_state.round == 8 and end_point < 30:
                return True
            if game_state.round == 8 and end_point > 30:
                return False
            if end_point >= 60:
                return False
        if p > self.random_action_prob[game_state.round_state.state]:
            with torch.inference_mode():
                return self.auto_definitely_action(game_state, use_mask)
        else:
            return self.auto_random_action(game_state)
    
    def auto_definitely_action(self, game_state, use_mask=True):
        action_output = None
        if game_state.round_state.wait_action == True:
            state = game_state.round_state.state
            
            # 【責務1】特徴量の作成（完全にCPU上の処理）
            t0 = time.perf_counter()
            feature_cpu = game_state.feature_tensor.unsqueeze(0)
            self.t_feat_gen += (time.perf_counter() - t0)
            
            # 【責務2】GPUへの転送
            feature_gpu = self._move_to_gpu(feature_cpu)
            
            # 【責務3】推論の実行
            mask = game_state.round_state.action_mask
            action_output = self.__predict(state, feature_gpu, mask)     
        return action_output
    
    def auto_random_action(self, game_state):
        action_output = None
        if game_state.round_state.wait_action == True:
            state = game_state.round_state.state
            if state == 'discard':
                turn_player = game_state.round_state.turn_player
                action_output = random.choice(game_state.round_state.hand[turn_player])
            elif state in ['discard-pick', 'draw-pick']:
                action_output = random.choice(game_state.round_state.pairing_card)
            elif state == 'koikoi':
                action_output = random.choice([True, False])
        return action_output
        
        
class Arena():
    def __init__(self, agent_1, agent_2, game_state_kwargs={}):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.game_state_kwargs = game_state_kwargs
        
        self.test_point = {1:[], 2:[]}
        self.test_winner = []
    
    def multi_game_test(self, num_game, clear_result=True): 
        def n_count(l,x):
            return np.sum(np.array(l)==x)
        if clear_result:
            self.clear_test_result()
        for ii in range(num_game):
            self.__duel()
        self.test_win_num = [n_count(self.test_winner,ii) for ii in [0,1,2]]
        self.test_win_rate = [n/sum(self.test_win_num) for n in self.test_win_num]
        return
        
    def __duel(self):
        self.game_state = koikoigame.KoiKoiGameState(**self.game_state_kwargs)
        while True:
            if self.game_state.game_over == True:
                break
            elif self.game_state.round_state.round_over == True:
                self.game_state.new_round()
            else:
                if self.game_state.round_state.turn_player == 1:
                    action = self.agent_1.auto_action(self.game_state)
                    self.game_state.round_state.step(action)
                else:
                    action = self.agent_2.auto_action(self.game_state)
                    self.game_state.round_state.step(action)
        self.test_point[1].append(self.game_state.point[1])
        self.test_point[2].append(self.game_state.point[2])
        self.test_winner.append(self.game_state.winner)
        return
    
    def test_result_str(self):
        assert len(self.test_winner) > 0
        win_num = self.test_win_num
        win_rate = self.test_win_rate
        s = f'{sum(win_num)} games tested, '
        s += f'{win_num[1]} wins, {win_num[2]} loses, {win_num[0]} draws '
        s += f'({win_rate[1]:.2f}, {win_rate[2]:.2f}, {win_rate[0]:.2f}), '
        s += f'{np.mean(self.test_point[1]):.1f} points'
        return s
                    
    def clear_test_result(self):
        self.test_point = {1:[], 2:[]}
        self.test_winner = []
        return 


class AgentForTest():
    def __init__(self, discard_model, pick_model, koikoi_model, random_action_prob=[0.,0.,0.,0.]):
        self.model = {'discard':discard_model, 'discard-pick':pick_model, 
                      'draw-pick':pick_model, 'koikoi':koikoi_model}
        for key in self.model.keys():
            self.model[key].eval()

        card_list = [[i+1,j+1] for i in range(12) for j in range(4)]
        self.action_dict = {'discard':card_list, 'discard-pick':card_list, 
                            'draw-pick':card_list, 'koikoi':(False, True)}
        
        self.random_action_prob = {'discard':random_action_prob[0],
                                   'discard-pick':random_action_prob[1],
                                   'draw':0,
                                   'draw-pick':random_action_prob[2],
                                   'koikoi':random_action_prob[3]}
        
        # ★ 精密計測用のタイマー変数を追加
        self.t_feat_gen = 0.0
        self.t_forward = 0.0
        self.t_post_proc = 0.0
        self.last_feature = {'discard': None, 'discard-pick': None, 'draw-pick': None, 'koikoi': None}

    def __predict(self, state, feature, mask):
        t1 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            output_tensor = self.model[state](feature)
        self.t_forward += (time.perf_counter() - t1)
        
        t2 = time.perf_counter()
        output = output_tensor.squeeze(0)

        mask_np = np.array(mask, dtype=np.bool_)
        mask_tensor = torch.from_numpy(mask_np).to(output.device, non_blocking=True)
        min_val = torch.finfo(output.dtype).min
        output = output.masked_fill(~mask_tensor, min_val)
        
        best_index = output.argmax().item()
        action_output = self.action_dict[state][best_index]
        
        self.t_post_proc += (time.perf_counter() - t2)
        return action_output
    
    def auto_action(self, game_state, use_mask=True, for_test=True):
        p = random.random()
        if game_state.round_state.wait_action == False:
            return None
        if (for_test == True) and (game_state.round_state.state == 'koikoi'):
            turn_player = game_state.round_state.turn_player
            end_point = game_state.point[turn_player] \
                + game_state.round_state.yaku_point(turn_player)
            if game_state.round == 8 and end_point < 30:
                return True
            if game_state.round == 8 and end_point > 30:
                return False
            if end_point >= 60:
                return False
        if p > self.random_action_prob[game_state.round_state.state]:
            return self.auto_definitely_action(game_state, use_mask)
        else:
            return self.auto_random_action(game_state)
    
    def auto_definitely_action(self, game_state, use_mask=True):
        action_output = None
        if game_state.round_state.wait_action==True:
            state = game_state.round_state.state
            # 1. feature_tensor 生成時間を計測
            t0 = time.perf_counter()
            feature = game_state.feature_tensor.unsqueeze(0)
            self.t_feat_gen += (time.perf_counter() - t0)
            self.last_feature[state] = feature.squeeze(0).detach()
            mask = game_state.round_state.action_mask
            action_output = self.__predict(state, feature, mask)     
        return action_output
    
    def auto_random_action(self, game_state):
        action_output = None
        if game_state.round_state.wait_action == True:
            state = game_state.round_state.state
            if state == 'discard':
                turn_player = game_state.round_state.turn_player
                action_output = random.choice(game_state.round_state.hand[turn_player])
            elif state in ['discard-pick', 'draw-pick']:
                action_output = random.choice(game_state.round_state.pairing_card)
            elif state == 'koikoi':
                action_output = random.choice([True, False])
        return action_output      