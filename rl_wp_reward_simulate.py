#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 14:23:52 2021

@author: guansanghai
"""

import torch
import numpy as np
import random

import koikoigame
import koikoilearn

import pickle
import multiprocessing
from koikoinet2L import DiscardModel, PickModel, KoiKoiModel

rl_folder = 'model_rl_wp'

discard_model_path = f'{rl_folder}/discard_0_0.pt'
pick_model_path = f'{rl_folder}/pick_0_0.pt'
koikoi_model_path = f'{rl_folder}/koikoi_0_0.pt'

def load_model_state(path, model_class):
    model = model_class()
    loaded_data = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    
    if isinstance(loaded_data, torch.nn.Module):
        model.load_state_dict(loaded_data.state_dict())
    else:
        model.load_state_dict(loaded_data)
        
    for module in model.modules():
        if type(module).__name__ in ['MultiheadAttention', 'TransformerEncoderLayer']:
            if not hasattr(module, 'batch_first'): module.batch_first = False
            if not hasattr(module, 'norm_first'): module.norm_first = False
    return model.eval()

# if __name__ == '__main__': の直前のロード部分を修正
discard_model = load_model_state(discard_model_path, DiscardModel)
pick_model = load_model_state(pick_model_path, PickModel)
koikoi_model = load_model_state(koikoi_model_path, KoiKoiModel)
ai_agent = koikoilearn.Agent(discard_model, pick_model, koikoi_model)

def make_win_prob_dict(agent, round_num, point, dealer, n_test):
    game_state_kwargs = {'round_num':round_num, 
                         'init_point':[point,60-point],
                         'init_dealer':dealer}
    arena = koikoilearn.Arena(agent, agent, game_state_kwargs)
    arena.multi_game_test(n_test)
    result = (round_num, point, dealer, arena.test_win_num)
    print(result)
    return result

if __name__ == '__main__':

    dealer = 1
    n_test = 200
    
    result = []
    pool = multiprocessing.Pool(24)
    for round_num in [8,7,6,5,4,3,2,1]:
        for point in range(1,60):
            args = (ai_agent, ai_agent, round_num, point, dealer, n_test)
            pool.apply_async(make_win_prob_dict, args=args, callback=result.append)
    pool.close()
    pool.join()

    '''
    for round_num in [8,7,6,5,4,3,2,1]:
        for point in range(1,60):
            result.append(make_win_prob_dict(ai_agent, ai_agent, round_num, point, dealer, n_test))
    '''
    
    result.sort()
    print(result)
    
    with open('result_wp_sim.pkl','wb') as f:
        pickle.dump(result,f)

