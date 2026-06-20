import torch.nn.modules.linear as my_linear
setattr(my_linear, '_LinearWithBias', my_linear.Linear)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 23:58:43 2021

@author: guansanghai
"""

import os
from koikoigame import KoiKoiGameState
from koikoilearn import AgentForTest
import koikoigui as gui
import torch
from koikoinet2L import DiscardModel, PickModel, KoiKoiModel

your_name = 'Player'
ai_name = 'RL' # 'SL', 'RL'
record_path = 'gamerecords_player/'

# 
assert ai_name in ['RL','SL']
record_fold = record_path + ai_name + '/'

for path in [record_path, record_fold]:
    if not os.path.isdir(path):
        os.mkdir(path)

if ai_name == 'SL':
    discard_model_path = 'model_agent/discard_sl.pt'
    pick_model_path = 'model_agent/pick_sl.pt'
    koikoi_model_path = 'model_agent/koikoi_sl.pt'
elif ai_name == 'RL':
    discard_model_path = 'model_rl/discard_20_96.pt'
    pick_model_path = 'model_rl/pick_20_96.pt'
    koikoi_model_path = 'model_rl/koikoi_20_96.pt'
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_native_model(path, device, model_class):
    """通常の torch.load でモデルを読み込み、必要な属性の補正とFP32化を行う"""
    model = model_class().to(device)
    
    # 過去のモデル(Module全体)と新しいモデル(state_dict)の両方に対応
    loaded_data = torch.load(path, map_location=device, weights_only=False)
    if isinstance(loaded_data, torch.nn.Module):
        model.load_state_dict(loaded_data.state_dict())
    else:
        model.load_state_dict(loaded_data)
        
    for module in model.modules():
        if type(module).__name__ in ['MultiheadAttention', 'TransformerEncoderLayer']:
            if not hasattr(module, 'batch_first'): module.batch_first = False
            if not hasattr(module, 'norm_first'): module.norm_first = False
    return model.float().eval()

game_state = KoiKoiGameState(player_name=[your_name, ai_name], 
                             record_path=record_fold, 
                             save_record=True)

discard_model = load_native_model(discard_model_path, device, DiscardModel)
pick_model    = load_native_model(pick_model_path, device, PickModel)
koikoi_model  = load_native_model(koikoi_model_path, device, KoiKoiModel)

ai_agent = AgentForTest(discard_model, pick_model, koikoi_model)

window = gui.InitGUI()
window = gui.UpdateGameStatusGUI(window, game_state)

while True:
    state = game_state.round_state.state
    turn_player = game_state.round_state.turn_player
    wait_action = game_state.round_state.wait_action
    
    action = None
    
    if game_state.game_over == True:
        window = gui.ShowGameOverGUI(window, game_state)
        gui.Close(window)   
        break
    
    elif state == 'round-over':
        window = gui.ShowRoundOverGUI(window, game_state)
        game_state.new_round()
        window = gui.ClearBoardGUI(window)
        window = gui.UpdateGameStatusGUI(window, game_state)
        window = gui.UpdateCardAndYaku(window, game_state)  
    
    # Player's Turn
    elif turn_player == 1:
        if state == 'discard':
            window = gui.UpdateTurnPlayer(window, game_state)
            window = gui.UpdateCardAndYaku(window, game_state)
            window, action = gui.WaitDiscardGUI(window, game_state)
            game_state.round_state.discard(action)
            
        elif state == 'discard-pick':
            if wait_action:
                window, action = gui.WaitPickGUI(window, game_state)
            game_state.round_state.discard_pick(action)
            
        elif state == 'draw':
            window = gui.UpdateCardAndYaku(window, game_state)
            window = gui.WaitAnyClick(window) 
            
            game_state.round_state.draw(action)
            
        elif state == 'draw-pick':
            window = gui.ShowPileCardGUI(window, game_state)
            if wait_action:
                window, action = gui.WaitPickGUI(window, game_state)
            else:
                window = gui.WaitAnyClick(window)
            game_state.round_state.draw_pick(action)
            
        elif state == 'koikoi':
            window = gui.UpdateCardAndYaku(window, game_state)
            if wait_action:
                window, action = gui.WaitKoiKoi(window)
            game_state.round_state.claim_koikoi(action)
    
    # Opponent's Turn
    elif turn_player == 2:
        if state == 'discard':
            window = gui.UpdateTurnPlayer(window, game_state)
            window = gui.UpdateCardAndYaku(window, game_state)
            action = ai_agent.auto_action(game_state)
            game_state.round_state.discard(action)
            window = gui.WaitAnyClick(window)
            window = gui.UpdateOpDiscardCardGUI(window, game_state)
            
        elif state == 'discard-pick':
            action = ai_agent.auto_action(game_state)  
            game_state.round_state.discard_pick(action)
            window = gui.WaitAnyClick(window)
            
        elif state == 'draw':
            window = gui.UpdateCardAndYaku(window, game_state)
            game_state.round_state.draw(action)
            window = gui.WaitAnyClick(window) 
            
        elif state == 'draw-pick':
            window = gui.ShowPileCardGUI(window, game_state)
            action = ai_agent.auto_action(game_state)
            window = gui.WaitAnyClick(window)
            game_state.round_state.draw_pick(action)
            
        elif state == 'koikoi':
            window = gui.UpdateCardAndYaku(window, game_state)
            action = ai_agent.auto_action(game_state)
            window = gui.ShowOpKoiKoi(window, game_state, action)
            game_state.round_state.claim_koikoi(action)
            
