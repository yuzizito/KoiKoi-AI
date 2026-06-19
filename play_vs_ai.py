import torch.nn.modules.linear as my_linear; my_linear._LinearWithBias = my_linear.Linear
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
import torch # 1.8.1

# Demo for playing 8-round koi-koi games vs trained AI
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
    discard_model_path = 'traced_discard.pt'
    pick_model_path = 'traced_pick.pt'
    koikoi_model_path = 'traced_koikoi.pt'
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------
# 【JIT対応】torch.jit.load を用い、ロード段階からデバイスを決定する
# ---------------------------------------------------------

game_state = KoiKoiGameState(player_name=[your_name,ai_name], 
                             record_path=record_fold, 
                             save_record=True)

if ai_name == 'RL':
    # 強化学習で作成されたJITモデルは内部がbfloat16化されているため、ロードして即座に.float()でFP32化する
    discard_model = torch.jit.load(discard_model_path, map_location=device).float()
    pick_model    = torch.jit.load(pick_model_path, map_location=device).float()
    koikoi_model  = torch.jit.load(koikoi_model_path, map_location=device).float()
else:
    # 従来のSLモデル（ネイティブnn.Module）用のフォールバック
    discard_model = torch.load(discard_model_path, map_location=device, weights_only=False)
    for module in discard_model.modules():
        if type(module).__name__ in ['MultiheadAttention', 'TransformerEncoderLayer']:
            if not hasattr(module, 'batch_first'): module.batch_first = False
            if not hasattr(module, 'norm_first'): module.norm_first = False
            
    pick_model = torch.load(pick_model_path, map_location=device, weights_only=False)
    for module in pick_model.modules():
        if type(module).__name__ in ['MultiheadAttention', 'TransformerEncoderLayer']:
            if not hasattr(module, 'batch_first'): module.batch_first = False
            if not hasattr(module, 'norm_first'): module.norm_first = False
            
    koikoi_model = torch.load(koikoi_model_path, map_location=device, weights_only=False)
    for module in koikoi_model.modules():
        if type(module).__name__ in ['MultiheadAttention', 'TransformerEncoderLayer']:
            if not hasattr(module, 'batch_first'): module.batch_first = False
            if not hasattr(module, 'norm_first'): module.norm_first = False

    discard_model = discard_model.float()
    pick_model    = pick_model.float()
    koikoi_model  = koikoi_model.float()

discard_model.eval()
pick_model.eval()
koikoi_model.eval()

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
            
