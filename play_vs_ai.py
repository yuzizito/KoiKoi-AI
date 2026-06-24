import os
import sys
import torch
import koikoigui as gui
from koikoigame import KoiKoiGameState
import koikoilearn
from koikoinet2L import DiscardModel, PickModel, KoiKoiModel

import torch.nn.modules.linear as my_linear
setattr(my_linear, '_LinearWithBias', my_linear.Linear)

YOUR_NAME = 'Player'
AI_NAME = 'Com'
RECORD_PATH = 'gamerecords_player/rl/'

os.makedirs(RECORD_PATH, exist_ok=True)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_native_model(path, model_class):
    """ネイティブモデルを読み込み、互換性パッチを適用して評価モードにする"""
    model = model_class().to(DEVICE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
        
    state_dict = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()
        
    model.load_state_dict(state_dict)
    
    for module in model.modules():
        if type(module).__name__ == 'MultiheadAttention':
            if not hasattr(module, 'batch_first'): module.batch_first = False
        elif type(module).__name__ == 'TransformerEncoderLayer':
            if not hasattr(module, 'norm_first'): module.norm_first = False
            
    return model.float().eval()

# ゲーム状態の初期化
game_state = KoiKoiGameState(player_name=[YOUR_NAME, AI_NAME], record_path=RECORD_PATH, save_record=True)

# 最新の Agent クラスを使用
ai_agent = koikoilearn.Agent(
    load_native_model('model_agent/discard.pt', DiscardModel),
    load_native_model('model_agent/pick.pt', PickModel),
    load_native_model('model_agent/koikoi.pt', KoiKoiModel)
)

window = gui.InitGUI()
window = gui.UpdateGameStatusGUI(window, game_state)

while True:
    rs = game_state.round_state
    state = rs.state
    turn_player = rs.turn_player
    wait_action = rs.wait_action
    
    action = None
    
    if game_state.game_over:
        window = gui.ShowGameOverGUI(window, game_state)
        gui.Close(window)   
        break
    
    elif state == 'round-over':
        window = gui.ShowRoundOverGUI(window, game_state)
        game_state.new_round()
        window = gui.ClearBoardGUI(window)
        window = gui.UpdateGameStatusGUI(window, game_state)
        window = gui.UpdateCardAndYaku(window, game_state)  
    
    # プレイヤーのターン (1次元インデックスのカードIDを受け渡す)
    elif turn_player == 1:
        if state == 'discard':
            window = gui.UpdateTurnPlayer(window, game_state)
            window = gui.UpdateCardAndYaku(window, game_state)
            window, action = gui.WaitDiscardGUI(window, game_state)
            rs.discard(action)
            
        elif state == 'discard-pick':
            if wait_action:
                window, action = gui.WaitPickGUI(window, game_state)
            rs.discard_pick(action)
            
        elif state == 'draw':
            window = gui.UpdateCardAndYaku(window, game_state)
            window = gui.WaitAnyClick(window) 
            rs.draw(action)
            
        elif state == 'draw-pick':
            window = gui.ShowPileCardGUI(window, game_state)
            if wait_action:
                window, action = gui.WaitPickGUI(window, game_state)
            else:
                window = gui.WaitAnyClick(window)
            rs.draw_pick(action)
            
        elif state == 'koikoi':
            window = gui.UpdateCardAndYaku(window, game_state)
            if wait_action:
                window, action = gui.WaitKoiKoi(window)
            rs.claim_koikoi(action)
    
    # AIのターン (C++連動の高速推論から1次元アクションを取得)
    elif turn_player == 2:
        if state == 'discard':
            window = gui.UpdateTurnPlayer(window, game_state)
            window = gui.UpdateCardAndYaku(window, game_state)
            action = ai_agent.auto_action(game_state, for_test=True)
            rs.discard(action)
            window = gui.WaitAnyClick(window)
            window = gui.UpdateOpDiscardCardGUI(window, game_state)
            
        elif state == 'discard-pick':
            action = ai_agent.auto_action(game_state, for_test=True)  
            rs.discard_pick(action)
            window = gui.WaitAnyClick(window)
            
        elif state == 'draw':
            window = gui.UpdateCardAndYaku(window, game_state)
            rs.draw(action)
            window = gui.WaitAnyClick(window) 
            
        elif state == 'draw-pick':
            window = gui.ShowPileCardGUI(window, game_state)
            action = ai_agent.auto_action(game_state, for_test=True)
            window = gui.WaitAnyClick(window)
            rs.draw_pick(action)
            
        elif state == 'koikoi':
            window = gui.UpdateCardAndYaku(window, game_state)
            action = ai_agent.auto_action(game_state, for_test=True)
            window = gui.ShowOpKoiKoi(window, game_state, action)
            rs.claim_koikoi(action)
            
