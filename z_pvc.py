import os, torch
import sys
import koikoigui as gui
from koikoigame import KoiKoiGameState
import koikoilearn
from koikoinet_v2 import DiscardStrategyNet, PickStrategyNet, KoiKoiStrategyNet

import torch.nn.modules.linear as my_linear
setattr(my_linear, '_LinearWithBias', my_linear.Linear)

YOUR_NAME, AI_NAME = 'Player', 'Com'
RECORD_PATH = 'gamerecords_player/'
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
ai_agent = koikoilearn.Agent(
    load_native_model('model_v2/discard_strat.pt', DiscardStrategyNet),
    load_native_model('model_v2/pick_strat.pt', PickStrategyNet),
    load_native_model('model_v2/koikoi_strat.pt', KoiKoiStrategyNet)
)
window = gui.InitGUI()
window = gui.UpdateGameStatusGUI(window, game_state)

def handle_player_turn(rs, state, wait_action, window, game_state):
    """プレイヤーの入力待ちとアクション実行"""
    action = None
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
    return window

def handle_ai_turn(rs, state, wait_action, window, game_state):
    """AIのターンにおける推論実行とGUI更新"""
    action = None
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
    return window

window = gui.UpdateGameStatusGUI(window, game_state)

while True:
    # ゲーム終了判定
    if game_state.game_over:
        window = gui.ShowGameOverGUI(window, game_state)
        gui.Close(window)   
        break
    
    rs = game_state.round_state
    
    # ラウンド終了時の処理
    if rs.state == 'round-over':
        window = gui.ShowRoundOverGUI(window, game_state)
        game_state.new_round()
        window = gui.ClearBoardGUI(window)
        window = gui.UpdateGameStatusGUI(window, game_state)
        window = gui.UpdateCardAndYaku(window, game_state)
        continue
    
    # プレイヤーまたはAIのターン処理
    if rs.turn_player == 1:
        window = handle_player_turn(rs, rs.state, rs.wait_action, window, game_state)
    else:
        window = handle_ai_turn(rs, rs.state, rs.wait_action, window, game_state)
