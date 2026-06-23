import os
import torch
import koikoilearn
from koikoinet2L import DiscardModel, PickModel, KoiKoiModel

import torch.nn.modules.linear as my_linear
setattr(my_linear, '_LinearWithBias', my_linear.Linear)

A_PATHS = {'discard': 'model_agent/discard_state.pt', 'pick': 'model_agent/pick_state.pt', 'koikoi': 'model_agent/koikoi_state.pt'}
B_PATHS = {'discard': 'model_agent/discard_gu.pt', 'pick': 'model_agent/pick_gu.pt', 'koikoi': 'model_agent/koikoi_gu.pt'}
RECORD_PATH = 'gamerecords_agents/'

os.makedirs(RECORD_PATH, exist_ok=True)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_agent_models(paths):
    """指定されたパスから3つのモデルを読み込み、エージェントを構築する"""
    models = {'discard': DiscardModel(), 'pick': PickModel(), 'koikoi': KoiKoiModel()}
    
    for key, model in models.items():
        if os.path.exists(paths[key]):
            state_dict = torch.load(paths[key], map_location=DEVICE, weights_only=False)
            if isinstance(state_dict, torch.nn.Module):
                state_dict = state_dict.state_dict()
            model.load_state_dict(state_dict)
            
        model = model.to(DEVICE).eval()
        for module in model.modules():
            if type(module).__name__ == 'MultiheadAttention':
                if not hasattr(module, 'batch_first'): module.batch_first = False
            elif type(module).__name__ == 'TransformerEncoderLayer':
                if not hasattr(module, 'norm_first'): module.norm_first = False
                
    return koikoilearn.Agent(models['discard'], models['pick'], models['koikoi'])

agent_a = load_agent_models(A_PATHS)
agent_b = load_agent_models(B_PATHS)

arena = koikoilearn.Arena(
    agent_a, agent_b, 
    game_state_kwargs={'player_name': ['A', 'B'], 'record_path': RECORD_PATH, 'save_record': True}
)

print("[A vs B] AI同士の対戦を開始します...")
arena.multi_game_test(200)

print(f"\n対戦結果 -> A の勝利数: {arena.test_win_num[1]} / B の勝利数: {arena.test_win_num[2]} / 引き分け: {arena.test_win_num[0]}")