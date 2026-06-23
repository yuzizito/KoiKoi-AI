import random
import torch
import numpy as np
import koikoigame

class Agent:
    def __init__(self, discard_model, pick_model, koikoi_model, random_action_prob=None):
        self.model = {
            'discard': discard_model, 
            'discard-pick': pick_model, 
            'draw-pick': pick_model, 
            'koikoi': koikoi_model
        }
        for model in self.model.values():
            model.eval()

        if random_action_prob is None:
            random_action_prob = [0.0, 0.0, 0.0, 0.0]
            
        self.random_action_prob = {
            'discard': random_action_prob[0],
            'discard-pick': random_action_prob[1],
            'draw': 0.0,
            'draw-pick': random_action_prob[2],
            'koikoi': random_action_prob[3]
        }

    def __predict(self, state, feature_cpu, game_state):
        """モデル推論とC++バックエンドを利用した最適アクションの選択"""
        # 動的にモデルのデバイスを取得して転送
        device = next(self.model[state].parameters()).device
        feature_device = feature_cpu.to(device, non_blocking=True)
        
        with torch.inference_mode():
            output_tensor = self.model[state](feature_device)
        
        # GPUからCPUに変換
        logits_np = output_tensor.squeeze(0).cpu().numpy()
        
        state_types = {'discard': 0, 'discard-pick': 1, 'draw-pick': 1, 'koikoi': 2}
        state_type = state_types[state]
        turn_p = game_state.round_state.turn_player
        
        # C++側でマスクチェックとargmaxを一瞬で行う
        best_index = game_state.round_state.cpp_state.get_best_action(state_type, turn_p, logits_np)
        
        # こいこいの場合はboolean、それ以外はカードのインデックス(整数)を返す
        return bool(best_index) if state == 'koikoi' else best_index
    
    def auto_action(self, game_state, for_test=True):
        """現在のゲーム状態から自動でアクションを決定する"""
        rs = game_state.round_state
        if not rs.wait_action:
            return None
            
        state = rs.state
        
        # テスト・検証時の強制終了ロジック
        if for_test and state == 'koikoi':
            turn_player = rs.turn_player
            end_point = game_state.point[turn_player] + rs.yaku_point(turn_player)
            from koikoigame import MAX_ROUND
            
            if end_point >= 60:
                return False
            if game_state.round == MAX_ROUND:
                return end_point < 30
                
        # ランダム行動（探索）の判定
        if random.random() <= self.random_action_prob.get(state, 0.0):
            return self._auto_random_action(game_state, state)
            
        # 決定論的（モデル推論）行動
        feature_cpu = game_state.feature_tensor.unsqueeze(0)
        return self.__predict(state, feature_cpu, game_state)
    
    def _auto_random_action(self, game_state, state):
        """ランダムな有効アクションを選択する"""
        rs = game_state.round_state
        if state == 'discard':
            return random.choice(rs.hand[rs.turn_player])
        elif state in ('discard-pick', 'draw-pick'):
            return random.choice(rs.pairing_card)
        elif state == 'koikoi':
            return random.choice([True, False])
        return None
    
# 他のスクリプトからの呼び出し（後方互換性）のためのエイリアス
AgentForTest = Agent
    
class Arena:
    def __init__(self, agent_1, agent_2, game_state_kwargs=None):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.game_state_kwargs = game_state_kwargs or {}
        
        self.test_point = {1: [], 2: []}
        self.test_winner = []
        self.test_win_num = [0, 0, 0]
        self.test_win_rate = [0.0, 0.0, 0.0]
    
    def multi_game_test(self, num_game, clear_result=True): 
        """複数回の対戦テストを実行して結果を集計する"""
        if clear_result:
            self.test_point = {1: [], 2: []}
            self.test_winner = []
            
        for _ in range(num_game):
            self._duel()
            
        # 結果の集計 (0: 引き分け, 1: P1勝利, 2: P2勝利)
        for i in range(3):
            self.test_win_num[i] = self.test_winner.count(i)
            
        total_games = sum(self.test_win_num)
        if total_games > 0:
            self.test_win_rate = [n / total_games for n in self.test_win_num]
        
    def _duel(self):
        """1ゲーム分の対戦を行う"""
        game = koikoigame.KoiKoiGameState(**self.game_state_kwargs)
        
        while not game.game_over:
            if game.round_state.round_over:
                game.new_round()
            else:
                agent = self.agent_1 if game.round_state.turn_player == 1 else self.agent_2
                action = agent.auto_action(game, for_test=True)
                game.round_state.step(action)
                
        self.test_point[1].append(game.point[1])
        self.test_point[2].append(game.point[2])
        self.test_winner.append(game.winner)
    
    def test_result_str(self):
        """テスト結果のサマリーを文字列で返す"""
        if not self.test_winner:
            return "No games tested."
            
        total = sum(self.test_win_num)
        w_1, w_2, d = self.test_win_num[1], self.test_win_num[2], self.test_win_num[0]
        r_1, r_2, r_d = self.test_win_rate[1], self.test_win_rate[2], self.test_win_rate[0]
        mean_pt = np.mean(self.test_point[1])
        
        return (f"{total} games tested, {w_1} wins, {w_2} loses, {d} draws "
                f"({r_1:.2f}, {r_2:.2f}, {r_d:.2f}), {mean_pt:.1f} points")