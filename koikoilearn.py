# author: shguan3
"""
Koi-Koi Reinforcement Learning Decision & Arena Engine
Refactored for zero hot-path overhead, strict type annotations, and clean state routing.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import koikoigame
from koikoigame import MAX_ROUND

# ステート名からC++側のビットボード用EnumIDへの静的マッピング
_STATE_TYPE_MAP: Dict[str, int] = {
    "discard": 0,
    "discard-pick": 1,
    "draw-pick": 1,
    "koikoi": 2,
}


class Agent:
    """強化学習モデルを統括し、盤面状態をもとに最適行動を選択する意思決定エージェント"""

    def __init__(
        self,
        discard_model: Union[nn.Module, torch.jit.ScriptModule],
        pick_model: Union[nn.Module, torch.jit.ScriptModule],
        koikoi_model: Union[nn.Module, torch.jit.ScriptModule],
    ):
        self.models: Dict[str, Union[nn.Module, torch.jit.ScriptModule]] = {
            "discard": discard_model,
            "discard-pick": pick_model,
            "draw-pick": pick_model,
            "koikoi": koikoi_model,
        }

        # Design Intent: ホットパス高速化のため、初期化時にデバイスを確定キャッシュする
        try:
            self.device: torch.device = next(discard_model.parameters()).device
        except Exception:
            self.device = torch.device("cpu")

        for m in self.models.values():
            m.eval()

    def auto_action(
        self, game_state: koikoigame.KoiKoiGameState, for_test: bool = True
    ) -> Optional[Union[int, bool]]:
        """現在のゲーム状態から自動でアクションを決定する"""
        rs = game_state.round_state
        if not rs.wait_action:
            return None

        state = rs.state

        # テスト・検証時のヒューリスティックな強制打ち切りロジック
        if for_test and state == "koikoi":
            tp = rs.turn_player
            current_pt = game_state.point[tp] + rs.yaku_point(tp)

            if current_pt >= 60:
                return False
            if game_state.round >= MAX_ROUND:
                return current_pt < 30

        feat_cpu = game_state.feature_tensor.unsqueeze(0)
        return self._predict(state, feat_cpu, game_state)

    def _predict(
        self,
        state: str,
        feat_cpu: torch.Tensor,
        game_state: koikoigame.KoiKoiGameState,
    ) -> Union[int, bool]:
        """モデル推論とC++バックエンドを利用した最適アクションの選択"""
        model = self.models[state]
        feat_device = feat_cpu.to(self.device, non_blocking=True)

        with torch.inference_mode():
            output = model(feat_device)

        # NeuRDモデルは (logits, value) のタプルを返す
        logits_tensor = output[0] if isinstance(output, tuple) else output
        logits_np = logits_tensor.squeeze(0).cpu().numpy()

        st_type = _STATE_TYPE_MAP[state]
        turn_p = game_state.round_state.turn_player

        best_idx = game_state.round_state.cpp_state.get_best_action(
            st_type, turn_p, logits_np
        )
        return bool(best_idx) if state == "koikoi" else int(best_idx)

    def _auto_random_action(
        self, game_state: koikoigame.KoiKoiGameState, state: str
    ) -> Optional[Union[int, bool]]:
        """ランダムな有効アクションを選択する"""
        rs = game_state.round_state
        if state == "discard":
            return random.choice(rs.hand[rs.turn_player])
        elif state in ("discard-pick", "draw-pick"):
            return random.choice(rs.pairing_card)
        elif state == "koikoi":
            return random.choice([True, False])
        return None


# 後方互換性エイリアス
AgentForTest = Agent


class Arena:
    """エージェント同士を複数回対戦させ、勝率や平均スコアを測定する検証環境"""

    def __init__(
        self,
        agent_1: Agent,
        agent_2: Agent,
        game_state_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.game_state_kwargs: Dict[str, Any] = game_state_kwargs or {}

        self.test_point: Dict[int, List[int]] = {1: [], 2: []}
        self.test_winner: List[int] = []
        self.test_win_num: List[int] = [0, 0, 0]
        self.test_win_rate: List[float] = [0.0, 0.0, 0.0]

    def multi_game_test(self, num_game: int, clear_result: bool = True) -> None:
        """複数回の対戦テストを実行して結果を集計する"""
        if clear_result:
            self.test_point = {1: [], 2: []}
            self.test_winner = []

        for i in range(num_game):
            self.game_state_kwargs["init_dealer"] = (i % 2) + 1
            self._duel()

        # 結果集計 (0: 引き分け, 1: P1勝利, 2: P2勝利)
        for i in range(3):
            self.test_win_num[i] = self.test_winner.count(i)

        total_games = sum(self.test_win_num)
        if total_games > 0:
            self.test_win_rate = [n / total_games for n in self.test_win_num]

    def _duel(self) -> None:
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
        self.test_winner.append(game.winner or 0)

    def test_result_str(self) -> str:
        """テスト結果のサマリーを文字列で返す"""
        if not self.test_winner:
            return "No games tested."

        total = sum(self.test_win_num)
        w_1, w_2, d = self.test_win_num[1], self.test_win_num[2], self.test_win_num[0]
        r_1, r_2, r_d = self.test_win_rate[1], self.test_win_rate[2], self.test_win_rate[0]
        mean_pt = float(np.mean(self.test_point[1]))

        return (
            f"{total} games tested, {w_1} wins, {w_2} loses, {d} draws "
            f"({r_1:.2f}, {r_2:.2f}, {r_d:.2f}), {mean_pt:.1f} points"
        )