# author: shguan3
import configparser
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import koikoicore

MAX_ROUND = 6


def apply_koikoi_config(cfg_path: str = "koikoi.cfg") -> None:
    default_rules = {
        "five_lights": 10, "four_lights": 8, "rainy_four_lights": 7,
        "three_lights": 5, "bdb": 5,
        "enable_flower_sake": True, "enable_moon_sake": True,
        "flower_moon_base_pt": 1, "flower_moon_koikoi_pt": 3,
        "red_blue_ribbon": 10, "red_ribbon": 5, "blue_ribbon": 5,
    }

    if os.path.exists(cfg_path):
        config = configparser.ConfigParser()
        config.read(cfg_path, encoding="utf-8")
        if "Rules" in config:
            cfg_rules = config["Rules"]
            apply_dict = {}
            for key, default_val in default_rules.items():
                if key in cfg_rules:
                    val = (
                        cfg_rules.getboolean(key)
                        if isinstance(default_val, bool)
                        else cfg_rules.getint(key)
                    )
                    apply_dict[key] = val
            koikoicore.set_rules(apply_dict)
            return

    koikoicore.set_rules(default_rules)


apply_koikoi_config()


class KoiKoiRoundState:
    """1ラウンド内の配札・場札・手番進行を管理するステートマシン"""

    def __init__(self, dealer: Optional[int] = None):
        self.dealer: int = random.choice([1, 2]) if dealer is None else dealer
        self.turn_16: int = 1
        self.turn_point: int = 0
        self.state: str = "init"
        self.wait_action: bool = False
        self.winner: Optional[int] = None
        self.exhausted: bool = False

        self.hand: Dict[int, List[int]] = {1: [], 2: []}
        self.pile: Dict[int, List[int]] = {1: [], 2: []}
        self.koikoi: Dict[int, List[int]] = {1: [0] * 8, 2: [0] * 8}

        self.field_slot: List[int] = []
        self.stock: List[int] = []
        self.show: List[int] = []
        self.collect: List[int] = []
        self.log: Dict = {}

        self.cpp_state = koikoicore.KoiKoiStateManager()
        self._deal_card()

    @property
    def turn_player(self) -> int:
        return 1 if (self.turn_16 + self.dealer) % 2 == 0 else 2

    @property
    def idle_player(self) -> int:
        return 3 - self.turn_player

    @property
    def turn_8(self) -> int:
        return (self.turn_16 + 1) // 2

    @property
    def field(self) -> List[int]:
        return sorted(c for c in self.field_slot if c != -1)

    @property
    def pairing_card(self) -> List[int]:
        if not self.show:
            return []
        target_suit = self.show[0] // 4
        return [c for c in self.field if c // 4 == target_suit]

    @property
    def koikoi_num(self) -> Dict[int, int]:
        return {1: sum(self.koikoi[1]), 2: sum(self.koikoi[2])}

    @property
    def round_over(self) -> bool:
        return self.state == "round-over"

    @property
    def round_point(self) -> Dict[int, Optional[int]]:
        if self.winner is None:
            return {1: None, 2: None}
        if self.exhausted:
            return {1: 1, 2: -1} if self.dealer == 1 else {1: -1, 2: 1}

        pts = self.yaku_point(self.winner)
        return {1: pts, 2: -pts} if self.winner == 1 else {1: -pts, 2: pts}

    @property
    def action_mask(self) -> np.ndarray:
        st_type = 0 if self.state == "discard" else (1 if self.state in ("discard-pick", "draw-pick") else 2)
        return self.cpp_state.get_action_mask(st_type, self.turn_player)

    def new_round(self) -> None:
        self.__init__(dealer=self.winner)

    def yaku_point(self, player: int) -> int:
        return self.cpp_state.get_yaku_point(player, sum(self.koikoi[player]))

    def yaku(self, player: int) -> List[Tuple[int, str, int]]:
        return self.cpp_state.get_yaku_list(player, sum(self.koikoi[player]))

    def step(self, action: Optional[Union[int, bool]]) -> None:
        # Design Intent: bool(action) 強制変換を廃止し、生値をディスパッチ
        if self.state == "discard":
            self.discard(action if isinstance(action, int) else None)
        elif self.state == "discard-pick":
            self.discard_pick(action if isinstance(action, int) else None)
        elif self.state == "draw":
            self.draw()
        elif self.state == "draw-pick":
            self.draw_pick(action if isinstance(action, int) else None)
        elif self.state == "koikoi":
            self.claim_koikoi(action if isinstance(action, bool) else None)

    # =========================================================================
    # 公開API（ステート進行）
    # =========================================================================

    def discard(self, card: Optional[int] = None) -> None:
        self.turn_point = self.yaku_point(self.turn_player)
        if card is not None:
            self.hand[self.turn_player].remove(card)
            self.show = [card]

        pairs = self.pairing_card
        self.log[f"turn{self.turn_16}"] = {"pairCard": pairs.copy()}

        pair_bb = sum(1 << c for c in pairs)
        self.cpp_state.discard(self.turn_player, card or 0, pair_bb, self.turn_16)

        self.state = "discard-pick"
        self.wait_action = len(pairs) == 2

    def discard_pick(self, card: Optional[int] = None) -> None:
        # ガードを撤廃し、None（自動配置・自動獲得）をそのまま低レイヤへ流す
        self._collect_card(card)
        coll_bb = sum(1 << c for c in self.collect)
        field_bb = sum(1 << c for c in self.field_slot if c != -1)

        self.cpp_state.discard_pick(self.turn_player, coll_bb, field_bb, self.turn_16)
        self.state = "draw"
        self.wait_action = False

    def draw(self, card: Optional[int] = None) -> None:
        drawn = self.stock.pop() if self.stock else 0
        self.show = [drawn]
        pairs = self.pairing_card
        self.log[f"turn{self.turn_16}"]["pairCard2"] = pairs.copy()

        pair_bb = sum(1 << c for c in pairs)
        self.cpp_state.draw(drawn, pair_bb, self.turn_16)

        self.state = "draw-pick"
        self.wait_action = len(pairs) == 2

    def draw_pick(self, card: Optional[int] = None) -> None:
        self._collect_card(card)
        coll_bb = sum(1 << c for c in self.collect)
        field_bb = sum(1 << c for c in self.field_slot if c != -1)

        self.cpp_state.draw_pick(self.turn_player, coll_bb, field_bb, self.turn_16)
        self.state = "koikoi"

        curr_pts = self.yaku_point(self.turn_player)
        self.wait_action = (curr_pts > self.turn_point) and (self.turn_8 < 8)

    def claim_koikoi(self, is_koikoi: Optional[bool] = None) -> None:
        # =====================================================================
        # 3値ロジック (True:続行 / False:上がり / None:役なしスキップ) の完全復元
        # =====================================================================
        if (self.yaku_point(self.turn_player) > self.turn_point) and (self.turn_8 == 8):
            is_koikoi = False

        self.koikoi[self.turn_player][self.turn_8 - 1] = 1 if is_koikoi is True else 0

        if is_koikoi is False:  # 明示的な「勝負(上がり)」のときだけラウンド終了
            self.state = "round-over"
            self.wait_action = False
            self.winner = self.turn_player
        elif self.turn_16 == 16:
            self.state = "round-over"
            self.wait_action = False
            self.exhausted = True
            self.winner = self.dealer
        else:                   # True(こいこい) または None(役なし自動パス) のとき手番交代
            self.turn_16 += 1
            self.state = "discard"
            self.wait_action = True

    # =========================================================================
    # 低レイヤ・アルゴリズム
    # =========================================================================

    def _collect_card(self, card: Optional[int] = None) -> None:
        pairs = self.pairing_card
        if not pairs:
            self.collect = []
            self.field_slot[self.field_slot.index(-1)] = self.show[0]
        elif len(pairs) in (1, 3):
            self.collect = self.show + pairs
            for pc in pairs:
                self.field_slot[self.field_slot.index(pc)] = -1
            self.pile[self.turn_player].extend(self.collect)
        else:  # len(pairs) == 2 の時だけ実カードID(int)が使われる
            self.collect = self.show + [card or 0]
            self.field_slot[self.field_slot.index(card or 0)] = -1
            self.pile[self.turn_player].extend(self.collect)

    def _deal_card(self) -> None:
        def has_four_same_suit(cards: List[int]) -> bool:
            counts = [0] * 12
            for c in cards:
                counts[c // 4] += 1
                if counts[c // 4] == 4:
                    return True
            return False

        while True:
            deck = list(range(48))
            random.shuffle(deck)
            h1, h2, f = sorted(deck[:8]), sorted(deck[8:16]), sorted(deck[16:24])

            if not (has_four_same_suit(h1) or has_four_same_suit(h2) or has_four_same_suit(f)):
                self.hand[1], self.hand[2] = h1, h2
                self.field_slot = f + [-1] * 10
                self.stock = deck[24:]
                break

        self.log["basic"] = {"initBoard": self.field.copy()}

        h1_bb = sum(1 << c for c in self.hand[1])
        h2_bb = sum(1 << c for c in self.hand[2])
        f_bb = sum(1 << c for c in self.field_slot if c != -1)
        s_bb = sum(1 << c for c in self.stock)
        self.cpp_state.init_board(h1_bb, h2_bb, f_bb, s_bb)

        self.state = "discard"
        self.wait_action = True


class KoiKoiGameState:
    def __init__(
        self,
        round_num: int = 1,
        round_total: int = MAX_ROUND,
        init_point: Tuple[int, int] = (30, 30),
        init_dealer: Optional[int] = None,
        player_name: Tuple[str, str] = ("Player1", "Player2"),
        record_path: str = "",
        save_record: bool = False,
    ):
        self.round_total = round_total
        self.init_point: Tuple[int, int] = (init_point[0], init_point[1])
        self.init_dealer = init_dealer
        self.player_name = {1: player_name[0], 2: player_name[1]}
        self.record_path = record_path
        self.save_record = save_record

        self.round = round_num
        self.point = {1: self.init_point[0], 2: self.init_point[1]}
        self.game_over = False
        self.winner: Optional[int] = None
        self.log: Dict = {}

        self.round_state = KoiKoiRoundState(dealer=self.init_dealer)
        self._init_record()

    @property
    def feature_np(self) -> np.ndarray:
        rs = self.round_state
        tp, ip = rs.turn_player, rs.idle_player
        return rs.cpp_state.get_feature(
            rs.state == "koikoi",
            self.point[tp], self.point[ip],
            self.round, rs.turn_16, rs.dealer,
            sum(rs.koikoi[tp]), sum(rs.koikoi[ip]),
            tp, ip, MAX_ROUND,
        )

    @property
    def feature_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.feature_np)

    def new_game(self) -> None:
        self.__init__(
            round_num=1, round_total=self.round_total,
            init_point=self.init_point, init_dealer=self.init_dealer,
            player_name=(self.player_name[1], self.player_name[2]),
            record_path=self.record_path, save_record=self.save_record,
        )

    def new_round(self) -> None:
        if not self.round_state.round_over:
            raise RuntimeError("Cannot start new round before current round is over.")

        pts = self.round_state.round_point
        self.point[1] += pts[1] or 0
        self.point[2] += pts[2] or 0
        self._record_round_result()

        if self.point[1] <= 0 or self.point[2] <= 0 or self.round >= self.round_total:
            self.game_over = True
            self.winner = 1 if self.point[1] > self.point[2] else (2 if self.point[2] > self.point[1] else 0)
            self._record_game_result()
        else:
            self.round_state.new_round()
            self.round += 1

    def _init_record(self) -> None:
        self.log = {
            "info": {
                "startTime": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
                "endTime": None,
                "player1Name": self.player_name[1], "player2Name": self.player_name[2],
                "player1InitPts": self.point[1], "player2InitPts": self.point[2],
                "numRound": self.round_total,
            },
            "result": {"isOver": False, "gameWinner": None, "player1EndPts": None, "player2EndPts": None},
            "record": {},
        }

    def _record_round_result(self) -> None:
        r_log = self.round_state.log.copy()
        basic = r_log.setdefault("basic", {}).copy()
        basic.update({
            "player1RoundPts": self.round_state.round_point[1],
            "player2RoundPts": self.round_state.round_point[2],
            "Dealer": self.round_state.dealer,
        })
        r_log["basic"] = basic
        self.log["record"][f"round{self.round}"] = r_log

    def _record_game_result(self) -> None:
        self.log["info"]["endTime"] = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log["result"] = {
            "isOver": True, "gameWinner": self.winner,
            "player1EndPts": self.point[1], "player2EndPts": self.point[2],
        }
        if self.save_record and self.record_path:
            fname = f"{self.record_path}{self.log['info']['startTime']} {self.player_name[1]} vs {self.player_name[2]}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(self.log, f, ensure_ascii=False, indent=2)