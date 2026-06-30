# author: shguan3
"""
Koi-Koi Human vs AI GUI Play Engine (Unified Single-Script Architecture)
"""

import configparser
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import z_gui
from neurd import NeuRD

import core

MAX_RND = core.MAX_RND
REC_DIR = "gamerecords_player/"
os.makedirs(REC_DIR, exist_ok=True)
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_cfg(path: str = "koikoi.cfg") -> None:
    rules = {
        "lt5": 10, "lt4": 8, "rainy4": 7, "lt3": 5, "bdb": 5,
        "ena_f_sake": True, "ena_m_sake": True,
        "fm_base": 1, "fm_koi": 3,
        "rb_rib": 10, "r_rib": 5, "b_rib": 5,
    }
    if os.path.exists(path):
        cp = configparser.ConfigParser()
        cp.read(path, encoding="utf-8")
        if "Rules" in cp:
            sec = cp["Rules"]
            core.set_rules({k: sec.getboolean(k) if isinstance(v, bool) else sec.getint(k) for k, v in rules.items() if k in sec})
            return
    core.set_rules(rules)


class Round:
    """1ラウンド内の手番進行を管理するプロキシ（コアロジックをcore.Envに完全委譲）"""

    _ST_MAP = {
        core.GState.DISCARD: "dis",
        core.GState.DISCARD_PICK: "dis_pic",
        core.GState.DRAW: "drw",
        core.GState.DRAW_PICK: "drw_pic",
        core.GState.KOIKOI: "koi",
        core.GState.ROUND_OVER: "over",
        core.GState.GAME_OVER: "over",
    }

    def __init__(self, dlr: Optional[int] = None):
        seed = random.randint(0, 0x7FFFFFFF)
        self.cpp_env = core.Env(seed)
        if dlr is not None:
            self.cpp_env.dlr = dlr
            self.cpp_env.win = dlr
        self.cpp_env.reset_round()
        self.log: Dict[str, Any] = {}

    @property
    def dlr(self) -> int: return self.cpp_env.dlr
    @property
    def t16(self) -> int: return self.cpp_env.t16
    @property
    def st(self) -> str: return self._ST_MAP[self.cpp_env.st]
    @property
    def wait(self) -> bool: return self.cpp_env.wait
    @property
    def ex(self) -> bool: return self.cpp_env.ex
    @property
    def tp(self) -> int: return self.cpp_env.turn_p()
    @property
    def ip(self) -> int: return self.cpp_env.idle_p()
    @property
    def t8(self) -> int: return self.cpp_env.turn8()
    @property
    def is_over(self) -> bool: return self.st == "over"
    @property
    def mgr(self) -> core.StateMgr: return self.cpp_env.mgr

    @property
    def win(self) -> Optional[int]:
        w = self.cpp_env.win
        return None if w == 0 else w

    @property
    def koi(self) -> Dict[int, List[int]]:
        return {1: [self.cpp_env.koi_num(1)], 2: [self.cpp_env.koi_num(2)]}

    @property
    def pts(self) -> Dict[int, Optional[int]]:
        if not self.is_over: return {1: None, 2: None}
        return {1: self.cpp_env.rnd_pt(1), 2: self.cpp_env.rnd_pt(2)}

    @property
    def mask(self) -> np.ndarray:
        t = 0 if self.st == "dis" else (2 if self.st == "koi" else 1)
        return self.mgr.mask(t, self.tp)

    @property
    def hand(self) -> Dict[int, List[int]]:
        return {1: self.cpp_env.get_hand(1), 2: self.cpp_env.get_hand(2)}

    @property
    def pile(self) -> Dict[int, List[int]]:
        return {1: self.cpp_env.get_pile(1), 2: self.cpp_env.get_pile(2)}

    @property
    def f_slots(self) -> List[int]:
        return self.cpp_env.get_field()

    @property
    def f_act(self) -> List[int]:
        return sorted(c for c in self.f_slots if c != -1)

    @property
    def shw(self) -> List[int]:
        return self.cpp_env.get_show()

    @property
    def pairs(self) -> List[int]:
        shw = self.shw
        return [c for c in self.f_act if c // 4 == shw[0] // 4] if shw else []

    def yaku_pt(self, p: int) -> int:
        pile_bb = sum(1 << c for c in self.cpp_env.get_pile(p))
        return core.calc_yaku_pt(pile_bb, self.cpp_env.koi_num(p))

    def reset(self) -> None:
        self.__init__(dlr=self.win)

    def step(self, act: Optional[Union[int, bool, str]]) -> None:
        if self.st == "koi":
            if not self.wait:
                # ★修正点: 役なし自動スキップ時はC++へ「続行パス(-1)」を通知
                c_act = -1
            elif isinstance(act, str):
                c_act = 1 if act.lower() == "koi" else 0
            else:
                c_act = 1 if bool(act) else 0
        else:
            c_act = 0 if act is None else int(act)

        self.log[f"t{self.t16}_{self.st}"] = act
        self.cpp_env.step(c_act)

    def dis(self, c: Optional[int] = None) -> None: self.step(c)
    def pic(self, c: Optional[int] = None) -> None: self.step(c)
    def drw(self) -> None: self.step(None)
    def claim(self, kk: Optional[Union[bool, str]] = None) -> None: self.step(kk)


class Game:
    """こいこいゲーム全体（全ラウンド進行・スコア記録）の統括マシン"""

    def __init__(
        self,
        tot: int = MAX_RND,
        init_pt: Tuple[int, int] = (30, 30),
        dlr: Optional[int] = None,
        names: Tuple[str, str] = ("P1", "P2"),
        rec_dir: str = "",
        save: bool = False,
    ):
        self.tot, self.init_pt, self.init_dlr = tot, init_pt, dlr
        self.names = {1: names[0], 2: names[1]}
        self.rec_dir, self.save = rec_dir, save

        self.rnd, self.pts = 1, {1: init_pt[0], 2: init_pt[1]}
        self.is_over, self.win = False, None
        self.log: Dict[str, Any] = {}
        self.cur = Round(dlr=dlr)
        self._init_log()

    @property
    def feats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rs = self.cur
        tp, ip = rs.tp, rs.ip
        c_np, g_np = rs.mgr.feat(
            rs.st == "koi",
            self.pts[tp] or 0, self.pts[ip] or 0,
            self.rnd, rs.t16, rs.dlr == tp,
            rs.cpp_env.koi_num(tp), rs.cpp_env.koi_num(ip),
            tp, ip, self.tot
        )
        return torch.from_numpy(c_np).float(), torch.from_numpy(g_np).float()

    def reset(self) -> None:
        self.__init__(tot=self.tot, init_pt=self.init_pt, dlr=self.init_dlr, names=(self.names[1], self.names[2]), rec_dir=self.rec_dir, save=self.save)

    def step_rnd(self) -> None:
        if not self.cur.is_over: raise RuntimeError("Round not over")
        p = self.cur.pts
        self.pts[1] += p[1] or 0; self.pts[2] += p[2] or 0
        self._rec_rnd()

        if self.pts[1] <= 0 or self.pts[2] <= 0 or self.rnd >= self.tot:
            self.is_over = True
            self.win = 1 if self.pts[1] > self.pts[2] else (2 if self.pts[2] > self.pts[1] else 0)
            self._rec_game()
        else:
            self.cur.reset()
            self.rnd += 1

    def _init_log(self) -> None:
        t_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.log = {"info": {"start": t_str, "p1": self.names[1], "p2": self.names[2], "pts": self.pts.copy(), "tot_rnd": self.tot}, "res": {}, "rec": {}}

    def _rec_rnd(self) -> None:
        rl = self.cur.log.copy()
        b = rl.setdefault("basic", {}).copy()
        b.update({"p1_rnd_pt": self.cur.pts[1], "p2_rnd_pt": self.cur.pts[2], "dlr": self.cur.dlr})
        rl["basic"] = b
        self.log["rec"][f"rnd{self.rnd}"] = rl

    def _rec_game(self) -> None:
        self.log["info"]["end"] = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.log["res"] = {"over": True, "win": self.win, "final_pts": self.pts.copy()}
        if self.save and self.rec_dir:
            fn = f"{self.rec_dir}{self.log['info']['start']} {self.names[1]} vs {self.names[2]}.json"
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(self.log, f, ensure_ascii=False, indent=2)


class Agent:
    """単一共有モデルによるこいこい推論エージェント"""

    def __init__(self, net: NeuRD):
        self.net = net.eval()

    @torch.inference_mode()
    def act(self, g: Game, test: bool = True) -> Optional[Union[int, str]]:
        rs = g.cur
        if not rs.wait: return None
        st = rs.st

        if test and st == "koi":
            tp = rs.tp
            cur_pt = (g.pts[tp] or 0) + rs.yaku_pt(tp)
            if cur_pt >= 60: return False
            if g.rnd >= MAX_RND: return cur_pt < 30

        cx, gx = [t.unsqueeze(0).to(DEV) for t in g.feats]
        c_pi, _, k_pi, _ = self.net(cx, gx)
        logit = k_pi if st == "koi" else c_pi

        mask = torch.from_numpy(rs.mask).to(DEV)
        logit = logit.masked_fill(~mask, -1e9)
        best = int(torch.argmax(logit, dim=-1).item())
        return ("koi" if best else "stop") if st == "koi" else best


def load_net(path: str) -> NeuRD:
    net = NeuRD().to(DEV)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
    else:
        print(f"[Warn] Model file {path} not found. Using random initialized weights.")
    return net.float().eval()


def handle_p_turn(rs: Round, st: str, wait: bool, win, g: Game):
    """プレイヤー手番のGUI制御とアクション入力"""
    act = None
    if st == "dis":
        win = z_gui.UpdateTurnPlayer(win, g)
        win = z_gui.UpdateCardAndYaku(win, g)
        win, act = z_gui.WaitDiscardGUI(win, g)
        rs.dis(cast(int, act))
    elif st in ("dis_pic", "drw_pic"):
        if st == "drw_pic": win = z_gui.ShowPileCardGUI(win, g)
        if wait: win, act = z_gui.WaitPickGUI(win, g)
        elif st == "drw_pic": win = z_gui.WaitAnyClick(win)
        rs.pic(cast(int, act))
    elif st == "drw":
        win = z_gui.UpdateCardAndYaku(win, g)
        win = z_gui.WaitAnyClick(win)
        rs.drw()
    elif st == "koi":
        win = z_gui.UpdateCardAndYaku(win, g)
        if wait: win, act = z_gui.WaitKoiKoi(win)
        rs.claim(cast(str, act))
    return win


def handle_ai_turn(rs: Round, st: str, wait: bool, win, g: Game, ai: Agent):
    """AI手番のモデル推論とGUIアニメーション同期"""
    act = None
    if st == "dis":
        win = z_gui.UpdateTurnPlayer(win, g)
        win = z_gui.UpdateCardAndYaku(win, g)
        act = ai.act(g)
        rs.dis(cast(int, act))
        win = z_gui.WaitAnyClick(win)
        win = z_gui.UpdateOpDiscardCardGUI(win, g)
    elif st in ("dis_pic", "drw_pic"):
        if st == "drw_pic": win = z_gui.ShowPileCardGUI(win, g)
        act = ai.act(g)
        rs.pic(cast(int, act))
        win = z_gui.WaitAnyClick(win)
    elif st == "drw":
        win = z_gui.UpdateCardAndYaku(win, g)
        rs.drw()
        win = z_gui.WaitAnyClick(win)
    elif st == "koi":
        win = z_gui.UpdateCardAndYaku(win, g)
        act = ai.act(g)
        win = z_gui.ShowOpKoiKoi(win, g, cast(str, act))
        rs.claim(cast(str, act))
    return win


def main():
    init_cfg()
    g = Game(names=("Player", "Com"), rec_dir=REC_DIR, save=True)
    ai = Agent(load_net("model/arena/model.pt"))

    win = z_gui.InitGUI()
    win = z_gui.UpdateGameStatusGUI(win, g)

    while True:
        if g.is_over:
            win = z_gui.ShowGameOverGUI(win, g)
            z_gui.Close(win)
            break

        rs = g.cur
        if rs.st == "over":
            win = z_gui.ShowRoundOverGUI(win, g)
            g.step_rnd()
            win = z_gui.ClearBoardGUI(win)
            win = z_gui.UpdateGameStatusGUI(win, g)
            win = z_gui.UpdateCardAndYaku(win, g)
            continue

        if rs.tp == 1:
            win = handle_p_turn(rs, rs.st, rs.wait, win, g)
        else:
            win = handle_ai_turn(rs, rs.st, rs.wait, win, g, ai)


if __name__ == "__main__":
    main()