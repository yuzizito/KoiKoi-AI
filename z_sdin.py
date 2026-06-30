# -*- coding: utf-8 -*-
"""
Hanafuda Screen Scanner & AI Recommendation Tool
Refactored with RoiConfig dataclass, strict matching, and clean AI dispatch.
"""

import os
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageGrab, ImageTk

import koikoicore
from neurd import NeuRD


@dataclass(frozen=True)
class RoiConfig:
    """画面キャプチャ対象領域 (left, top, right, bottom) の一元管理"""
    field: Tuple[int, int, int, int] = (505, 650, 920, 880)
    hand_1p: Tuple[int, int, int, int] = (450, 900, 960, 1100)
    take_1p: Tuple[int, int, int, int] = (960, 810, 1220, 1043)
    take_2p: Tuple[int, int, int, int] = (960, 480, 1220, 730)
    round_num: Tuple[int, int, int, int] = (1135, 735, 1150, 755)
    score_1p: Tuple[int, int, int, int] = (560, 895, 585, 915)
    drawn_card: Tuple[int, int, int, int] = (440, 700, 505, 810)
    koikoi_lbl: Tuple[int, int, int, int] = (710, 855, 750, 875)


ROI = RoiConfig()
PATH_NUM = "./png/num"
PATH_NORMAL = "./png/normal"
PATH_SMALL = "./png/small"
PATH_KOI_IMG = "./png/koi.png"


class HanafudaScannerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("花札こいこい 画面解析＆AI推奨ツール")
        self.root.geometry("480x570")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.is_running = True
        self.photo_refs: List[ImageTk.PhotoImage] = []

        # --- 状態トラッキング ---
        self.prev_hand: Set[int] = set()
        self.prev_seen: Set[int] = set()
        self.prev_field: Set[int] = set()
        self.my_discard_history: Set[int] = set()
        self.op_played_suits: Set[int] = set()
        self.op_ignored_suits: Set[int] = set()

        self.current_dealer = 1
        self.dealer_determined = False
        self.last_op_discard_time = 0.0
        self.is_stock_mode = False

        self._init_ai_models()
        self._create_widgets()
        self._load_all_templates()

        threading.Thread(target=self._scan_loop, daemon=True).start()

    def _init_ai_models(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.discard_model = NeuRD(is_koikoi=False).to(self.device)
        self.pick_model = NeuRD(is_koikoi=False).to(self.device)
        self.koikoi_model = NeuRD(is_koikoi=True).to(self.device)

        try:
            self.discard_model.load_state_dict(torch.load("model_v2/discard.pt", map_location=self.device, weights_only=True))
            self.pick_model.load_state_dict(torch.load("model_v2/pick.pt", map_location=self.device, weights_only=True))
            self.koikoi_model.load_state_dict(torch.load("model_v2/koikoi.pt", map_location=self.device, weights_only=True))
            for m in (self.discard_model, self.pick_model, self.koikoi_model):
                m.eval()
            print("[AI] Models loaded successfully.")
        except Exception as e:
            print(f"[AI] Model loading skipped or failed: {e}")

    def _create_widgets(self) -> None:
        ttk.Style().theme_use("clam")
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.lbl_dealer = ttk.Label(main_frame, text="現在の親: 1P (自動判定)", font=("Helvetica", 10, "bold"), foreground="green")
        self.lbl_dealer.pack(anchor=tk.W, pady=(0, 5))

        self.btn_stock = tk.Button(
            main_frame, text="▼ 山札をめくった (Pick推論を実行)", font=("Helvetica", 11, "bold"),
            bg="#e0e0e0", relief="raised", pady=6, command=self._toggle_stock_mode
        )
        self.btn_stock.pack(fill=tk.X, pady=(0, 8))

        self.lbl_best_action = ttk.Label(main_frame, text="推奨手: --", font=("Helvetica", 13, "bold"), foreground="blue")
        self.lbl_best_action.pack(anchor=tk.W, pady=(5, 0))

        self.frm_best_imgs = ttk.Frame(main_frame)
        self.frm_best_imgs.pack(anchor=tk.W, pady=(5, 10))

        ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=5)

        self.lbl_round = ttk.Label(main_frame, text="ラウンド数: --", font=("Helvetica", 10, "bold"))
        self.lbl_round.pack(anchor=tk.W, pady=1)
        self.lbl_score = ttk.Label(main_frame, text="1P側得点: --", font=("Helvetica", 10, "bold"))
        self.lbl_score.pack(anchor=tk.W, pady=1)
        self.lbl_hand = ttk.Label(main_frame, text="1P手札: --", font=("Helvetica", 9))
        self.lbl_hand.pack(anchor=tk.W, pady=1)
        self.lbl_field = ttk.Label(main_frame, text="場札: --", font=("Helvetica", 9))
        self.lbl_field.pack(anchor=tk.W, pady=1)
        self.lbl_drawn = ttk.Label(main_frame, text="めくられた山札: (手札モード中)", font=("Helvetica", 9))
        self.lbl_drawn.pack(anchor=tk.W, pady=1)
        self.lbl_take1p = ttk.Label(main_frame, text="1P取札: --", font=("Helvetica", 9))
        self.lbl_take1p.pack(anchor=tk.W, pady=1)
        self.lbl_take2p = ttk.Label(main_frame, text="2P取札: --", font=("Helvetica", 9))
        self.lbl_take2p.pack(anchor=tk.W, pady=1)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        ttk.Button(btn_frame, text="終了", command=self._on_closing).pack(side=tk.RIGHT)

    def _toggle_stock_mode(self) -> None:
        self.is_stock_mode = not self.is_stock_mode
        self._refresh_stock_btn_ui()

    def _refresh_stock_btn_ui(self) -> None:
        if self.is_stock_mode:
            self.btn_stock.config(text="▲ 【山札モード中】 (クリックで手札モードへ戻る)", bg="#ffb74d")
        else:
            self.btn_stock.config(text="▼ 山札をめくった (Pick推論を実行)", bg="#e0e0e0")

    def _load_templates(self, dir_path: str, digits: int, max_val: int, start_val: int = 0) -> Dict[int, np.ndarray]:
        res = {}
        if not os.path.exists(dir_path): return res
        for i in range(start_val, max_val + 1):
            path = os.path.join(dir_path, f"{str(i).zfill(digits)}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None: res[i] = img
        return res

    def _load_all_templates(self) -> None:
        self.templates_round = self._load_templates(PATH_NUM, 2, 6, 1)
        self.templates_score = {k - 10: v for k, v in self._load_templates(PATH_NUM, 2, 19, 10).items()}
        self.templates_normal = self._load_templates(PATH_NORMAL, 2, 47, 0)
        self.templates_small = self._load_templates(PATH_SMALL, 2, 47, 0)
        self.template_koi = cv2.imread(PATH_KOI_IMG, cv2.IMREAD_COLOR) if os.path.exists(PATH_KOI_IMG) else None

    def _match_cards(self, roi_img: np.ndarray, t_dict: Dict[int, np.ndarray], threshold: float = 0.92) -> List[int]:
        if roi_img is None or not t_dict: return []
        cands = []
        for c_id, t_img in t_dict.items():
            if roi_img.shape[0] < t_img.shape[0] or roi_img.shape[1] < t_img.shape[1]: continue
            res = cv2.matchTemplate(roi_img, t_img, cv2.TM_CCOEFF_NORMED)
            for pt in zip(*np.where(res >= threshold)[::-1]):
                cands.append((res[pt[1], pt[0]], c_id, pt))

        cands.sort(key=lambda x: x[0], reverse=True)
        confirmed, used_ids = [], set()
        for score, c_id, pt in cands:
            if c_id in used_ids: continue
            if not any(np.hypot(pt[0] - cp[0], pt[1] - cp[1]) < 25 for cp in confirmed):
                confirmed.append(pt)
                used_ids.add(c_id)
        return sorted(list(used_ids))

    def _match_single_digit(self, roi_img: np.ndarray, t_dict: Dict[int, np.ndarray], threshold: float = 0.88) -> Union[int, str]:
        best_val, max_score = "--", threshold
        for num, t_img in t_dict.items():
            if roi_img is None or roi_img.shape[0] < t_img.shape[0] or roi_img.shape[1] < t_img.shape[1]: continue
            _, max_val_res, _, _ = cv2.minMaxLoc(cv2.matchTemplate(roi_img, t_img, cv2.TM_CCOEFF_NORMED))
            if max_val_res > max_score: max_score, best_val = max_val_res, num
        return best_val

    def _match_score(self, roi_img: np.ndarray, t_dict: Dict[int, np.ndarray]) -> Union[int, str]:
        if roi_img is None or roi_img.ndim < 3: return "--"
        w = roi_img.shape[1]
        t = self._match_single_digit(roi_img[:, :w // 2], t_dict)
        o = self._match_single_digit(roi_img[:, w // 2:], t_dict)
        if t == "--" and o == "--": return "--"
        return int(f"{t if t != '--' else 0}{o if o != '--' else 0}")

    @staticmethod
    def _card_str(c_id: int) -> str:
        return f"{c_id // 4 + 1}月-{c_id % 4 + 1}" if 0 <= c_id < 48 else "--"

    def _update_tracking(self, hand: List[int], field: List[int], t1: List[int], t2: List[int], drawn: List[int]) -> None:
        c_hand, c_field, c_t1, c_t2, c_drawn = set(hand), set(field), set(t1), set(t2), set(drawn)
        c_seen = c_hand | c_field | c_t1 | c_t2 | c_drawn
        new_cards = c_seen - self.prev_seen
        missing_hand = self.prev_hand - c_hand

        if len(c_hand) == 8 and not c_t1 and not c_t2 and len(self.prev_hand) != 8:
            self.current_dealer = 1
            self.dealer_determined = False
            self.my_discard_history.clear()
            self.op_played_suits.clear()
            self.op_ignored_suits.clear()
            self.is_stock_mode = False
            self._refresh_stock_btn_ui()

        if not self.dealer_determined and self.prev_seen:
            if len(c_hand) < 8: self.current_dealer, self.dealer_determined = 1, True
            elif new_cards and not missing_hand and not c_drawn: self.current_dealer, self.dealer_determined = 2, True

        self.my_discard_history.update(missing_hand)
        if missing_hand and self.is_stock_mode:
            self.is_stock_mode = False
            self._refresh_stock_btn_ui()

        now = time.time()
        if not missing_hand and not c_drawn:
            for c in (c_field - self.prev_field) & new_cards:
                if now - self.last_op_discard_time > 2.0:
                    suit = c // 4
                    self.op_played_suits.add(suit)
                    self.last_op_discard_time = now
                    self.op_ignored_suits.update({fc // 4 for fc in self.prev_field if fc // 4 != suit})

        self.prev_hand, self.prev_seen, self.prev_field = c_hand, c_seen, c_field

    def _infer_best_action(self, rnd: Union[int, str], score: Union[int, str], hand: List[int], field: List[int], t1: List[int], t2: List[int], drawn: List[int], is_kk: bool) -> Tuple[str, List[int]]:
        if not hand and not drawn and not is_kk and not self.is_stock_mode: return "アクションなし", []

        p1_pt = max(1, min(59, int(score) - 50 if str(score).isdigit() else 0))
        t_16 = (9 - len(hand) if hand else 1) * 2 - (1 if self.current_dealer == 1 else 0)
        r_num = int(rnd) if str(rnd).isdigit() else 1

        to_bb = lambda cards: sum(1 << c for c in cards if 0 <= c < 48)
        to_mask = lambda suits: sum(1 << s for s in suits)

        h1_bb, f_bb, p1_bb, p2_bb = to_bb(hand), to_bb(field), to_bb(t1), to_bb(t2)
        seen_bb = h1_bb | f_bb | p1_bb | p2_bb | (1 << drawn[0] if drawn else 0)
        st_bb = ((1 << 48) - 1) ^ seen_bb

        def get_tensor(h_val: int, f_val: int, kk_flag: bool = False) -> torch.Tensor:
            cpp_st = koikoicore.KoiKoiStateManager()
            cpp_st.set_state_from_vision(h_val, 0, f_val, st_bb, p1_bb, p2_bb, to_bb(self.my_discard_history), to_mask(self.op_played_suits), to_mask(self.op_ignored_suits))
            np_feat = cpp_st.get_feature(kk_flag, p1_pt, 60 - p1_pt, r_num, t_16, self.current_dealer, 0, 0, 1, 2, 6)
            return torch.from_numpy(np_feat).unsqueeze(0).float().to(self.device)

        if is_kk:
            with torch.inference_mode():
                out = self.koikoi_model(get_tensor(h1_bb, f_bb, True))
                np_l = (out[0] if isinstance(out, tuple) else out).squeeze(0).cpu().numpy()
            return ("推奨手(判断): こいこいする", []) if np_l[1] > np_l[0] else ("推奨手(判断): 勝負(ストップ)", [])

        if self.is_stock_mode:
            if not drawn: return "山札の絵柄を認識できませんでした", []
            d_card = drawn[0]
            pairs = [c for c in field if c // 4 == d_card // 4]
            if len(pairs) == 2:
                with torch.inference_mode():
                    out = self.pick_model(get_tensor(h1_bb, f_bb))
                    np_l = (out[0] if isinstance(out, tuple) else out).squeeze(0).cpu().numpy()
                best = max(pairs, key=lambda idx: np_l[idx])
                return f"推奨手(取る): {self._card_str(best)}", [best]
            elif len(pairs) == 1: return f"自動取得: {self._card_str(pairs[0])}", [pairs[0]]
            elif len(pairs) == 3: return "自動取得(総取り)", pairs
            else: return "場に出る", [d_card]

        if not hand: return "手札なし", []
        with torch.inference_mode():
            out = self.discard_model(get_tensor(h1_bb, f_bb))
            np_l = (out[0] if isinstance(out, tuple) else out).squeeze(0).cpu().numpy()

        best_d = max(hand, key=lambda idx: np_l[idx])
        pairs = [c for c in field if c // 4 == best_d // 4]
        if len(pairs) == 2:
            with torch.inference_mode():
                p_out = self.pick_model(get_tensor(h1_bb & ~(1 << best_d), f_bb))
                p_np = (p_out[0] if isinstance(p_out, tuple) else p_out).squeeze(0).cpu().numpy()
            best_p = max(pairs, key=lambda idx: p_np[idx])
            return f"推奨手: 捨={self._card_str(best_d)} / 取={self._card_str(best_p)}", [best_d, best_p]
        return f"推奨手(捨てる): {self._card_str(best_d)}", [best_d]

    def _update_ui(self, text: str, c_ids: List[int]) -> None:
        self.lbl_best_action.config(text=text)
        for w in self.frm_best_imgs.winfo_children(): w.destroy()
        self.photo_refs.clear()
        for cid in c_ids:
            path = f"{PATH_NORMAL}/{cid:02d}.png"
            if os.path.exists(path):
                photo = ImageTk.PhotoImage(Image.open(path).resize((45, 70)))
                self.photo_refs.append(photo)
                ttk.Label(self.frm_best_imgs, image=photo).pack(side=tk.LEFT, padx=3)

    def _scan_loop(self) -> None:
        while self.is_running:
            try:
                screen = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
                crop = lambda r: screen[r[1]:r[3], r[0]:r[2]]

                rnd = self._match_single_digit(crop(ROI.round_num), self.templates_round)
                scr = self._match_score(crop(ROI.score_1p), self.templates_score)
                f_cards = self._match_cards(crop(ROI.field), self.templates_normal, 0.90)
                h_cards = self._match_cards(crop(ROI.hand_1p), self.templates_normal, 0.90)
                d_cards = self._match_cards(crop(ROI.drawn_card), self.templates_normal, 0.85) if self.is_stock_mode else []
                t1 = self._match_cards(crop(ROI.take_1p), self.templates_small, 0.88)
                t2 = self._match_cards(crop(ROI.take_2p), self.templates_small, 0.88)

                is_kk = False
                if self.template_koi is not None:
                    _, max_v, _, _ = cv2.minMaxLoc(cv2.matchTemplate(crop(ROI.koikoi_lbl), self.template_koi, cv2.TM_CCOEFF_NORMED))
                    is_kk = max_v > 0.80

                self.root.after(0, self._on_scan_result, rnd, scr, h_cards, f_cards, t1, t2, d_cards, is_kk)
            except Exception as e:
                print(f"[Scanner Error] {e}")
            time.sleep(0.1)

    def _on_scan_result(self, rnd, scr, hand, field, t1, t2, drawn, is_kk) -> None:
        self._update_tracking(hand, field, t1, t2, drawn)
        self.lbl_dealer.config(text=f"現在の親: {'1P' if self.current_dealer == 1 else '2P'} (自動判定)")
        self.lbl_round.config(text=f"ラウンド数: {rnd}")
        self.lbl_score.config(text=f"1P側得点: {scr}")
        self.lbl_hand.config(text=f"1P手札: {hand}")
        self.lbl_field.config(text=f"場札: {field}")
        self.lbl_drawn.config(text=f"めくられた山札: {drawn if self.is_stock_mode else '(手札モード中)'}")
        self.lbl_take1p.config(text=f"1P取札: {t1}")
        self.lbl_take2p.config(text=f"2P取札: {t2}")

        try:
            txt, cids = self._infer_best_action(rnd, scr, hand, field, t1, t2, drawn, is_kk)
            self._update_ui(txt, cids)
        except Exception as e:
            self.lbl_best_action.config(text="推奨手: 解析エラー")
            print(f"[Inference UI Error] {e}")

    def _on_closing(self) -> None:
        self.is_running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    HanafudaScannerApp(root)
    root.mainloop()