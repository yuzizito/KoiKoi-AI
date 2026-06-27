import os
import time
import tkinter as tk
from tkinter import ttk
import threading
import cv2
import numpy as np
from PIL import Image, ImageGrab, ImageTk

# AI関連モジュール
import torch
import koikoicore
from koikoinet_v2 import NeuRDModel

# ==========================================
# 設定・仕様定義
# ==========================================
ROI_FIELD = (505, 650, 920, 880)
ROI_HAND_1P = (450, 900, 960, 1100)
ROI_TAKE_1P = (960, 810, 1220, 1043)
ROI_TAKE_2P = (960, 480, 1220, 730)
ROI_ROUND = (1135, 735, 1150, 755)
ROI_SCORE_1P = (560, 895, 585, 915)
ROI_DRAWN = (440, 700, 505, 810)
ROI_KOIKOI = (710, 855, 750, 875)

PATH_NUM = "./png/num"
PATH_NORMAL = "./png/normal"
PATH_SMALL = "./png/small"
PATH_KOI_IMG = "./png/koi.png"

class HanafudaScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("花札こいこい 画面解析＆AI推奨ツール")
        self.root.geometry("480x570")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.templates_round = {}
        self.templates_score = {}
        self.templates_normal = {}
        self.templates_small = {}
        self.template_koi = None
        
        self.is_running = True
        self.photo_refs = [] 
        
        # --- 状態管理・トラッキング用 ---
        self.prev_hand = set()
        self.prev_seen = set()
        self.prev_field = set()
        
        self.my_discard_history = set()
        self.op_played_suits = set()
        self.op_ignored_suits = set()
        
        self.current_dealer = 1
        self.dealer_determined = False
        self.last_op_discard_time = 0
        
        # Design Intent: 人間側からの明示的なアクション介入フラグ
        self.is_stock_mode = False 
        
        self._init_ai_model()
        self._create_widgets()
        self._load_all_templates()
        
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()

    def _init_ai_model(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.discard_model = NeuRDModel(is_koikoi=False).to(self.device)
        self.pick_model = NeuRDModel(is_koikoi=False).to(self.device)
        self.koikoi_model = NeuRDModel(is_koikoi=True).to(self.device)
        
        try:
            self.discard_model.load_state_dict(torch.load('model_v2/discard.pt', map_location=self.device, weights_only=True))
            self.pick_model.load_state_dict(torch.load('model_v2/pick.pt', map_location=self.device, weights_only=True))
            self.koikoi_model.load_state_dict(torch.load('model_v2/koikoi.pt', map_location=self.device, weights_only=True))
            
            for m in [self.discard_model, self.pick_model, self.koikoi_model]:
                m.eval()
            print("[AI] All models loaded successfully.")
        except Exception as e:
            print(f"[AI] Failed to load models: {e}")

    def _create_widgets(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_dealer = ttk.Label(main_frame, text="現在の親: 1P (自動判定)", font=("Helvetica", 10, "bold"), foreground="green")
        self.lbl_dealer.pack(anchor=tk.W, pady=(0, 5))
        
        # Design Intent: 背景色を動的に変更して視覚的フィードバックを与えるため標準tk.Buttonを使用
        self.btn_stock = tk.Button(
            main_frame,
            text="▼ 山札をめくった (Pick推論を実行)",
            font=("Helvetica", 11, "bold"),
            bg="#e0e0e0",
            relief="raised",
            pady=6,
            command=self._toggle_stock_mode
        )
        self.btn_stock.pack(fill=tk.X, pady=(0, 8))
        
        self.lbl_best_action = ttk.Label(main_frame, text="推奨手: --", font=("Helvetica", 13, "bold"), foreground="blue")
        self.lbl_best_action.pack(anchor=tk.W, pady=(5, 0))
        
        self.frm_best_imgs = ttk.Frame(main_frame)
        self.frm_best_imgs.pack(anchor=tk.W, pady=(5, 10))
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=5)
        
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
        self.btn_close = ttk.Button(btn_frame, text="終了", command=self.on_closing)
        self.btn_close.pack(side=tk.RIGHT)

    def _toggle_stock_mode(self):
        self.is_stock_mode = not self.is_stock_mode
        self._refresh_stock_btn_ui()

    def _refresh_stock_btn_ui(self):
        if self.is_stock_mode:
            self.btn_stock.config(text="▲ 【山札モード中】 (クリックで手札モードへ戻る)", bg="#ffb74d")
        else:
            self.btn_stock.config(text="▼ 山札をめくった (Pick推論を実行)", bg="#e0e0e0")

    def _load_templates_from_dir(self, dir_path, prefix_digits=2, max_val=47, start_val=0):
        templates = {}
        if not os.path.exists(dir_path): return templates
        for i in range(start_val, max_val + 1):
            full_path = os.path.join(dir_path, f"{str(i).zfill(prefix_digits)}.png")
            if os.path.exists(full_path):
                img = cv2.imread(full_path, cv2.IMREAD_COLOR)
                if img is not None: templates[i] = img
        return templates

    def _load_all_templates(self):
        self.templates_round = self._load_templates_from_dir(PATH_NUM, 2, 6, 1)
        self.templates_score = {k - 10: v for k, v in self._load_templates_from_dir(PATH_NUM, 2, 19, 10).items()}
        self.templates_normal = self._load_templates_from_dir(PATH_NORMAL, 2, 47, 0)
        self.templates_small = self._load_templates_from_dir(PATH_SMALL, 2, 47, 0)
        if os.path.exists(PATH_KOI_IMG):
            self.template_koi = cv2.imread(PATH_KOI_IMG, cv2.IMREAD_COLOR)

    def _match_cards(self, roi_img, template_dict, threshold=0.92):
        if roi_img is None: return []
        all_candidates = []
        min_dist = 25 
        for card_id, t_img in template_dict.items():
            if roi_img.shape[0] < t_img.shape[0] or roi_img.shape[1] < t_img.shape[1]: continue
            if roi_img.shape[2] != t_img.shape[2]: continue
            res = cv2.matchTemplate(roi_img, t_img, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                all_candidates.append({"card_id": card_id, "pt": pt, "score": res[pt[1], pt[0]]})
                
        all_candidates = sorted(all_candidates, key=lambda x: x["score"], reverse=True)
        confirmed_matches, used_ids = [], set()
        
        for cand in all_candidates:
            pt, c_id = cand["pt"], cand["card_id"]
            if c_id in used_ids: continue
            is_overlapping = any(np.sqrt((pt[0] - c["pt"][0])**2 + (pt[1] - c["pt"][1])**2) < min_dist for c in confirmed_matches)
            if not is_overlapping:
                confirmed_matches.append(cand)
                used_ids.add(c_id)
        return sorted([item["card_id"] for item in confirmed_matches])

    def _match_single_number(self, roi_img, template_dict, threshold=0.88):
        best_val, max_score = -1, -1
        for num, t_img in template_dict.items():
            if roi_img is None or t_img is None or roi_img.shape[0] < t_img.shape[0] or roi_img.shape[1] < t_img.shape[1]: continue
            res = cv2.matchTemplate(roi_img, t_img, cv2.TM_CCOEFF_NORMED)
            _, max_val_res, _, _ = cv2.minMaxLoc(res)
            if max_val_res > threshold and max_val_res > max_score: max_score, best_val = max_val_res, num
        return best_val if best_val != -1 else "--"

    def _match_score_2digits(self, roi_img, template_dict, threshold=0.88):
        if roi_img is None or len(roi_img.shape) < 3: return "--"
        h, w, c = roi_img.shape
        tens = self._match_single_number(roi_img[:, :w // 2, :], template_dict, threshold)
        ones = self._match_single_number(roi_img[:, w // 2:, :], template_dict, threshold)
        if tens == "--" and ones == "--": return "--"
        return int((str(tens) if tens != "--" else "0") + (str(ones) if ones != "--" else "0"))

    def _card_to_str(self, card_id):
        return f"{card_id // 4 + 1}月-{card_id % 4 + 1}" if 0 <= card_id < 48 else "--"

    def _update_tracking(self, hand, field, take1p, take2p, drawn):
        curr_hand = set(hand)
        curr_field = set(field)
        curr_take1p = set(take1p)
        curr_take2p = set(take2p)
        curr_drawn = set(drawn)
        
        curr_seen = curr_hand | curr_field | curr_take1p | curr_take2p | curr_drawn
        new_cards = curr_seen - self.prev_seen
        missing_hand = self.prev_hand - curr_hand
        
        is_round_start = False
        
        if len(curr_hand) == 8 and len(curr_take1p) == 0 and len(curr_take2p) == 0:
            if len(self.prev_hand) != 8:
                self.current_dealer = 1
                self.dealer_determined = False
                self.my_discard_history.clear()
                self.op_played_suits.clear()
                self.op_ignored_suits.clear()
                self.is_stock_mode = False # ラウンド初期化で安全解除
                self._refresh_stock_btn_ui()
                is_round_start = True 
        
        if not self.dealer_determined and not is_round_start and len(self.prev_seen) > 0:
            if len(curr_hand) < 8:
                self.current_dealer = 1
                self.dealer_determined = True
            elif len(new_cards) > 0 and len(missing_hand) == 0 and len(curr_drawn) == 0:
                self.current_dealer = 2
                self.dealer_determined = True

        for c in missing_hand:
            self.my_discard_history.add(c)
            
        # Design Intent: 1Pが手札を捨てた瞬間＝次のターンのDiscardフェーズに入ったため、山札モードをスマート解除する
        if len(missing_hand) > 0 and self.is_stock_mode:
            self.is_stock_mode = False
            self._refresh_stock_btn_ui()
            
        now = time.time()
        if not is_round_start and len(missing_hand) == 0 and len(curr_drawn) == 0:
            field_new = curr_field - self.prev_field
            for c in field_new:
                if c in new_cards:
                    if now - self.last_op_discard_time > 2.0:
                        suit = c // 4
                        self.op_played_suits.add(suit)
                        self.last_op_discard_time = now
                        for fc in self.prev_field:
                            if fc // 4 != suit: self.op_ignored_suits.add(fc // 4)
                                
        self.prev_hand = curr_hand
        self.prev_seen = curr_seen
        self.prev_field = curr_field

    def _infer_best_action(self, rnd, score, hand, field, take1p, take2p, drawn, is_koikoi_phase):
        if not hand and not drawn and not is_koikoi_phase and not self.is_stock_mode:
            return "アクションなし", []
            
        raw_score = int(score) if str(score).isdigit() else 0
        p1_score = max(1, min(59, raw_score - 50))
        p2_score = 60 - p1_score
        
        turn = 9 - len(hand) if len(hand) > 0 else 1
        turn_16 = turn * 2 - 1 if self.current_dealer == 1 else turn * 2
        round_num = int(rnd) if str(rnd).isdigit() else 1
        
        def to_bb(cards): return sum(1 << c for c in cards if isinstance(c, int) and 0 <= c < 48)
        def to_suit_mask(suits): return sum(1 << s for s in suits)
            
        hand1_bb, field_bb, pile1_bb, pile2_bb = to_bb(hand), to_bb(field), to_bb(take1p), to_bb(take2p)
        my_disc_bb = to_bb(self.my_discard_history)
        op_played_mask = to_suit_mask(self.op_played_suits)
        op_ignored_mask = to_suit_mask(self.op_ignored_suits)
        
        def get_feat_tensor(h1_bb_val, f_bb_val, p1_bb_val, p2_bb_val, st_bb_val, is_kk=False):
            cpp_state = koikoicore.KoiKoiStateManager()
            cpp_state.set_state_from_vision(h1_bb_val, 0, f_bb_val, st_bb_val, p1_bb_val, p2_bb_val, my_disc_bb, op_played_mask, op_ignored_mask)
            feat_np = cpp_state.get_feature(is_kk, p1_score, p2_score, round_num, turn_16, self.current_dealer, 0, 0, 1, 2, 6)
            return torch.from_numpy(feat_np).unsqueeze(0).float().to(self.device)

        seen_bb = hand1_bb | field_bb | pile1_bb | pile2_bb
        if drawn: seen_bb |= (1 << drawn[0])
        stock_bb = ((1 << 48) - 1) ^ seen_bb
        
        # 1. こいこい推論
        if is_koikoi_phase:
            feat_tensor = get_feat_tensor(hand1_bb, field_bb, pile1_bb, pile2_bb, stock_bb, is_kk=True)
            with torch.inference_mode():
                logits = self.koikoi_model(feat_tensor)
                logits_np = (logits[0] if isinstance(logits, tuple) else logits).squeeze(0).cpu().numpy()
            return ("推奨手(判断): こいこいする", []) if logits_np[1] > logits_np[0] else ("推奨手(判断): 勝負(ストップ)", [])

        # 2. 山札(Pick)フェーズ【手動ボタン優先】
        if self.is_stock_mode:
            if not drawn:
                return "山札の絵柄を認識できませんでした", []
                
            drawn_card = drawn[0]
            drawn_suit = drawn_card // 4
            pairs = [c for c in field if c // 4 == drawn_suit]
            stock_bb = ((1 << 48) - 1) ^ (seen_bb | (1 << drawn_card))
            
            if len(pairs) == 2:
                feat_tensor = get_feat_tensor(hand1_bb, field_bb, pile1_bb, pile2_bb, stock_bb)
                with torch.inference_mode():
                    logits = self.pick_model(feat_tensor)
                    logits_np = (logits[0] if isinstance(logits, tuple) else logits).squeeze(0).cpu().numpy()
                best_idx = max(pairs, key=lambda idx: logits_np[idx])
                return f"推奨手(取る): {self._card_to_str(best_idx)}", [best_idx]
            elif len(pairs) == 1:
                return f"自動取得: {self._card_to_str(pairs[0])}", [pairs[0]]
            elif len(pairs) == 3:
                return f"自動取得(総取り)", pairs
            else:
                return "場に出る", [drawn_card]
                
        # 3. 手札(Discard)フェーズ
        if not hand: return "手札なし", []
        feat_tensor = get_feat_tensor(hand1_bb, field_bb, pile1_bb, pile2_bb, stock_bb)
        with torch.inference_mode():
            logits = self.discard_model(feat_tensor)
            logits_np = (logits[0] if isinstance(logits, tuple) else logits).squeeze(0).cpu().numpy()
            
        best_discard_idx = max(hand, key=lambda idx: logits_np[idx])
        discard_suit = best_discard_idx // 4
        pairs = [c for c in field if c // 4 == discard_suit]
        
        if len(pairs) == 2:
            new_hand1_bb = hand1_bb & ~(1 << best_discard_idx)
            pick_feat_tensor = get_feat_tensor(new_hand1_bb, field_bb, pile1_bb, pile2_bb, stock_bb)
            with torch.inference_mode():
                pick_logits = self.pick_model(pick_feat_tensor)
                pick_logits_np = (pick_logits[0] if isinstance(pick_logits, tuple) else pick_logits).squeeze(0).cpu().numpy()
            best_pick_idx = max(pairs, key=lambda idx: pick_logits_np[idx])
            return f"推奨手: 捨={self._card_to_str(best_discard_idx)} / 取={self._card_to_str(best_pick_idx)}", [best_discard_idx, best_pick_idx]
        else:
            return f"推奨手(捨てる): {self._card_to_str(best_discard_idx)}", [best_discard_idx]

    def _update_best_action_ui(self, action_text, card_ids):
        self.lbl_best_action.config(text=action_text)
        for widget in self.frm_best_imgs.winfo_children(): widget.destroy()
        self.photo_refs.clear()
        for c_id in card_ids:
            path = f"{PATH_NORMAL}/{c_id:02d}.png"
            if os.path.exists(path):
                img = Image.open(path).resize((45, 70))
                photo = ImageTk.PhotoImage(img)
                self.photo_refs.append(photo)
                lbl = ttk.Label(self.frm_best_imgs, image=photo)
                lbl.pack(side=tk.LEFT, padx=3)

    def _scan_loop(self):
        while self.is_running:
            try:
                screen_color = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
                crop_roi = lambda roi: screen_color[roi[1]:roi[3], roi[0]:roi[2], :]

                current_round = self._match_single_number(crop_roi(ROI_ROUND), self.templates_round)
                current_score = self._match_score_2digits(crop_roi(ROI_SCORE_1P), self.templates_score)
                
                cards_field = self._match_cards(crop_roi(ROI_FIELD), self.templates_normal, threshold=0.90)
                cards_hand = self._match_cards(crop_roi(ROI_HAND_1P), self.templates_normal, threshold=0.90)
                
                # Design Intent: 山札モードがONの時だけ、ROI内の画像を正規札(00~47)と強制マッチングさせる
                cards_drawn = []
                if self.is_stock_mode:
                    cards_drawn = self._match_cards(crop_roi(ROI_DRAWN), self.templates_normal, threshold=0.85)
                
                cards_take1p = self._match_cards(crop_roi(ROI_TAKE_1P), self.templates_small, threshold=0.88)
                cards_take2p = self._match_cards(crop_roi(ROI_TAKE_2P), self.templates_small, threshold=0.88)

                if len(cards_take1p) % 2 != 0: cards_take1p = self._match_cards(crop_roi(ROI_TAKE_1P), self.templates_small, threshold=0.82)
                if len(cards_take2p) % 2 != 0: cards_take2p = self._match_cards(crop_roi(ROI_TAKE_2P), self.templates_small, threshold=0.82)

                is_koikoi_phase = False
                if self.template_koi is not None:
                    res = cv2.matchTemplate(crop_roi(ROI_KOIKOI), self.template_koi, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > 0.80: is_koikoi_phase = True

                self.root.after(0, self._update_gui_labels, 
                                current_round, current_score, cards_hand, 
                                cards_field, cards_take1p, cards_take2p, cards_drawn, is_koikoi_phase)

            except Exception as e:
                print(f"[ループ内エラー] 解析に失敗しました: {e}")
            
            time.sleep(0.1)

    def _update_gui_labels(self, rnd, score, hand, field, take1p, take2p, drawn, is_koikoi_phase):
        self._update_tracking(hand, field, take1p, take2p, drawn)
        
        self.lbl_dealer.config(text=f"現在の親: {'1P' if self.current_dealer == 1 else '2P'} (自動判定)")
        self.lbl_round.config(text=f"ラウンド数: {rnd}")
        self.lbl_score.config(text=f"1P側得点: {score}")
        self.lbl_hand.config(text=f"1P手札: {hand}")
        self.lbl_field.config(text=f"場札: {field}")
        
        drawn_str = f"{drawn} (手動解析中)" if self.is_stock_mode else "(手札モード中)"
        self.lbl_drawn.config(text=f"めくられた山札: {drawn_str}")
        
        self.lbl_take1p.config(text=f"1P取札: {take1p}")
        self.lbl_take2p.config(text=f"2P取札: {take2p}")
        
        try:
            action_text, best_ids = self._infer_best_action(rnd, score, hand, field, take1p, take2p, drawn, is_koikoi_phase)
            self._update_best_action_ui(action_text, best_ids)
        except Exception as e:
            self.lbl_best_action.config(text="推奨手: 解析エラー")
            print(f"[推論エラー]: {e}")

    def on_closing(self):
        self.is_running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HanafudaScannerApp(root)
    root.mainloop()