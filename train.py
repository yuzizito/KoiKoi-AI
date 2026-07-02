"""
Koi-Koi NeuRD Main Training Pipeline (Concise & Fully Unified)
"""

import io
import os
import sys
import threading
import time
import configparser
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

import core
from neurd import NeuRD, C_CH, G_CH

INIT_MODEL = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "15"
torch.set_num_threads(1)

def apply_koikoi_rules(cfg_path: str = "koikoi.cfg"):
    """koikoi.cfgを読み込んでC++のcoreモジュールにルールを適用する"""
    if not os.path.exists(cfg_path):
        print(f"  | [Rules] '{cfg_path}' が見つかりません。デフォルトのルールを使用します。")
        return

    config = configparser.ConfigParser()
    config.read(cfg_path, encoding='utf-8')

    if 'Rules' not in config:
        print(f"  | [Rules] '{cfg_path}' に [Rules] セクションがありません。デフォルトのルールを使用します。")
        return

    rules = config['Rules']
    rule_dict = {}

    # bool型パラメータのマッピング
    if 'enable_flower_sake' in rules:
        rule_dict['ena_f_sake'] = rules.getboolean('enable_flower_sake')
    if 'enable_moon_sake' in rules:
        rule_dict['ena_m_sake'] = rules.getboolean('enable_moon_sake')

    # int型パラメータのマッピング (キー指定のタイポを修正)
    if 'five_lights' in rules: rule_dict['lt5'] = rules.getint('five_lights')
    if 'four_lights' in rules: rule_dict['lt4'] = rules.getint('four_lights')
    if 'rainy_four_lights' in rules: rule_dict['rainy4'] = rules.getint('rainy_four_lights') # ← ココを修正しました
    if 'three_lights' in rules: rule_dict['lt3'] = rules.getint('three_lights')
    if 'bdb' in rules: rule_dict['bdb'] = rules.getint('bdb')
    if 'flower_moon_base_pt' in rules: rule_dict['fm_base'] = rules.getint('flower_moon_base_pt')
    if 'flower_moon_koikoi_pt' in rules: rule_dict['fm_koi'] = rules.getint('flower_moon_koikoi_pt')
    if 'red_blue_ribbon' in rules: rule_dict['rb_rib'] = rules.getint('red_blue_ribbon')
    if 'red_ribbon' in rules: rule_dict['r_rib'] = rules.getint('red_ribbon')
    if 'blue_ribbon' in rules: rule_dict['b_rib'] = rules.getint('blue_ribbon')

    # 念のため None が混入していないかフィルタリング (Pybind11のキャストエラーを完全に防止)
    rule_dict = {k: v for k, v in rule_dict.items() if v is not None}

    # C++側へ適用
    core.set_rules(rule_dict)
    print(f"  | [Rules] '{cfg_path}' の設定をシミュレータに適用しました。")

@dataclass
class TrainCfg:
    start_lp: int = 0
    max_lps: int = 3000
    koikoi_start_lp: int = 50 # Design Intent: カリキュラム学習の導入。序盤は手札の切り方と取り方の学習に集中させるため、指定ループまでこいこいフェーズを無効化する
    use_importance_sampling: bool = False
    gain: float = 1.414
    lr_max: float = 1e-4 # 学習率：ニューラルネットの重みの更新量
    lr_min: float = 1e-5
    eta: float = 0.03 # ロジットの更新量
    temp: float = 1.0 # 高いほど幅広い手を探索する
    eps_max: float = 0.02
    eps_min: float = 0.01
    noise: float = 0.00 # 出力分布にノイズを不均一に加算する
    max_grad: float = 1.0
    skip_threshold: float = 1.5
    adamw_eps: float = 1e-7 # 1e-6～1e-7 安定していれば1e-7でよい

    g_ix: Dict[str, float] = field(default_factory=lambda: {"discard": 0.01, "pick": 0.005, "koikoi": 0.005})
    v_coef: Dict[str, float] = field(default_factory=lambda: {"discard": 0.1, "pick": 0.01, "koikoi": 0.01})

    n_cpu: int = 1
    lp_games: int = 52
    b_size: int = 256 # ランダム性が強いゲームは大きく
    arena_int: int = 50
    arena_games: int = 500

    mod_dir: str = "model"
    phases: Tuple[str, ...] = ("discard", "pick", "koikoi")
    dev_str: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def dev(self) -> torch.device: return torch.device(self.dev_str)
    @property
    def g_th(self) -> int: return max(1, self.lp_games // self.n_cpu)
    @property
    def cap_d(self) -> int: return 256*13
    @property
    def cap_p(self) -> int: return 256
    @property
    def cap_k(self) -> int: return 256

class VCoefTuner:
    def __init__(self, initial_coefs: dict, window: int = 50): # 過去50バッチ分の移動平均を保持する
        self.window = window
        self.act_gn = {k: [] for k in initial_coefs}
        self.crt_gn = {k: [] for k in initial_coefs}
        self.coef = initial_coefs.copy()

    def record_and_tune(self, ph: str, net: torch.nn.Module) -> float:
        pi_head = cast(nn.Linear, net.card_pi if ph in ("discard", "pick") else net.koi_pi)
        q_head  = cast(nn.Linear, net.card_q if ph in ("discard", "pick") else net.koi_q)

        if pi_head.weight.grad is None or q_head.weight.grad is None:
            return self.coef[ph]

        # Design Intent: backward()によって既に現在のv_coefが掛けられた状態の勾配が計算されているため、スケール適用前の純粋なCriticの勾配ノルムを逆算する
        cur_coef = max(self.coef[ph], 1e-8)
        gn_a = torch.norm(pi_head.weight.grad).item()
        gn_c_unscaled = torch.norm(q_head.weight.grad).item() / cur_coef

        self.act_gn[ph].append(gn_a)
        self.crt_gn[ph].append(gn_c_unscaled)

        if len(self.act_gn[ph]) > self.window:
            self.act_gn[ph].pop(0)
            self.crt_gn[ph].pop(0)

        avg_a = sum(self.act_gn[ph]) / len(self.act_gn[ph])
        avg_c = sum(self.crt_gn[ph]) / len(self.crt_gn[ph])

        if avg_c > 1e-8:
            # Design Intent: ||∇actor|| / (new_coef * ||∇critic_unscaled||) = const となる最適係数の算出
            new_coef = avg_a / avg_c * 0.5
            # Design Intent: まれに発生する勾配消失/爆発による極端な係数の適用を防ぐ安全装置
            self.coef[ph] = max(1e-3, min(1.0, new_coef))

        return self.coef[ph]

def _get_jit_bytes(net: torch.nn.Module, device: torch.device) -> bytes:
    """モデルを指定デバイス上でトレースしてバイト列に変換するヘルパー関数"""
    net = net.to(device)
    dc = torch.zeros((1, C_CH, 48), dtype=torch.float32, device=device)
    dg = torch.zeros((1, G_CH), dtype=torch.float32, device=device)
    tr_net = torch.jit.trace(net, (dc, dg), check_trace=False)
    fr_net = torch.jit.freeze(tr_net)
    if hasattr(torch.jit, "optimize_for_inference"):
        opt_net = torch.jit.optimize_for_inference(fr_net)
    else:
        opt_net = fr_net
    buf = io.BytesIO()
    torch.jit.save(opt_net, buf)
    return buf.getvalue()

def get_opp_bytes(opp_path: str, fallback_net: torch.nn.Module, dev: torch.device) -> bytes:
    net = NeuRD().cpu()
    if os.path.exists(opp_path):
        net.load_state_dict(torch.load(opp_path, map_location="cpu", weights_only=True))
    else:
        net.load_state_dict(fallback_net.state_dict())
        print(f"  | [Arena] 対戦相手が見つかりません。現在の自身と対戦します。")
    net.eval()
    return _get_jit_bytes(net, dev)

def run_arena(net: torch.nn.Module, cfg: 'TrainCfg', lp: int, opp_b: bytes):
    t0 = time.perf_counter()
    is_trn = net.training
    net.eval()
    p1_b = _get_jit_bytes(net, cfg.dev)
    if is_trn: net.train()
    
    res = core.run_arena(cfg.arena_games, p1_b, opp_b, cfg.dev_str, 6)

    sc = int((res.get('win', 0)*2 + res.get('draw', 0))/4)
    print(f"■ Arena  Score: {sc:>3}  (Time: {time.perf_counter() - t0:.2f}s)")
    print("   LP | dis_A dis_C | pic_A pic_C | koi_A koi_C |   ER   GN  Ent   Qs   As")

# --------------------------------------------------------------------------------

# ER  有効ランク：ネットワークが獲得した特徴表現の多様性 一桁まで下がったら要注意
# GN  勾配ノルム：ネットの重み更新時のクリップ前の勾配の大きさ max_grad=3に張り付いたら注意 0は学習の死
# Ent 方策エントロピー：モデルの探索度合い 早い段階で急落したら早期収束の懸念あり
# Qs  価値予測の標準偏差：Critic（価値ネットワーク）が予測したOld values（状態価値）の標準偏差 0ならモード崩壊
# As  アドバンテージの標準偏差：報酬とモデルの予想価値の誤差の標準偏差 Criticの精度が上がるほど値が下がる 0ならバグ

def log_health_step(pipe: Any, lp: int, dt: float, smp: dict, a: dict, c: dict, rd: list):
    """【独立ロガー】平時1行＋検知時「Block1 MLP 内部演算の完全分解＆発散起点特定レポート」"""
    import math

    if not hasattr(pipe, "_er_cache") or lp % 40 == 0:
        c_s = rd[0]["discard"]["card_states"][:128].to(pipe.cfg.dev)
        g_s = rd[0]["discard"]["global_states"][:128].to(pipe.cfg.dev)
        if c_s.size(0) > 0:
            with torch.no_grad():
                h = pipe.net.net(c_s, g_s)[:, 1:]
                sv = torch.linalg.svdvals(h.reshape(-1, h.size(-1)).float())
                p = sv[sv > 1e-6] / sv[sv > 1e-6].sum()
                pipe._er_cache = torch.exp(-(p * torch.log(p)).sum()).item()
        else: pipe._er_cache = 0.0
    er = getattr(pipe, "_er_cache", 0.0)

    culprit_ph = max(pipe.trn.keys(), key=lambda p: getattr(pipe.trn[p], "last_gn", 0.0))
    gn = getattr(pipe.trn[culprit_ph], "last_gn", 0.0)

    d_l = [d["discard"]["old_logits"] for d in rd if "discard" in d and d["discard"]["old_logits"].numel()>0]
    d_v = [d["discard"]["old_values"] for d in rd if "discard" in d and d["discard"]["old_values"].numel()>0]
    d_r = [d["discard"]["rewards"]    for d in rd if "discard" in d and d["discard"]["rewards"].numel()>0]
    if d_l:
        cl, cv, cr = torch.cat(d_l, 0).float(), torch.cat(d_v, 0).float(), torch.cat(d_r, 0).float()
        prb = F.softmax(cl, dim=-1)
        ent = -(prb * F.log_softmax(cl, dim=-1)).sum(-1).mean().item()
        qs, as_ = cv.std().item(), (cr - cv).std().item()
    else: ent, qs, as_ = 0.0, 0.0, 0.0

    if lp == 0:
        print(f"time {dt:4.1f}   sample dis {smp.get('discard',0):05d}  pic {smp.get('pick',0):05d}  koi {smp.get('koikoi',0):05d}")
        print("   LP | dis_A dis_C | pic_A pic_C | koi_A koi_C |   ER   GN  Ent   Qs   As")

    da = abs(a.get("discard", 0.0)); pa = abs(a.get("pick", 0.0)); ka = abs(a.get("koikoi", 0.0));
    dc = abs(c.get("discard", 0.0)); pc = abs(c.get("pick", 0.0)); kc = abs(c.get("koikoi", 0.0));
    print(f"{lp:05d} | {da:5.3f} {dc:5.3f} | {pa:5.3f} {pc:5.3f} | {ka:5.3f} {kc:5.3f} | {er:4.1f} {gn:4.2f} {ent:4.2f} {qs:4.2f} {as_:4.2f}")

# --------------------------------------------------------------------------------

class Trainer:
    def __init__(self, net: nn.Module, opt: torch.optim.Optimizer, ph: str, cfg: TrainCfg, tuner: VCoefTuner):
        self.net, self.opt, self.ph, self.cfg = net, opt, ph, cfg
        self.tuner = tuner
        self.coef = tuner.coef[ph]

    def step(self, cur_l, cur_q, old_l, adv, ret, act, mask, old_pi, old_mu, old_v):
        dim = cur_l.size(-1)
        m_sl = mask[:, :dim].bool()
        oh = F.one_hot(act, num_classes=dim).to(cur_l.dtype)
        
        # --- NeuRD理論に基づくImportance Sampling補正 ---
        # Importance Sampling Ratio ρ = π(a) / μ(a) とクリッピング (NeuRD論文準拠)
        rho = (old_pi / old_mu.clamp(min=1e-7)).clamp(max=10.0)
        th = self.cfg.g_ix[self.ph]
        is_w = rho / old_pi.clamp(min=(th if th > 0.0 else 1e-7))
        if not self.cfg.use_importance_sampling:
            is_w = torch.ones_like(is_w)

        dy = torch.where(m_sl, self.cfg.eta * (adv * is_w).unsqueeze(-1) * oh, torch.zeros_like(cur_l))
        act_l = -(dy.detach() * cur_l).sum(dim=-1).mean()
        crt_l = F.smooth_l1_loss(cur_q.gather(1, act.view(-1, 1)).view(-1), ret.view(-1), beta=1.0)
        return act_l + self.coef * crt_l, act_l.item(), crt_l.item()

    def train_ep(self, b: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
        self.net.train()
        c_st, g_st = b["card_states"], b["global_states"]
        N = c_st.size(0)
        if N < self.cfg.b_size: return 0.0, 0.0, 0.0
        
        tot, al, cl, nb = 0.0, 0.0, 0.0, 0
        self.last_gn = 0.0
        ix = torch.randperm(N, device=c_st.device)
        
        for i in range(0, N - self.cfg.b_size + 1, self.cfg.b_size):
            ii = ix[i : i + self.cfg.b_size]
            cp, cq, kp, kq = self.net(c_st[ii], g_st[ii])
            clg, cvl = (cp, cq) if self.ph in ("discard", "pick") else (kp, kq)
            l, a, c = self.step(clg, cvl, b["old_logits"][ii], b["adv"][ii], b["rewards"][ii], b["actions"][ii], b["legal_masks"][ii], b["old_pi"][ii], b["old_mu"][ii], b["old_values"][ii])
            self.opt.zero_grad(set_to_none=True); l.backward()
            self.coef = self.tuner.record_and_tune(self.ph, self.net)
            gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.cfg.max_grad)
            if gn.item() > self.last_gn:
                self.last_gn = gn.item()
            if gn.item() > self.cfg.skip_threshold:
                print(f"  [Warn] Gradient Skipped in {self.ph} (GN: {gn.item():.2f} > {self.cfg.skip_threshold})")
                self.opt.zero_grad(set_to_none=True) # 勾配を破棄してステップを進めない
                continue
            self.opt.step(); tot += l.item(); al += a; cl += c; nb += 1
        return tot / nb, al / nb, cl / nb

class Pipeline:
    def __init__(self, cfg: TrainCfg):
        self.cfg, self.stop = cfg, threading.Event()
        os.makedirs(cfg.mod_dir, exist_ok=True)
        apply_koikoi_rules("koikoi.cfg")
        self._load_wpm()
        self._init_net(); self._listen()
        
    def _load_wpm(self):
        import pickle
        import numpy as np
        if not os.path.exists('win_prob_mat.pkl'):
            raise FileNotFoundError("win_prob_mat.pkl が見つかりません。先に生成スクリプトを実行してください。")
        with open('win_prob_mat.pkl', 'rb') as f:
            self.wpm = pickle.load(f).astype(np.float32)

    def _init_net(self):
        dc = torch.zeros((1, C_CH, 48), dtype=torch.float32, device=self.cfg.dev)
        dg = torch.zeros((1, G_CH), dtype=torch.float32, device=self.cfg.dev)

        self.net = NeuRD().cpu()
        if INIT_MODEL:
            for nm, md in self.net.named_modules():
                if "self_attn" in nm or "attn" in nm:
                    for w in [getattr(md, k, None) for k in ("in_proj_weight", "q_proj_weight", "k_proj_weight")]:
                        if isinstance(w, torch.Tensor): nn.init.orthogonal_(w, gain=self.cfg.gain)

        pt = os.path.join(self.cfg.mod_dir, "model.pt")
        if not INIT_MODEL and os.path.exists(pt): self.net.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))

        self.net.to(self.cfg.dev).float()
        self.tr_net = torch.jit.trace(self.net, (dc, dg), check_trace=False)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr_max, eps=self.cfg.adamw_eps, weight_decay=0.0)
        self.v_tuner = VCoefTuner(self.cfg.v_coef, window=100)
        self.trn = {p: Trainer(self.net, opt, p, self.cfg, self.v_tuner) for p in self.cfg.phases}

    def _listen(self):
        threading.Thread(target=lambda: [self.stop.set() for line in sys.stdin if line.strip().lower() == "q"], daemon=True).start()

    def _sim(self, force_stop_koikoi=False):
        return core.run_sim(self.cfg.n_cpu, self.cfg.g_th, self.cfg.g_th, self.cfg.cap_d, self.cfg.cap_p,
            self.cfg.cap_k, self._get_model_bytes(), self.cfg.dev_str, core.MAX_RND, self.cfg.temp,
            getattr(self, "current_eps", self.cfg.eps_max), self.cfg.noise, force_stop_koikoi, self.wpm)

    def _unpack(self, rd, active_phases):
        smp, al, cl = {}, {}, {}
        keys = ("card_states", "global_states", "actions", "rewards", "old_logits", "old_values", "legal_masks", "old_pi", "old_mu")
        for ph in self.cfg.phases:
            # Design Intent: 対象フェーズに含まれていない（50ループ未満のこいこい等）場合は学習をスキップする
            if ph not in active_phases:
                smp[ph], al[ph], cl[ph] = 0, 0.0, 0.0
                continue
                
            bufs = {k: [d[ph][k] for d in rd if isinstance(d, dict) and ph in d and d[ph].get("card_states") is not None and d[ph]["card_states"].numel() > 0] for k in keys}
            if bufs["card_states"]:
                b = {k: torch.cat(v, dim=0).to(self.cfg.dev, non_blocking=True) for k, v in bufs.items()}
                adv = b["rewards"] - b["old_values"]
                b["adv"] = adv
                _, a, c = self.trn[ph].train_ep(b)
                smp[ph], al[ph], cl[ph] = b["card_states"].size(0), a, c
            else: smp[ph], al[ph], cl[ph] = 0, 0.0, 0.0
        return smp, al, cl

    def run(self):
        print("学習開始 (Press 'q' + Enter to stop)")
        
        # Design Intent: アリーナの対戦相手モデルを初期化時に一度だけシリアライズし保持する
        self.opp_b = get_opp_bytes("model/arena/model.pt", self.net, self.cfg.dev)
        
        # 初回のデータ収集
        initial_force_stop = self.cfg.start_lp < self.cfg.koikoi_start_lp
        self._update_decay_params(self.cfg.start_lp)
        rd = self._sim(force_stop_koikoi=initial_force_stop)
        
        for lp in range(self.cfg.start_lp, self.cfg.max_lps):
            # 学習フェーズの切り替え判定
            force_stop = lp < self.cfg.koikoi_start_lp
            active_phases = ("discard", "pick") if force_stop else ("discard", "pick", "koikoi")

            self._update_decay_params(lp)
            t0 = time.perf_counter()
            
            smp, a, c = self._unpack(rd, active_phases)
            core.reload_model(self._get_model_bytes())
            
            # 次のループのシミュレーション生成用
            next_force_stop = (lp + 1) < self.cfg.koikoi_start_lp
            rd = self._sim(force_stop_koikoi=next_force_stop)
            
            log_health_step(self, lp, time.perf_counter() - t0, smp, a, c, rd)
            
            # -------------------------------------------------------------------------- arena
            if lp % self.cfg.arena_int == 0:
                run_arena(self.net, self.cfg, lp, self.opp_b)
                # Design Intent: 初回ロード後は空のバイト列を渡し、C++側のキャッシュを利用させる
                # 同時にPython側のメモリも解放し、プロセス間転送コストを最小化する
                self.opp_b = b""
            # --------------------------------------------------------------------------------
            if self.stop.is_set(): break
            
        torch.save(self.net.state_dict(), os.path.join(self.cfg.mod_dir, "model.pt"))
        core.close_sim(); print("終了しました。")
        
    def _get_model_bytes(self) -> bytes:
        dc = torch.zeros((1, C_CH, 48), dtype=torch.float32, device=self.cfg.dev)
        dg = torch.zeros((1, G_CH), dtype=torch.float32, device=self.cfg.dev)
        
        # トレース時のみ評価モード(Dropout無効化等)に切り替え、最新の重みでトレースし直す
        self.net.eval()
        tr_net = torch.jit.trace(self.net, (dc, dg), check_trace=False)
        self.net.train() # 学習モードに復帰
        
        buf = io.BytesIO()
        torch.jit.save(tr_net, buf)
        return buf.getvalue()
    
    def _update_decay_params(self, lp: int):
        # Design Intent: 指数関数的減衰(Exponential Decay)による学習率の動的更新
        if self.cfg.lr_max <= 0 or self.cfg.lr_min <= 0 or self.cfg.eps_max <= 0 or self.cfg.eps_min <= 0:
            raise ValueError("Max and min values must be strictly positive for exponential decay.")
        if self.cfg.max_lps <= 0:
            raise ValueError("max_lps must be strictly positive.")
        
        progress = min(1.0, lp / self.cfg.max_lps)
        
        current_lr = self.cfg.lr_max * ((self.cfg.lr_min / self.cfg.lr_max) ** progress)
        for param_group in next(iter(self.trn.values())).opt.param_groups:
            param_group['lr'] = current_lr
            
        self.current_eps = self.cfg.eps_max * ((self.cfg.eps_min / self.cfg.eps_max) ** progress)


if __name__ == "__main__":
    Pipeline(TrainCfg()).run()