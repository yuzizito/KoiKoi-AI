"""
Koi-Koi NeuRD Main Training Pipeline (Concise & Fully Unified)
"""

import io
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import core
from neurd import NeuRD, C_CH, G_CH

INIT_MODEL = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "15"
torch.set_num_threads(1)

@dataclass
class TrainCfg:
    start_lp: int = 0
    max_lps: int = 2000
    gain: float = 1.414
    lr_max: float = 3e-6 # 学習率
    lr_min: float = 1e-7
    eta: float = 0.01
    gamma: float = 0.995
    gae_lam: float = 0.97
    temp: float = 2.0 # 高いほど幅広い手を探索する
    eps_max: float = 0.05
    eps_min: float = 0.01
    noise: float = 0.00 # 出力分布にノイズを不均一に加算する
    max_grad: float = 3
    weight_decay: float = 1e-3 # 標準（中型モデル） 1e-2、小型モデル　1e-3～1e-4
    adamw_eps: float = 1e-6 # 1e-6～1e-8

    g_ix: Dict[str, float] = field(default_factory=lambda: {"discard": 0.02, "pick": 0.01, "koikoi": 0.01})
    v_coef: Dict[str, float] = field(default_factory=lambda: {"discard": 0.1, "pick": 0.01, "koikoi": 0.01})

    n_cpu: int = 1
    lp_games: int = 512
    b_size: int = 512
    sync_int: int = 1
    arena_int: int = 20
    arena_games: int = 500

    mod_dir: str = "model"
    phases: Tuple[str, ...] = ("discard", "pick", "koikoi")
    dev_str: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def dev(self) -> torch.device: return torch.device(self.dev_str)
    @property
    def g_th(self) -> int: return max(1, self.lp_games // self.n_cpu)
    @property
    def cap_d(self) -> int: return self.g_th * 128
    @property
    def cap_p(self) -> int: return self.g_th * 12
    @property
    def cap_k(self) -> int: return self.g_th * 12

class VCoefTuner:
    def __init__(self, initial_coefs: dict, window: int = 50): # 過去50バッチ分の移動平均を保持する
        self.window = window
        self.act_gn = {k: [] for k in initial_coefs}
        self.crt_gn = {k: [] for k in initial_coefs}
        self.coef = initial_coefs.copy()

    def record_and_tune(self, ph: str, net: torch.nn.Module) -> float:
        pi_head = net.card_pi if ph in ("discard", "pick") else net.koi_pi
        q_head  = net.card_q if ph in ("discard", "pick") else net.koi_q

        if pi_head.weight.grad is None or q_head.weight.grad is None:
            return self.coef[ph]

        # Design Intent: backward()によって既に現在のv_coefが掛けられた状態の勾配が計算されているため、スケール適用前の純粋なCriticの勾配ノルムを逆算する
        cur_coef = max(self.coef[ph], 1e-8)
        gn_a = pi_head.weight.grad.norm().item()
        gn_c_unscaled = q_head.weight.grad.norm().item() / cur_coef

        self.act_gn[ph].append(gn_a)
        self.crt_gn[ph].append(gn_c_unscaled)

        if len(self.act_gn[ph]) > self.window:
            self.act_gn[ph].pop(0)
            self.crt_gn[ph].pop(0)

        avg_a = sum(self.act_gn[ph]) / len(self.act_gn[ph])
        avg_c = sum(self.crt_gn[ph]) / len(self.crt_gn[ph])

        if avg_c > 1e-8:
            # Design Intent: ||∇actor|| / (new_coef * ||∇critic_unscaled||) = 1.0 となる最適係数の算出
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
    buf = io.BytesIO()
    torch.jit.save(tr_net, buf)
    return buf.getvalue()

def load_arena_opponent_bytes(opponent_path: str, fallback_state_dict: dict, device: torch.device) -> bytes:
    """対戦相手のモデルを読み込み、シリアライズされたバイト列を返す"""
    net = NeuRD().cpu()
    
    if os.path.exists(opponent_path):
        net.load_state_dict(torch.load(opponent_path, map_location="cpu", weights_only=True))
    else:
        net.load_state_dict(fallback_state_dict)
        print(f"  | [Arena] 対戦相手が見つかりません。現在の自身と対戦します。")
    
    net.eval()
    
    return _get_jit_bytes(net, device)

def run_arena_test(net_state_dict: dict, cfg: 'TrainCfg', lp: int):
    """別スレッドで動作し、学習ループを止めずにC++側のアリーナテストを実行する関数"""
    t0 = time.perf_counter()
    
    p1_net = NeuRD().cpu()
    p1_net.load_state_dict(net_state_dict)
    p1_net.eval()
    
    p1_bytes = _get_jit_bytes(p1_net, cfg.dev)
    p2_bytes = load_arena_opponent_bytes("model/arena/model.pt", net_state_dict, cfg.dev)
    
    res = core.run_arena(cfg.arena_games, p1_bytes, p2_bytes, cfg.dev_str, 6)

    win = res.get('win', 0)
    lose = res.get('lose', 0)
    draw = res.get('draw', 0)
    score = int((win*2 + draw)/5)
    
    print(f"■ Arena  Win: {win:>4}  Lose: {lose:>4}  Draw: {draw:>4}  Score: {score:>3}")

# --------------------------------------------------------------------------------

# ER  有効ランク：ネットワークが獲得した特徴表現の多様性　一桁まで下がったら要注意
# GN  勾配ノルム：ネットの重み更新時のクリップ前の勾配の大きさ　max_grad=3に張り付いたら注意 0は学習の死
# Ent 方策エントロピー：モデルの探索度合い　早い段階で急落したら早期収束の懸念あり
# Qs  価値予測の標準偏差：Critic（価値ネットワーク）が予測したOld values（状態価値）の標準偏差　0ならモード崩壊
# As  アドバンテージの標準偏差：報酬とモデルの予想価値の誤差の標準偏差　Criticの精度が上がるほど値が下がる　0ならバグ

def log_health_step(pipe: any, lp: int, dt: float, smp: dict, a: dict, c: dict, rd: list):
    """【独立ロガー】平時1行＋検知時「Block1 MLP 内部演算の完全分解＆発散起点特定レポート」"""
    import math

    if not hasattr(pipe, "_er_cache") or lp % 100 == 0:
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

    def fmt_e(v: float) -> str:
        if v == 0.0 or v != v or abs(v) < 1e-9: return " 0e+0"
        m, e = f"{v:.0e}".split('e')
        return f"{int(float(m)):>2d}e{max(-9, min(9, int(e))):+d}"

    def get_ar(ph: str):
        act, crt = a.get(ph, 0.0), c.get(ph, 0.0)
        return fmt_e(act), fmt_e(act / crt if crt != 0 else 0.0)

    if lp == 0:
        print(f"time {dt:4.1f}   sample dis {smp.get('discard',0):05d}  pic {smp.get('pick',0):05d}  koi {smp.get('koikoi',0):05d}")
    
    if lp % 30 == 0:
        print("   LP | dis_A dis_R | pic_A pic_R | koi_A koi_R |   ER   GN  Ent   Qs   As")

    dA, dR = get_ar("discard"); pA, pR = get_ar("pick"); kA, kR = get_ar("koikoi")
    print(f"{lp:05d} | {dA} {dR} | {pA} {pR} | {kA} {kR} | {er:4.1f} {gn:4.2f} {ent:4.2f} {qs:4.2f} {as_:4.2f}")

    for ph, trn in pipe.trn.items():
        d = getattr(trn, "heavy_dump", None)
        if d:
            def _print_mlp(title, data):
                ln2, l1, gl, l2, mlp = data['ln2'], data['lin1'], data['gelu'], data['lin2'], data['mlp']
                print(f"  +--- [{title}] ----------------------------------------------------------------+")
                print(f"  | 【1. LN2 出力 (MLP入力直前)】")
                print(f"  |  ・ 平均ノルム : {ln2['n']:.4f} | 分散 : {ln2['v']:.4f} | 有効ランク : {ln2['er']:.2f} | PC1寄与 : {ln2['pc1']:.1f}%")
                print(f"  | 【2. Linear1 出力 (次元拡張)】")
                print(f"  |  ・ 平均ノルム : {l1['n']:.4f} | 平均活性 : {l1['mean']:+.4f} (Max: {l1['max']:+.4f})")
                print(f"  |  ・ 重みノルム : {l1['wn']:.4f} | 【入出力ゲイン】 : {l1['gain']:.2f}倍")
                print(f"  | 【3. GELU 活性関数通過後 (非線形変換)】")
                print(f"  |  ・ 平均活性   : {gl['mean']:+.4f} (Max: {gl['max']:+.4f}) | ゼロ近傍(<1e-4)割合 : {gl['z_pct']:.1f}%")
                print(f"  |  ・ 【通過前後ゲイン】 : {gl['gain']:.2f}倍")
                print(f"  | 【4. Linear2 出力 (次元縮小)】")
                print(f"  |  ・ 平均ノルム : {l2['n']:.4f} | 重みノルム : {l2['wn']:.4f} | 【入出力ゲイン】 : {l2['gain']:.2f}倍")
                print(f"  | 【5. MLP 最終寄与 (Residual Stream加算＆Logit投影)】")
                print(f"  |  ・ 残差Add1に対するノルム比 : {mlp['res_ratio']:.2f}倍")
                print(f"  |  ・ 最終 Policy Logit 寄与率 : {mlp['share']:.1f}%")
                print(f"  +--------------------------------------------------------------------------------------------------+")

            print(f"  +=== [Block1 MLP 内部演算の完全分解＆発散起点特定 (GN: {d['gn']:.4f})] (起因ヘッド: {ph}) ===+")
            _print_mlp(f"異常値サンプル (ID: {d['id_ab']})", d['ab'])
            _print_mlp(f"正常値サンプル (ID: {d['id_nm']})", d['nm'])

            ab, nm = d['ab'], d['nm']
            # 増幅率の乖離度（異常/正常）を計算
            ratio_ln2 = ab['ln2']['n'] / max(1e-9, nm['ln2']['n'])
            ratio_l1  = ab['lin1']['gain'] / max(1e-9, nm['lin1']['gain'])
            ratio_gl  = ab['gelu']['gain'] / max(1e-9, nm['gelu']['gain'])
            ratio_l2  = ab['lin2']['gain'] / max(1e-9, nm['lin2']['gain'])

            print(f"  | [★システム自動診断：異常サンプルのMLP内部で『最初に』発散させた演算はどれか？]")
            
            # 判断ロジック: 正常時と比較して最も異常にゲインが跳ね上がっている部分を真犯人とする
            max_ratio = max(ratio_ln2, ratio_l1, ratio_gl, ratio_l2)
            
            if ratio_ln2 > 2.0 and ratio_ln2 == max_ratio:
                print(f"  |  <!> 【真犯人：入力(LN2)異常】")
                print(f"  |      MLP内部の重みは正常ですが、入力時点で既に正常サンプルの {ratio_ln2:.2f}倍 のノルムが流れ込んでいます。")
                print(f"  |      直前の残差結合(Add1)か、Block0からの蓄積ダメージが原因です。")
            elif ratio_l1 > 1.5 and ratio_l1 == max_ratio:
                print(f"  |  <!> 【真犯人：Linear1の重み発散】")
                print(f"  |      入力は正常ですが、Linear1を通過した瞬間にゲインが正常時の {ratio_l1:.2f}倍 に跳ね上がっています。")
                print(f"  |      処方箋: self.net.net.tr.layers[1].linear1 の重み初期化スケールを縮小するか、Weight Decayを強めてください。")
            elif ratio_gl > 1.5 and ratio_gl == max_ratio:
                print(f"  |  <!> 【真犯人：GELUによる非線形爆発 (GELU死の逆)】")
                print(f"  |      Linear1出力までは正常ですが、GELUを通過した瞬間に正の極大値が指数関数的に増幅されています。")
                print(f"  |      異常サンプルの特定の特徴だけがGELUの急勾配領域に乗り上げてバズーカを放っています。")
            elif ratio_l2 > 1.5 and ratio_l2 == max_ratio:
                print(f"  |  <!> 【真犯人：Linear2の重み発散】")
                print(f"  |      GELU通過までは正常に耐えましたが、最後の次元縮小行列(Linear2)が異常なゲイン(正常比{ratio_l2:.2f}倍)を生んでいます。")
                print(f"  |      処方箋: self.net.net.tr.layers[1].linear2 の重み初期化スケールを縮小してください。")
            else:
                print(f"  |  <!> 【連鎖的・微小増幅の蓄積】特定の1箇所の爆発ではなく、全段で1.1〜1.2倍ずつ異常増幅が掛け合わされています。")
                print(f"  |      処方箋: MLP内にDropout(0.1)を追加するか、LayerNormのepsを大きくしてスケールを抑え込んでください。")
            print(f"  +==================================================================================================+")
            trn.heavy_dump = None

# --------------------------------------------------------------------------------

class Trainer:
    def __init__(self, net: nn.Module, opt: torch.optim.Optimizer, ph: str, cfg: TrainCfg, tuner: VCoefTuner):
        self.net, self.opt, self.ph, self.cfg = net, opt, ph, cfg
        self.tuner = tuner
        self.coef = tuner.coef[ph]

    def step(self, cur_l, cur_q, old_l, adv, ret, act, mask, old_p, old_v):
        dim = cur_l.size(-1)
        m_sl = mask[:, :dim].bool()
        oh = F.one_hot(act, num_classes=dim).to(cur_l.dtype)
        th = self.cfg.g_ix[self.ph]
        is_w = 1.0 / old_p.clamp(min=(th if th > 0.0 else 1e-7))

        dy = torch.where(m_sl, self.cfg.eta * (adv * is_w).unsqueeze(-1) * oh, torch.zeros_like(cur_l))
        act_l = -(dy.detach() * cur_l).sum(dim=-1).mean()
        crt_l = F.smooth_l1_loss(cur_q.gather(1, act.view(-1, 1)).view(-1), ret.view(-1), beta=1.0)
        return act_l + self.coef * crt_l, act_l.item(), crt_l.item()

    def train_ep(self, b: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
        self.net.train()
        c_st, g_st = b["card_states"], b["global_states"]
        N = c_st.size(0)
        if N == 0: return 0.0, 0.0, 0.0
        tot, al, cl, nb = 0.0, 0.0, 0.0, 0
        self.last_gn = 0.0
        ix = torch.randperm(N, device=c_st.device)
        for i in range(0, N, self.cfg.b_size):
            ii = ix[i : i + self.cfg.b_size]
            cp, cq, kp, kq = self.net(c_st[ii], g_st[ii])
            clg, cvl = (cp, cq) if self.ph in ("discard", "pick") else (kp, kq)
            l, a, c = self.step(clg, cvl, b["old_logits"][ii], b["norm_adv"][ii], b["rewards"][ii], b["actions"][ii], b["legal_masks"][ii], b["old_probs"][ii], b["old_values"][ii])
            self.opt.zero_grad(set_to_none=True); l.backward()
            self.coef = self.tuner.record_and_tune(self.ph, self.net)
            gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.cfg.max_grad)
            if gn.item() > self.last_gn:
                self.last_gn = gn.item()
            self.opt.step(); tot += l.item(); al += a; cl += c; nb += 1
        return tot / nb, al / nb, cl / nb

class Pipeline:
    def __init__(self, cfg: TrainCfg):
        self.cfg, self.stop = cfg, threading.Event()
        os.makedirs(cfg.mod_dir, exist_ok=True)
        self._init_net(); self._listen()

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
        
        decay_params = []
        no_decay_params = []
        
        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue
            if "linear1.weight" in name or "linear2.weight" in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
                
        # Optimizerの定義 (AdamW)
        opt = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.cfg.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.cfg.lr_max, eps=self.cfg.adamw_eps)
        
        self.v_tuner = VCoefTuner(self.cfg.v_coef, window=100)
        self.trn = {p: Trainer(self.net, opt, p, self.cfg, self.v_tuner) for p in self.cfg.phases}

    def _listen(self):
        threading.Thread(target=lambda: [self.stop.set() for line in sys.stdin if line.strip().lower() == "q"], daemon=True).start()

    def _sim(self):
        return core.run_sim(self.cfg.n_cpu, self.cfg.g_th, self.cfg.g_th, self.cfg.cap_d, self.cfg.cap_p,
            self.cfg.cap_k, self.cfg.gamma, self._get_model_bytes(), self.cfg.dev_str, core.MAX_RND, self.cfg.temp,
            getattr(self, "current_eps", self.cfg.eps_max), self.cfg.noise, self.cfg.gae_lam)

    def _unpack(self, rd):
        smp, al, cl = {}, {}, {}
        keys = ("card_states", "global_states", "actions", "rewards", "old_logits", "old_values", "legal_masks", "old_action_probs")
        for ph in self.cfg.phases:
            bufs = {k: [d[ph][k] for d in rd if isinstance(d, dict) and ph in d and d[ph].get("card_states") is not None and d[ph]["card_states"].numel() > 0] for k in keys}
            if bufs["card_states"]:
                b = {("old_probs" if k == "old_action_probs" else k): torch.cat(v, dim=0).to(self.cfg.dev, non_blocking=True) for k, v in bufs.items()}
                adv = b["rewards"] - b["old_values"]
                b["norm_adv"] = (adv - adv.mean()) / adv.std().clamp(min=1e-6) if adv.numel() > 1 else adv
                _, a, c = self.trn[ph].train_ep(b)
                smp[ph], al[ph], cl[ph] = b["card_states"].size(0), a, c
            else: smp[ph], al[ph], cl[ph] = 0, 0.0, 0.0
        return smp, al, cl

    def run(self):
        print("学習開始 (Press 'q' + Enter to stop)")
        self._update_decay_params(self.cfg.start_lp)
        rd = self._sim()
        for lp in range(self.cfg.start_lp, self.cfg.max_lps):
            self._update_decay_params(lp)
            t0 = time.perf_counter()
            smp, a, c = self._unpack(rd)
            if lp % self.cfg.sync_int == 0:
                core.reload_model(self._get_model_bytes())
            rd = self._sim()
            log_health_step(self, lp, time.perf_counter() - t0, smp, a, c, rd)  # probe
            
            # -------------------------------------------------------------------------- arena
            if lp % self.cfg.arena_int == 0:
                state_copy = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                threading.Thread(
                    target=run_arena_test, 
                    args=(state_copy, self.cfg, lp),
                    daemon=True
                ).start()
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