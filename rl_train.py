# author: shguan3
# -*- coding: utf-8 -*-
"""
Koi-Koi NeuRD (Neural Replicator Dynamics) Main Training Pipeline
Pure PyTorch 2.x implementation: zero legacy patches, strict type safety, and clean orchestration.
"""

import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import koikoicore
from koikoigame import MAX_ROUND
from koikoinet_v2 import NeuRDModel

# =============================================================================
# 実行環境・OpenMPスレッド制御（最高速化設定）
# =============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "15"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def time_str() -> str:
    return time.strftime("%y%m%d %H%M%S", time.localtime())


def print_log(msg: str, log_path: str = "log.txt") -> None:
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")


@dataclass
class TrainingConfig:
    """学習ハイパーパラメータおよびシミュレーションスケールの一元管理"""
    start_loop: int = 0
    max_loops: int = 10000
    lr: float = 1e-5
    eta: float = 0.02

    gamma: float = 0.995
    gae_lambda: float = 0.97
    temperature: float = 0.7
    epsilon: float = 0.02
    uniform_noise_rate: float = 0.02

    value_coef: Dict[str, float] = field(default_factory=lambda: {
        "discard": 0.1,
        "pick": 0.01,
        "koikoi": 0.01,
    })

    cpu_count: int = 1
    loop_games: int = 512
    batch_size: int = 8192
    num_epochs: int = 3

    sync_model_interval: int = 1
    arena_test_interval: int = 100
    arena_test_games: int = 500

    model_dir: str = "model_v2"
    log_path: str = "log.txt"
    phases: Tuple[str, ...] = ("discard", "pick", "koikoi")

    device_str: str = field(
        default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_str)

    @property
    def games_per_thread(self) -> int:
        return max(1, self.loop_games // self.cpu_count)

    @property
    def cap_d(self) -> int: return self.games_per_thread * 128
    @property
    def cap_p(self) -> int: return self.games_per_thread * 12
    @property
    def cap_k(self) -> int: return self.games_per_thread * 12


class NeuRDTrainer:
    """各フェーズ（捨・取・こいこい）ごとのNeuRD更新ロジックをカプセル化する"""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, phase: str, config: TrainingConfig):
        self.model = model
        self.optimizer = optimizer
        self.phase = phase
        self.config = config
        self.coef = config.value_coef[phase]

    @torch.no_grad()
    def _compute_rollout_target(
        self, old_logits: torch.Tensor, norm_adv: torch.Tensor,
        actions: torch.Tensor, legal_masks: torch.Tensor, old_probs: torch.Tensor
    ) -> torch.Tensor:
        sampled_mu = old_probs.clamp(min=1e-4)
        b_idx = torch.arange(old_logits.size(0), device=old_logits.device)

        target_logits = old_logits.clone()
        target_logits[b_idx, actions] += self.config.eta * (norm_adv / sampled_mu)

        num_legals = legal_masks.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_legal = (target_logits * legal_masks).sum(dim=-1, keepdim=True) / num_legals
        return torch.where(legal_masks, target_logits - mean_legal, target_logits)

    def train_step(
        self, curr_logits: torch.Tensor, curr_values: torch.Tensor,
        old_logits: torch.Tensor, norm_adv: torch.Tensor, raw_returns: torch.Tensor,
        actions: torch.Tensor, legal_masks: torch.Tensor, old_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        act_dim = curr_logits.size(-1)
        masks_sl = legal_masks[:, :act_dim].bool()
        old_logits_sl = old_logits[:, :act_dim]

        target_centered = self._compute_rollout_target(old_logits_sl, norm_adv, actions, masks_sl, old_probs)

        num_legals = masks_sl.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        curr_mean = (curr_logits * masks_sl).sum(dim=-1, keepdim=True) / num_legals
        curr_centered = torch.where(masks_sl, curr_logits - curr_mean, curr_logits)

        critic_huber = F.smooth_l1_loss(curr_values.view(-1), raw_returns.view(-1), beta=1.0)
        critic_mse = F.mse_loss(curr_values.view(-1), raw_returns.view(-1))
        actor_loss = F.mse_loss(curr_centered[masks_sl], target_centered[masks_sl])

        total_loss = actor_loss + self.coef * critic_huber
        return total_loss, actor_loss, critic_huber, critic_mse

    def train_epochs(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float, float, float]:
        states = batch["states"]
        num_samples = states.size(0)
        if num_samples == 0:
            return 0.0, 0.0, 0.0, 0.0

        b_size = self.config.batch_size
        tot_l, act_l, crt_h, crt_m = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for _ in range(self.config.num_epochs):
            indices = torch.randperm(num_samples, device=states.device)
            for i in range(0, num_samples, b_size):
                idx = indices[i : i + b_size]
                self.optimizer.zero_grad(set_to_none=True)

                curr_l, curr_v = self.model(states[idx])
                loss, a_loss, ch_loss, cm_loss = self.train_step(
                    curr_logits=curr_l, curr_values=curr_v.view(-1),
                    old_logits=batch["old_logits"][idx], norm_adv=batch["norm_adv"][idx],
                    raw_returns=batch["rewards"][idx].view(-1), actions=batch["actions"][idx],
                    legal_masks=batch["legal_masks"][idx], old_probs=batch["old_probs"][idx],
                )

                loss.backward()
                self.optimizer.step()

                tot_l += loss.item(); act_l += a_loss.item()
                crt_h += ch_loss.item(); crt_m += cm_loss.item()
                n_batches += 1

        return tot_l / n_batches, act_l / n_batches, crt_h / n_batches, crt_m / n_batches


class TrainingOrchestrator:
    """C++並列シミュレータの起動、Rolloutデータのアンパック、モデル同期を統括する"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.stop_event = threading.Event()
        os.makedirs(config.model_dir, exist_ok=True)

        self.actor_nets: Dict[str, nn.Module] = {}
        self.traced_actor_nets: Dict[str, Any] = {}
        self.traced_opponent_nets: Dict[str, Any] = {}
        self.trainers: Dict[str, NeuRDTrainer] = {}

        self._init_models()
        self._start_exit_listener()

    def _init_models(self) -> None:
        dummy_normal = torch.zeros((1, 24, 48), dtype=torch.float32, device=self.config.device)
        dummy_koikoi = torch.zeros((1, 24, 48), dtype=torch.float32, device=self.config.device)

        # 1. 学習対象(P1)モデル
        for phase in self.config.phases:
            is_kk = (phase == "koikoi")
            net = NeuRDModel(is_koikoi=is_kk).cpu()
            path = os.path.join(self.config.model_dir, f"{phase}.pt")
            if os.path.exists(path):
                try:
                    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                except Exception as e:
                    print(f"[Warn] NeuRD load skipped for {phase} ({e}). Starting fresh.")

            net.to(self.config.device).float()
            self.actor_nets[phase] = net

            ex_in = dummy_koikoi if is_kk else dummy_normal
            self.traced_actor_nets[phase] = torch.jit.trace(net, ex_in, check_trace=False)

            opt = torch.optim.Adam(net.parameters(), lr=self.config.lr)
            self.trainers[phase] = NeuRDTrainer(net, opt, phase, self.config)

        # 2. アリーナ対戦相手(P2)モデル
        opp_paths = {
            "discard": os.path.join(self.config.model_dir, "arena", "discard.pt"),
            "pick":    os.path.join(self.config.model_dir, "arena", "pick.pt"),
            "koikoi":  os.path.join(self.config.model_dir, "arena", "koikoi.pt"),
        }
        for phase in self.config.phases:
            is_kk = (phase == "koikoi")
            opp_net = NeuRDModel(is_koikoi=is_kk).cpu()
            path = opp_paths[phase]
            if os.path.exists(path):
                opp_net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            else:
                print(f"[Warn] Arena opponent {path} not found. Using initial weights.")

            opp_net.to(self.config.device).eval()
            ex_in = dummy_koikoi if is_kk else dummy_normal
            self.traced_opponent_nets[phase] = torch.jit.trace(opp_net, ex_in, check_trace=False)

    def _start_exit_listener(self) -> None:
        def _listener():
            print("[システム] 画面上で 'q' キーを押して Enter を入力すると停止します。")
            for line in sys.stdin:
                if line.strip().lower() == "q":
                    print("[停止信号検知] モデルを保存して終了します。")
                    self.stop_event.set()
                    break

        threading.Thread(target=_listener, daemon=True).start()

    def _run_cpp_sim(self) -> List[Dict[str, Any]]:
        return koikoicore.run_parallel_simulations(
            self.config.cpu_count, self.config.games_per_thread, self.config.games_per_thread,
            self.config.cap_d, self.config.cap_p, self.config.cap_k, self.config.gamma,
            self.traced_actor_nets["discard"]._c, self.traced_actor_nets["pick"]._c, self.traced_actor_nets["koikoi"]._c,
            self.config.device_str, MAX_ROUND,
            self.config.temperature, self.config.epsilon, self.config.uniform_noise_rate, self.config.gae_lambda,
        )

    def _unpack_and_train(self, rollout_data: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        samples, act_losses, crt_losses = {}, {}, {}
        extract_keys = ("states", "actions", "rewards", "old_logits", "old_values", "legal_masks", "old_action_probs")

        for phase in self.config.phases:
            buffers = {k: [] for k in extract_keys}
            for d in rollout_data:
                if isinstance(d, dict) and phase in d:
                    st = d[phase].get("states")
                    if st is not None and st.numel() > 0:
                        for k in extract_keys: buffers[k].append(d[phase][k])

            if buffers["states"]:
                batch = {
                    ("old_probs" if k == "old_action_probs" else k): torch.cat(v, dim=0).to(self.config.device, non_blocking=True)
                    for k, v in buffers.items()
                }

                adv_raw = batch["rewards"] - batch["old_values"]
                adv_std = adv_raw.std().clamp(min=1e-6)
                batch["norm_adv"] = (adv_raw - adv_raw.mean()) / adv_std

                _, a_l, c_h, _ = self.trainers[phase].train_epochs(batch)
                samples[phase] = batch["states"].size(0)
                act_losses[phase] = a_l; crt_losses[phase] = c_h
            else:
                samples[phase] = 0; act_losses[phase] = 0.0; crt_losses[phase] = 0.0

        return samples, act_losses, crt_losses

    def _run_arena_benchmark(self, loop_idx: int) -> float:
        total_games = self.config.arena_test_games
        if total_games % 2 != 0: total_games += 1

        print_log(f"アリーナテスト実行中")
        res = koikoicore.run_arena_batch_simulations(
            total_games,
            self.traced_actor_nets["discard"]._c, self.traced_actor_nets["pick"]._c, self.traced_actor_nets["koikoi"]._c,
            self.traced_opponent_nets["discard"]._c, self.traced_opponent_nets["pick"]._c, self.traced_opponent_nets["koikoi"]._c,
            self.config.device_str, MAX_ROUND,
        )

        w, l, d = res["win"], res["lose"], res["draw"]
        played = w + l + d
        if played == 0: return 0.0

        score = (d / played) * 0.5 + (w / played)
        avg_pt = res["score"] / played
        print_log(f"■■■  arena {loop_idx:05d}   win {w}   lose {l}  draw {d}   score {avg_pt:.1f}pt  ■■■")
        return score

    def run_training_loop(self) -> None:
        print_log("学習ループ起動")
        rollout_data = self._run_cpp_sim()

        for loop_idx in range(self.config.start_loop, self.config.max_loops):
            t_start = time.perf_counter()
            do_sync = (loop_idx % self.config.sync_model_interval == 0)

            samples, a_loss, c_loss = self._unpack_and_train(rollout_data)

            if do_sync:
                for phase in self.config.phases:
                    self.traced_actor_nets[phase].load_state_dict(self.actor_nets[phase].state_dict())

            rollout_data = self._run_cpp_sim()
            elapsed = time.perf_counter() - t_start

            if loop_idx % 100 == 0:
                print_log(f"time{elapsed:04.1f}s  sample | discard: {samples['discard']:05d} | pick: {samples['pick']:05d} | koikoi: {samples['koikoi']:05d}")

            print_log(
                f"loop{loop_idx:05d}  Loss Act:Cri | "
                f"discard {a_loss['discard']:.6f}:{c_loss['discard']:.2f} | "
                f"pick {a_loss['pick']:.6f}:{c_loss['pick']:.2f} | "
                f"koikoi {a_loss['koikoi']:.6f}:{c_loss['koikoi']:.2f}"
            )

            if loop_idx > 0 and loop_idx % self.config.arena_test_interval == 0:
                self._run_arena_benchmark(loop_idx)

            if self.stop_event.is_set():
                break

        self._save_checkpoints()
        koikoicore.destroy_sim_manager()
        print_log(f"{time_str()} 正常に終了しました。")

    def _save_checkpoints(self) -> None:
        print_log(f"{time_str()} チェックポイントを保存中...")
        for phase in self.config.phases:
            path = os.path.join(self.config.model_dir, f"{phase}.pt")
            torch.save(self.actor_nets[phase].state_dict(), path)


if __name__ == "__main__":
    cfg = TrainingConfig()
    orchestrator = TrainingOrchestrator(cfg)
    orchestrator.run_training_loop()