import os
import time
import pickle
import random
import threading
import concurrent.futures

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np

from koikoigame import MAX_ROUND
import koikoilearn
import koikoicore

from koikoinet_v2 import NeuRDModel

# --- 環境設定・スレッド制御 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '15'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
torch.set_num_threads(1)

import torch.nn.modules.linear as my_linear
setattr(my_linear, '_LinearWithBias', my_linear.Linear)

# --- 定数定義 ---
START_LOOP_NUM = 0
LEARNING_RATE = 1e-5
BATCH_SIZE = 8192

CPU_COUNT = 1
LOOP_GAMES = 512
N_CORE_GAMES = LOOP_GAMES // CPU_COUNT
CAP_D = LOOP_GAMES // CPU_COUNT * 128
CAP_P = LOOP_GAMES // CPU_COUNT * 12
CAP_K = LOOP_GAMES // CPU_COUNT * 12

N_LOOP_ACTION_NET_UPDATE = 1
N_LOOP_ARENA_TEST = 100

TEMP = 0.7 # Softmax関数における行動選択時のランダム性
EPS = 0.01 # ランダム行動率
UNR = 0.01 # 多いほどエージェントの探索を促進する

MODEL_FOLDER = 'model_v2'
OPPONENT_PATHS = {
    'discard': 'model_v2/arena/discard.pt',
    'pick': 'model_v2/arena/pick.pt',
    'koikoi': 'model_v2/arena/koikoi.pt'
}
LOG_PATH = 'log.txt'
PHASES = ['discard', 'pick', 'koikoi']

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE_STR = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- グローバル状態 ---
stop_event = threading.Event()

# --- 共通ユーティリティ関数 ---
def time_str():
    return time.strftime("%y%m%d %H%M%S", time.localtime())

def print_log(log_str, log_path=LOG_PATH):
    with open(log_path, 'a', encoding='utf-8') as f:
        print(log_str)
        print(log_str, file=f)

def _patch_model_attributes(model):
    """古いバージョンのPyTorchモデルとの互換性を維持するための属性パッチ"""
    for module in model.modules():
        if type(module).__name__ == 'MultiheadAttention':
            if not hasattr(module, 'batch_first'):
                module.batch_first = False
        elif type(module).__name__ == 'TransformerEncoderLayer':
            if not hasattr(module, 'norm_first'):
                module.norm_first = False
    return model

def init_worker():
    """アリーナテスト用の子プロセスの初期化"""
    seed = (os.getpid() * int(time.time() * 1000)) % 123456789
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    global arena_agent
    models = {}
    for key in PHASES:
        is_kk = (key == 'koikoi')
        models[key] = NeuRDModel(is_koikoi=is_kk).cpu()
        
        path = OPPONENT_PATHS[key]
        if os.path.exists(path):
            models[key].load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        else:
            print(f"Warning: アリーナ対戦相手のモデルが見つかりません。パスを確認してください: {path}")
            
        _patch_model_attributes(models[key])
        models[key] = models[key].to(DEVICE).eval()
        
    arena_agent = koikoilearn.Agent(models['discard'], models['pick'], models['koikoi'])

def wait_for_exit_key():
    """バックグラウンドで終了コマンドを待機するスレッド用関数"""
    print("\n[システム] 画面上で 'q' キーを押して Enter を入力すると停止します。\n")
    while True:
        try:
            user_input = input()
            if user_input.strip().lower() == 'q':
                print("\n[停止シグナル検知] モデルを保存して終了します。")
                stop_event.set()
                break
        except Exception:
            break

def parallel_arena_test(current_state_dicts, n_games):
    """アリーナテスト実行"""
    models = {}
    for key in PHASES:
        is_kk = (key == 'koikoi')
        models[key] = NeuRDModel(is_koikoi=is_kk).cpu()
        models[key].load_state_dict(current_state_dicts[key])
        _patch_model_attributes(models[key])
        models[key] = models[key].to(DEVICE).eval()
        
    agent = koikoilearn.Agent(models['discard'], models['pick'], models['koikoi'])
        
    arena = koikoilearn.Arena(agent, arena_agent)
    arena.multi_game_test(n_games)
    
    result = arena.test_win_num
    result.append(np.mean(arena.test_point[1])) # type: ignore
    return result

def test_result_analysis(result, loop):
    """アリーナテストの集計と結果出力"""
    result = np.array(result)
    win_num = np.sum(result[:, [0, 1, 2]], axis=0)
    win_rate = win_num / np.sum(win_num)
    score = win_rate[0] * 0.5 + win_rate[1]
    point = np.mean(result[:, 3])
    
    print_log(f'■■■  arena {loop:05}   win {int(win_num[1])}   lose {int(win_num[2])}  draw {int(win_num[0])}   score {point:.1f}pt  ■■■')
    return score

@dataclass
class NeuRDConfig:
    eta: float = 0.01
    value_coef: float = 0.01
    
    temperature: float = TEMP
    epsilon: float = EPS 
    uniform_noise_rate: float = UNR

    # GAEパラメータ
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
class NeuRDTrainer:
    def __init__(self, model, optimizer, config: NeuRDConfig):
        self.model = model
        self.optimizer = optimizer
        self.config = config

    @torch.no_grad()
    def _get_minibatch_target(self, old_logits: torch.Tensor, old_values: torch.Tensor, 
                              actions: torch.Tensor, raw_returns: torch.Tensor, 
                              legal_masks: torch.Tensor, old_action_probs: torch.Tensor):
        """ミニバッチ内Advantage正規化 + IS補正 + 合法手ゼロ中心化ターゲット生成"""
        # 1. ミニバッチ内でのAdvantage正規化 (A = (A - mean) / (std + 1e-8))
        adv = raw_returns - old_values
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        sampled_mu = old_action_probs.clamp(min=1e-4)
        batch_idx  = torch.arange(old_logits.size(0), device=old_logits.device)

        # 2. IS補正付きターゲットロジットの導出
        target_logits = old_logits.clone()
        target_logits[batch_idx, actions] += self.config.eta * (adv_norm / sampled_mu)

        # 3. 合法手でのロジットゼロ中心化
        num_legals = legal_masks.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_legal = (target_logits * legal_masks).sum(dim=-1, keepdim=True) / num_legals
        
        centered_target = torch.where(legal_masks, target_logits - mean_legal, target_logits)
        return centered_target

    def train_step(self, current_logits: torch.Tensor, current_values: torch.Tensor, 
                   old_logits: torch.Tensor, old_values: torch.Tensor,
                   raw_returns: torch.Tensor, actions: torch.Tensor, 
                   legal_masks: torch.Tensor, old_action_probs: torch.Tensor):
        
        act_dim = current_logits.size(-1)
        masks_slice      = legal_masks[:, :act_dim]
        old_logits_slice = old_logits[:, :act_dim]

        # ミニバッチターゲットのオンデマンド生成
        target_centered = self._get_minibatch_target(
            old_logits_slice, old_values, actions, raw_returns, masks_slice, old_action_probs
        )

        # 予測ロジット側の中心化
        num_legals = masks_slice.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        curr_mean  = (current_logits * masks_slice).sum(dim=-1, keepdim=True) / num_legals
        curr_centered = torch.where(masks_slice, current_logits - curr_mean, current_logits)

        # Critic Loss (生のReturnを教師とする)
        critic_huber = F.smooth_l1_loss(current_values.view(-1), raw_returns.view(-1), beta=1.0)
        critic_mse   = F.mse_loss(current_values.view(-1), raw_returns.view(-1))

        # Actor Loss (all_dims 固定仕様)
        squared_diff = (curr_centered - target_centered) ** 2
        if masks_slice.dtype != torch.bool:
            masks_slice = masks_slice.bool()
        actor_loss = squared_diff[masks_slice].mean()

        total_loss = actor_loss + self.config.value_coef * critic_huber
        return total_loss, actor_loss, critic_huber, critic_mse

def train_epochs(trainer: NeuRDTrainer, states, actions, old_logits, old_values, raw_returns, legal_masks, old_action_probs, num_epochs: int, batch_size: int):
    num_samples = states.size(0)
    
    total_loss_sum   = 0.0
    actor_loss_sum   = 0.0
    critic_huber_sum = 0.0
    critic_mse_sum   = 0.0
    num_batches = 0
    
    for epoch in range(num_epochs):
        indices = torch.randperm(num_samples, device=states.device)
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            
            b_states = states[batch_idx]
            b_actions = actions[batch_idx]
            b_old_logits = old_logits[batch_idx]
            b_old_values = old_values[batch_idx]
            b_raw_returns = raw_returns[batch_idx]
            b_legal_masks = legal_masks[batch_idx]
            b_old_action_probs = old_action_probs[batch_idx]
            
            trainer.optimizer.zero_grad()
            b_current_logits, b_current_values = trainer.model(b_states)
            
            loss, actor_loss, c_huber, c_mse = trainer.train_step(
                current_logits=b_current_logits,
                current_values=b_current_values.view(-1),
                old_logits=b_old_logits,
                old_values=b_old_values,
                raw_returns=b_raw_returns.view(-1),
                actions=b_actions,
                legal_masks=b_legal_masks,
                old_action_probs=b_old_action_probs
            )
            
            loss.backward()
            trainer.optimizer.step()
            
            total_loss_sum   += loss.item()
            actor_loss_sum   += actor_loss.item()
            critic_huber_sum += c_huber.item()
            critic_mse_sum   += c_mse.item()
            num_batches += 1
            
    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    return total_loss_sum / num_batches, actor_loss_sum / num_batches, critic_huber_sum / num_batches, critic_mse_sum / num_batches

if __name__ == '__main__':
    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    
    exit_thread = threading.Thread(target=wait_for_exit_key, daemon=True)
    exit_thread.start()

    # ---------- 学習対象モデル(P1)のロードとトレース ----------
    action_net = {}
    traced_action_net = {}
    example_input_normal = torch.zeros((1, 24, 48), dtype=torch.float32, device=DEVICE)
    example_input_koikoi = torch.zeros((1, 24, 48), dtype=torch.float32, device=DEVICE)
    
    for key in PHASES:
        is_kk = (key == 'koikoi')
        action_net[key] = NeuRDModel(is_koikoi=is_kk).cpu()
        path = f'{MODEL_FOLDER}/{key}.pt'
        if os.path.exists(path):
            try:
                action_net[key].load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            except Exception as e:
                print(f"Warning: NeuRD重みの読み込みスキップ ({e})。初期状態から開始します。")
        
        _patch_model_attributes(action_net[key])
        action_net[key].to(DEVICE).float().eval()
        
        ex_input = example_input_koikoi if key == 'koikoi' else example_input_normal
        traced_action_net[key] = torch.jit.trace(action_net[key], ex_input, check_trace=False)

    # ---------- 対戦相手モデル(P2)のロードとトレース ----------
    # Design Intent: Multiprocessingを廃止したため、メインスレッドで一度だけ対戦相手をロード・トレースする
    print_log("アリーナ対戦相手モデルを読み込み中...")
    arena_opponent_models = {}
    traced_arena_opponent_models = {}
    
    for key in PHASES:
        is_kk = (key == 'koikoi')
        arena_opponent_models[key] = NeuRDModel(is_koikoi=is_kk).cpu()
        path = OPPONENT_PATHS[key]
        if os.path.exists(path):
            arena_opponent_models[key].load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        else:
            print(f"Warning: アリーナ対戦相手が見つかりません。初期状態を使用: {path}")
        
        _patch_model_attributes(arena_opponent_models[key])
        arena_opponent_models[key] = arena_opponent_models[key].to(DEVICE).eval()
        
        ex_input = example_input_koikoi if key == 'koikoi' else example_input_normal
        traced_arena_opponent_models[key] = torch.jit.trace(arena_opponent_models[key], ex_input, check_trace=False)

    # ========== 動作テスト用 NeuRD 初期化 ==========
    neurd_config = NeuRDConfig(
        eta=0.01, value_coef=0.01, temperature=1.0, epsilon=0.05, uniform_noise_rate=0.0
    )

    trainer = {
        key: NeuRDTrainer(
            model=action_net[key], 
            optimizer=torch.optim.Adam(action_net[key].parameters(), lr=LEARNING_RATE), 
            config=neurd_config
        )
        for key in PHASES
    }

    score = [0.0]

    def run_sim():
        return koikoicore.run_parallel_simulations(
            CPU_COUNT, N_CORE_GAMES, N_CORE_GAMES,          
            CAP_D, CAP_P, CAP_K, neurd_config.gamma,
            traced_action_net['discard']._c, traced_action_net['pick']._c, traced_action_net['koikoi']._c,
            DEVICE_STR, MAX_ROUND,
            neurd_config.temperature, neurd_config.epsilon, neurd_config.uniform_noise_rate,
            neurd_config.gae_lambda
        )

    def sync_neurd_training(data_list, trainers, num_epochs=3, batch_size=BATCH_SIZE):
        actor_losses, critic_huber_losses, critic_mse_losses, samples = {}, {}, {}, {}  # ★辞書分離
        for key in PHASES:
            tensors = {k: [] for k in ['states', 'actions', 'rewards', 'old_logits', 'old_values', 'legal_masks', 'old_action_probs']}
            for d in data_list:
                if isinstance(d, dict) and key in d:
                    s = d[key].get('states')
                    if s is not None and s.numel() > 0:
                        for k in tensors.keys():
                            tensors[k].append(d[key][k])
        
            if tensors['states']:
                batch = {k: torch.cat(v, dim=0).to(DEVICE) for k, v in tensors.items()}
                _, a_loss, c_huber, c_mse = train_epochs(
                    trainer=trainers[key],
                    states=batch['states'], actions=batch['actions'], old_logits=batch['old_logits'],
                    old_values=batch['old_values'], raw_returns=batch['rewards'], legal_masks=batch['legal_masks'],
                    old_action_probs=batch['old_action_probs'],
                    num_epochs=num_epochs, batch_size=batch_size
                )
                actor_losses[key] = a_loss
                critic_huber_losses[key] = c_huber
                critic_mse_losses[key] = c_mse
                samples[key] = batch['states'].shape[0]
            else:
                actor_losses[key] = 0.0
                critic_huber_losses[key] = 0.0
                critic_mse_losses[key] = 0.0
                samples[key] = 0
            
        return samples, actor_losses, critic_huber_losses, critic_mse_losses

    print_log("\n学習ループを起動します...")
    results = run_sim()
    
    for loop in range(START_LOOP_NUM, 10000):
        loop_start_time = time.perf_counter()
        sync_models = (loop % N_LOOP_ACTION_NET_UPDATE == 0)
        
        # 1. 取得したRolloutデータで即座に学習 (On-policy)
        samples, act_losses, crt_huber, crt_mse = sync_neurd_training(results, trainer, num_epochs=3, batch_size=BATCH_SIZE)
        
        # 推論用トレースモデルへの重み同期
        if sync_models:
            for key in PHASES:
                traced_action_net[key].load_state_dict(action_net[key].state_dict())
        
        # 次のOn-policyデータの収集
        results = run_sim()
        
        elapsed_time = time.perf_counter() - loop_start_time
        
        if loop % 100 == 0:
            print_log(f'sample | discard: {samples["discard"]:05} | pick: {samples["pick"]:05} | koikoi: {samples["koikoi"]:05}')
        
        print_log(f'loop{loop:05} time{elapsed_time:04.1f}s  Loss Actor:Critic | discard {act_losses["discard"]:.4f}:{crt_huber["discard"]:.4f} | pick {act_losses["pick"]:.4f}:{crt_huber["pick"]:.4f} | koikoi {act_losses["koikoi"]:.4f}:{crt_huber["koikoi"]:.4f}')
        
        # 4. 定期的なアリーナテスト
        if loop > 0 and loop % N_LOOP_ARENA_TEST == 0:
            total_arena_games = 500
            if total_arena_games % 2 != 0:
                total_arena_games += 1 # Validation: Duplicate Match のため必ず偶数(ペア)にする
                
            print_log(f"アリーナテスト実行中... (C++ Backend, Batched Inference: {total_arena_games} games)")
            
            # Design Intent: マルチスレッド分割を廃止し、全環境を1つの巨大バッチにまとめてGPU推論を最大化する
            arena_res = koikoicore.run_arena_batch_simulations(
                total_arena_games,
                traced_action_net['discard']._c, traced_action_net['pick']._c, traced_action_net['koikoi']._c,
                traced_arena_opponent_models['discard']._c, traced_arena_opponent_models['pick']._c, traced_arena_opponent_models['koikoi']._c,
                DEVICE_STR, MAX_ROUND
            )
            
            # 結果の集計と出力
            w, l, d = arena_res["win"], arena_res["lose"], arena_res["draw"]
            total_played = w + l + d
            
            if total_played > 0:
                wr = np.array([d, w, l]) / total_played
                s = wr[0] * 0.5 + wr[1]
                avg_pt = arena_res["score"] / total_played
                
                print_log(f'■■■  arena {loop:05}   win {w}   lose {l}  draw {d}   score {avg_pt:.1f}pt  ■■■')
                score.append(s)
            
        # 5. 安全な終了処理と保存
        if stop_event.is_set():
            break
        
    print_log(f'\n{time_str()} チェックポイントを保存中...')
    for key in PHASES:
        torch.save(action_net[key].state_dict(), f'{MODEL_FOLDER}/{key}.pt')
                
    koikoicore.destroy_sim_manager()
    print_log(f'{time_str()} 正常に終了しました。')
    