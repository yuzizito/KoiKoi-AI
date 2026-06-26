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
from koikoinet_v2 import NeuRDModel
import koikoilearn
import koikoicore

from koikoinet_v2 import DiscardModel, PickModel, KoiKoiModel, TargetQNet, NeuRDModel

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
BATCH_SIZE = 4096

CPU_COUNT = 1
LOOP_GAMES = 512
N_CORE_GAMES = LOOP_GAMES // CPU_COUNT
CAP_D = LOOP_GAMES // CPU_COUNT * 128
CAP_P = LOOP_GAMES // CPU_COUNT * 12
CAP_K = LOOP_GAMES // CPU_COUNT * 12

N_LOOP_ACTION_NET_UPDATE = 1
N_LOOP_ARENA_TEST = 50
ARENA_WORKERS = 4

TEMP = 1.0 # Softmax関数における行動選択時のランダム性
EPS = 0.05 # ランダム行動率
UNR = 0.0 # 多いほどエージェントの探索を促進する

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
    loss_mode: str = 'all_dims' # selected_only: 選択行動のみMSE / all_dims: 全48次元MSE
    return_scale_mode: str = 'win_loss' # win_loss: [-1, 1] / normalize: 点数を60で割る / raw: 生の値
    clip_min: float = -12.0
    clip_max: float = 12.0
    max_score: float = 60.0
    eta: float = 0.01
    value_coef: float = 0.5
    
    # 探索パラメータ（C++側への引き渡し用）
    temperature: float = TEMP
    epsilon: float = EPS 
    uniform_noise_rate: float = UNR
    
class NeuRDTrainer:
    def __init__(self, model, optimizer, config: NeuRDConfig):
        self.model = model
        self.optimizer = optimizer
        self.config = config

    def _scale_returns(self, raw_returns: torch.Tensor) -> torch.Tensor:
        """
        Design Intent: NeuRDの更新ステップ幅(ηA)を安定させるための報酬スケーリング。
        """
        mode = self.config.return_scale_mode
        if mode == 'win_loss':
            return torch.sign(raw_returns)
        elif mode == 'clip':
            return torch.clamp(raw_returns, min=self.config.clip_min, max=self.config.clip_max)
        elif mode == 'normalize':
            # Validation: ゼロ除算の防止
            if self.config.max_score <= 0:
                raise ValueError("max_score must be strictly positive.")
            return raw_returns / self.config.max_score
        elif mode == 'raw':
            return raw_returns
        else:
            raise ValueError(f"Unknown return_scale_mode: {mode}")

    @torch.no_grad()
    def prepare_fixed_targets(self, old_logits: torch.Tensor, old_values: torch.Tensor, actions: torch.Tensor, raw_returns: torch.Tensor):
        """
        Design Intent: NeuRDのEquation 9に基づく固定ターゲットをRollout直後に1回だけ計算する。
        複数エポック回す際にAdvantageやTargetを再計算しないことで、Actor Targetの固定要件を満たす。
        """
        scaled_returns = self._scale_returns(raw_returns)
        # old_values はC++から渡されたrollout収集時のfrozen value
        advantage = scaled_returns - old_values

        batch_idx = torch.arange(old_logits.size(0), device=old_logits.device)
        
        target_logits = old_logits.clone()
        target_logits[batch_idx, actions] += self.config.eta * advantage

        return scaled_returns, target_logits

    def train_step(self, current_logits: torch.Tensor, current_values: torch.Tensor, 
                   fixed_target_logits: torch.Tensor, scaled_returns: torch.Tensor, 
                   actions: torch.Tensor, legal_masks: torch.Tensor):
        """
        Design Intent: 事前計算された固定ターゲット(fixed_target_logits, scaled_returns)を使用してLossを計算する。
        """
        # Critic Loss: 固定されたスケール済み報酬をターゲットとする
        critic_loss = F.mse_loss(current_values.view(-1), scaled_returns.view(-1))

        # Actor Loss
        batch_idx = torch.arange(current_logits.size(0), device=current_logits.device)
        
        if self.config.loss_mode == 'all_dims':
            # Design Intent: 合法手数の違いによる勾配スケールの偏りを防ぐため、合法手のみの平均MSEをとる
            squared_diff = (current_logits - fixed_target_logits) ** 2
            
            # Validation: legal_masksがbool型であることを確認してインデックス抽出を安全に行う
            if legal_masks.dtype != torch.bool:
                legal_masks = legal_masks.bool()
                
            actor_loss = squared_diff[legal_masks].mean()
            
        elif self.config.loss_mode == 'selected_only':
            current_selected = current_logits[batch_idx, actions]
            target_selected = fixed_target_logits[batch_idx, actions]
            actor_loss = F.mse_loss(current_selected, target_selected)
            
        else:
            raise ValueError(f"Unknown loss_mode: {self.config.loss_mode}")

        total_loss = actor_loss + self.config.value_coef * critic_loss
        
        return total_loss, actor_loss, critic_loss

def train_epochs(trainer: NeuRDTrainer, states, actions, old_logits, old_values, raw_returns, legal_masks, num_epochs: int, batch_size: int):
    """
    Design Intent: Rolloutデータに対して、固定ターゲットを事前計算した上で複数エポック学習を回す。
    平均Lossを計算して返す。
    """
    num_samples = states.size(0)
    
    # 1. Rollout直後に1回だけターゲットを計算して固定する
    scaled_returns, fixed_target_logits = trainer.prepare_fixed_targets(
        old_logits=old_logits,
        old_values=old_values,
        actions=actions,
        raw_returns=raw_returns
    )
    
    total_loss_sum = 0.0
    actor_loss_sum = 0.0
    critic_loss_sum = 0.0
    num_batches = 0
    
    # 2. 複数エポックの学習ループ
    for epoch in range(num_epochs):
        indices = torch.randperm(num_samples, device=states.device)
        
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            
            b_states = states[batch_idx]
            b_actions = actions[batch_idx]
            b_legal_masks = legal_masks[batch_idx]
            
            b_scaled_returns = scaled_returns[batch_idx]
            b_fixed_target_logits = fixed_target_logits[batch_idx]
            
            trainer.optimizer.zero_grad()
            
            b_current_logits, b_current_values = trainer.model(b_states)
            
            loss, actor_loss, critic_loss = trainer.train_step(
                current_logits=b_current_logits,
                current_values=b_current_values.view(-1),
                fixed_target_logits=b_fixed_target_logits,
                scaled_returns=b_scaled_returns.view(-1),
                actions=b_actions,
                legal_masks=b_legal_masks
            )
            
            loss.backward()
            trainer.optimizer.step()
            
            total_loss_sum += loss.item()
            actor_loss_sum += actor_loss.item()
            critic_loss_sum += critic_loss.item()
            num_batches += 1
            
    if num_batches == 0:
        return 0.0, 0.0, 0.0
        
    return total_loss_sum / num_batches, actor_loss_sum / num_batches, critic_loss_sum / num_batches

if __name__ == '__main__':
    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
        
    exit_thread = threading.Thread(target=wait_for_exit_key, daemon=True)
    exit_thread.start()
        
    with open('win_prob_mat.pkl', 'rb') as f:
        win_prob_mat = pickle.load(f)

    action_net = {}
    for key in PHASES:
        is_kk = (key == 'koikoi')
        action_net[key] = NeuRDModel(is_koikoi=is_kk).cpu()
        
        path = f'{MODEL_FOLDER}/{key}.pt'
        if os.path.exists(path):
            try:
                # 構造が変更されたため、古い重みを読み込んだ場合はエラーとなり初期状態から始まります
                action_net[key].load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            except Exception as e:
                print(f"Warning: NeuRD重みの読み込みスキップ ({e})。初期状態から開始します。")
        
        _patch_model_attributes(action_net[key])
    
    example_input_normal = torch.zeros((1, 24, 48), dtype=torch.float32, device=DEVICE)
    example_input_koikoi = torch.zeros((1, 24, 48), dtype=torch.float32, device=DEVICE)

    traced_action_net = {}
    
    for key in PHASES:
        ex_input = example_input_koikoi if key == 'koikoi' else example_input_normal
        
        # NeuRDModelを評価モードでC++推論用にトレース
        action_net[key].to(DEVICE).float().eval()
        traced_action_net[key] = torch.jit.trace(action_net[key], ex_input, check_trace=False)

    # ========== 動作テスト用 NeuRD 初期化 ==========
    neurd_config = NeuRDConfig(
        loss_mode='selected_only', 
        return_scale_mode='win_loss', # まずは勝敗ベース[-1, 1]でテスト
        eta=0.01,
        value_coef=0.5,
        temperature=1.0,
        epsilon=0.05,
        uniform_noise_rate=0.0
    )

    # 既存の C++ Trainer (koikoicore.KoiKoiTrainer) を破棄し、Python側の NeuRDTrainer に置き換え
    trainer = {
        key: NeuRDTrainer(
            model=action_net[key], 
            optimizer=torch.optim.Adam(action_net[key].parameters(), lr=LEARNING_RATE), 
            config=neurd_config
        )
        for key in PHASES
    }

    score = [0.0]
    wp_mat_np = win_prob_mat.astype(np.float32) if win_prob_mat is not None else np.zeros((2, 9, 61), dtype=np.float32)

    # C++シミュレーション呼び出し用ヘルパー
    def run_sim():
        return koikoicore.run_parallel_simulations(
            CPU_COUNT, N_CORE_GAMES, N_CORE_GAMES,          
            CAP_D, CAP_P, CAP_K, 1.0, wp_mat_np,
            traced_action_net['discard']._c, traced_action_net['pick']._c, traced_action_net['koikoi']._c,
            DEVICE_STR, MAX_ROUND,
            neurd_config.temperature, neurd_config.epsilon, neurd_config.uniform_noise_rate # <-- 追加
        )

    def sync_neurd_training(data_list, trainers, num_epochs=3, batch_size=BATCH_SIZE):
        losses, samples = {}, {}
        for key in PHASES:
            tensors = {k: [] for k in ['states', 'actions', 'rewards', 'old_logits', 'old_values', 'legal_masks']}
            
            # Python側でバッチデータを抽出・結合
            for d in data_list:
                if isinstance(d, dict) and key in d:
                    s = d[key].get('states')
                    # Validation: テンソルが存在し、かつ空でないことを確認
                    if s is not None and s.numel() > 0:
                        for k in tensors.keys():
                            tensors[k].append(d[key][k])
            
            if tensors['states']:
                batch = {k: torch.cat(v, dim=0).to(DEVICE) for k, v in tensors.items()}
                
                # 同期的に複数エポックのNeuRD学習を実行
                avg_loss, _, _ = train_epochs(
                    trainer=trainers[key],
                    states=batch['states'],
                    actions=batch['actions'],
                    old_logits=batch['old_logits'],
                    old_values=batch['old_values'],
                    raw_returns=batch['rewards'],
                    legal_masks=batch['legal_masks'],
                    num_epochs=num_epochs,
                    batch_size=batch_size
                )
                losses[key] = avg_loss
                samples[key] = batch['states'].shape[0]
            else:
                losses[key] = 0.0
                samples[key] = 0
                
        return samples, losses
    
    print_log("\n学習ループを起動します...")
    results = run_sim()
    
    for loop in range(START_LOOP_NUM, 10000):
        loop_start_time = time.perf_counter()
        sync_models = (loop % N_LOOP_ACTION_NET_UPDATE == 0)
        
        # 1. 取得したRolloutデータで即座に学習 (On-policy)
        samples, losses = sync_neurd_training(results, trainer, num_epochs=3, batch_size=BATCH_SIZE)
        
        # 2. 推論用(C++送信用)トレースモデルへの重み同期
        if sync_models:
            for key in PHASES:
                # Pythonモデル(Actor-Critic両方を含む)の重みを、C++推論用のTracedモデルへ同期
                traced_action_net[key].load_state_dict(action_net[key].state_dict())
        
        # 3. 次のOn-policyデータの収集
        results = run_sim()
        
        elapsed_time = time.perf_counter() - loop_start_time
        print_log(f'loop {loop:05}  time {elapsed_time:04.1f}s    loss:sample  discard {losses["discard"]:.4f}:{samples["discard"]:05}  pick {losses["pick"]:.4f}:{samples["pick"]:05}  koikoi {losses["koikoi"]:.4f}:{samples["koikoi"]:05}')
        
        # 4. 定期的なアリーナテスト
        if loop > 0 and loop % N_LOOP_ARENA_TEST == 0:
            current_state_dicts = {
                key: {k: v.cpu() for k, v in traced_action_net[key].state_dict().items()}
                for key in PHASES
            }
            
            with mp.Pool(ARENA_WORKERS, initializer=init_worker) as pool:
                result_async = [
                    pool.apply_async(parallel_arena_test, args=(current_state_dicts, 200 // ARENA_WORKERS)) 
                    for _ in range(ARENA_WORKERS)
                ]
                arena_results = [res.get() for res in result_async]
            
            torch.cuda.empty_cache()
            s = test_result_analysis(arena_results, loop)
            score.append(s)
            
        # 5. 安全な終了処理と保存
        if stop_event.is_set():
            print_log(f'\n{time_str()} チェックポイントを保存中...')
            for key in PHASES:
                torch.save(action_net[key].state_dict(), f'{MODEL_FOLDER}/{key}.pt')
                
            koikoicore.destroy_sim_manager()
            print_log(f'{time_str()} プログラムを安全に終了しました。')
            break