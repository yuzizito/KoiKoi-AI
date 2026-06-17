import koikoigame
import time
import random

def run_profile_test(num_games=1000):
    game = koikoigame.KoiKoiGameState()
    
    t_start = time.perf_counter()
    
    # 累積用の変数
    total_bb = 0.0
    total_mask = 0.0
    total_copy = 0.0
    total_other = 0.0
    total_count = 0
    
    finished_games = 0
    while finished_games < num_games:
        if game.game_over:
            # === [修正点] リセットされる直前にタイマーの値を回収・累積 ===
            total_bb += getattr(game, 't_feat_bitboard', 0.0)
            total_mask += getattr(game, 't_feat_mask', 0.0)
            total_copy += getattr(game, 't_feat_copy', 0.0)
            total_other += getattr(game, 't_feat_other', 0.0)
            total_count += getattr(game, 't_feat_count', 0)
            
            game.new_game()
            finished_games += 1
            continue
            
        if game.round_state.round_over:
            game.new_round()
            continue

        if game.round_state.wait_action:
            # 内部で特徴量を呼び出して時間を計測
            _ = game.feature_np
            
            mask = game.round_state.action_mask
            valid_actions = [i for i, m in enumerate(mask) if m]
            
            if game.round_state.state == 'koikoi':
                action = random.choice([True, False])
            else:
                if len(valid_actions) > 0:
                    action_idx = random.choice(valid_actions)
                    action = action_idx  # <--- 整数IDをそのまま渡す
                else:
                    action = None
            game.round_state.step(action)
        else:
            game.round_state.step(None)

    # === [修正点] 最終結果の出力 ===
    total_time = total_bb + total_mask + total_copy + total_other
    print(f"Total loop time: {time.perf_counter() - t_start:.2f} s")
    
    print("\n--- feature_np() Profiling Result ---")
    print(f"Total Calls: {total_count}")
    if total_count > 0:
        print(f"Total feature_np() Time : {total_time:.4f} s")
        print(f"  1. Pybind11 / C++ : {total_bb:.4f} s ({total_bb/total_time*100:.1f}%)")
        print(f"  2. List Comp (Mask): {total_mask:.4f} s ({total_mask/total_time*100:.1f}%)")
        print(f"  3. .copy() Cost   : {total_copy:.4f} s ({total_copy/total_time*100:.1f}%)")
        print(f"  4. Other Setup    : {total_other:.4f} s ({total_other/total_time*100:.1f}%)")
    print("-------------------------------------\n")

if __name__ == '__main__':
    run_profile_test(1000)