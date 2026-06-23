import time
import random
import koikoigame

def run_profile_test(num_games=1000):
    game = koikoigame.KoiKoiGameState()
    t_start = perf_counter() # type: ignore
    
    total_bb, total_mask, total_copy, total_other = 0.0, 0.0, 0.0, 0.0
    total_count = 0
    finished_games = 0
    
    while finished_games < num_games:
        if game.game_over:
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

        rs = game.round_state
        if rs.wait_action:
            # 特徴量生成コストを計測
            _ = game.feature_np
            
            mask = rs.action_mask
            valid_actions = [i for i, m in enumerate(mask) if m]
            
            if rs.state == 'koikoi':
                action = random.choice([True, False])
            else:
                # 0〜47の1次元カードIDをそのまま渡す
                action = random.choice(valid_actions) if valid_actions else None
            rs.step(action)
        else:
            rs.step(None)

    total_time = total_bb + total_mask + total_copy + total_other
    print(f"Total loop time: {time.perf_counter() - t_start:.2f} s")
    print(f"Total feature_np() Calls: {total_count}")
    if total_count > 0:
        print(f"Total feature_np() Time : {total_time:.4f} s")

if __name__ == '__main__':
    run_profile_test(1000)