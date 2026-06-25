#include <torch/extension.h>
#include <torch/script.h>
#include <torch/optim.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <array>
#include <cmath>
#include <mutex>
#include <tuple>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <iostream>

#if defined(_MSVC_LANG) || defined(_MSC_VER)
#include <intrin.h>
#endif

namespace py = pybind11;

// ---------------------------------------------------------
// モデルや定数のグローバル管理
// ---------------------------------------------------------
struct SharedModels {
    torch::jit::script::Module discard, pick, koikoi;
    std::mutex mtx;
    bool loaded = false;
};
static SharedModels g_models;

struct KoiKoiMasks {
    uint64_t light = 0, seed = 0, ribbon = 0, dross = 0;
    uint64_t boar_deer_butterfly = 0, flower_sake = 0, moon_sake = 0;
    uint64_t red_ribbon = 0, blue_ribbon = 0, red_blue_ribbon = 0;
    uint64_t crane = 0, curtain = 0, moon = 0, rainman = 0, phoenix = 0, sake = 0;

    constexpr KoiKoiMasks() {
        auto add = [](uint64_t& mask, int s, int r) { mask |= (1ULL << ((s - 1) * 4 + (r - 1))); };
        add(crane, 1, 1); add(curtain, 3, 1); add(moon, 8, 1);
        add(rainman, 11, 1); add(phoenix, 12, 1); add(sake, 9, 1);

        light = crane | curtain | moon | rainman | phoenix;

        int seeds[][2] = {{2,1},{4,1},{5,1},{6,1},{7,1},{8,2},{9,1},{10,1},{11,2}};
        for (const auto& s : seeds) add(seed, s[0], s[1]);

        int ribbons[][2] = {{1,2},{2,2},{3,2},{4,2},{5,2},{6,2},{7,2},{9,2},{10,2},{11,3}};
        for (const auto& r : ribbons) add(ribbon, r[0], r[1]);

        int dross_list[][2] = {
            {1,3},{1,4},{2,3},{2,4},{3,3},{3,4},{4,3},{4,4},{5,3},{5,4},{6,3},{6,4},{7,3},
            {7,4},{8,3},{8,4},{9,3},{9,4},{10,3},{10,4},{11,4},{12,2},{12,3},{12,4},{9,1}
        };
        for (const auto& d : dross_list) add(dross, d[0], d[1]);

        add(boar_deer_butterfly, 6, 1); add(boar_deer_butterfly, 7, 1); add(boar_deer_butterfly, 10, 1);
        flower_sake = curtain | sake;
        moon_sake = moon | sake;

        add(red_ribbon, 1, 2); add(red_ribbon, 2, 2); add(red_ribbon, 3, 2);
        add(blue_ribbon, 6, 2); add(blue_ribbon, 9, 2); add(blue_ribbon, 10, 2);
        red_blue_ribbon = red_ribbon | blue_ribbon;
    }
};
static const KoiKoiMasks MASKS;

template<typename T, int Capacity>
struct FixedVec {
    T data_[Capacity];
    int size_ = 0;

    void clear() { size_ = 0; }
    void push_back(const T& val) { if (size_ < Capacity) data_[size_++] = val; }
    void pop_back() { if (size_ > 0) size_--; }
    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }
    int size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T& operator[](int i) { return data_[i]; }
    const T& operator[](int i) const { return data_[i]; }

    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }

    void erase(T* it) {
        int idx = static_cast<int>(it - data_);
        for(int i = idx; i < size_ - 1; ++i) {
            data_[i] = data_[i+1];
        }
        size_--;
    }

    void insert_end(const T* first, const T* last) {
        int n = static_cast<int>(last - first);
        for(int i = 0; i < n && size_ < Capacity; ++i) {
            data_[size_++] = first[i];
        }
    }
};

// --- ビット演算ヘルパー関数 ---
inline int popcount(uint64_t bb) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bb);
#elif defined(_MSC_VER)
    return (int)__popcnt64(bb);
#else
    bb =           bb - ((bb >> 1) & 0x5555555555555555ULL);
    bb = (bb & 0x3333333333333333ULL) + ((bb >> 2) & 0x3333333333333333ULL);
    return (int)((((bb + (bb >> 4)) & 0xF0F0F0F0F0F0F0FULL) * 0x101010101010101ULL) >> 56);
#endif
}

inline int ctz64(uint64_t bb) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bb);
#elif defined(_MSC_VER)
    unsigned long c; _BitScanForward64(&c, bb); return (int)c;
#else
    int c = 0; while((bb & 1) == 0) { bb >>= 1; c++; } return c;
#endif
}
// ----------------------------

int get_yaku_point_by_bitboard(uint64_t pile, int koikoi_num) {
    int total_point = 0;
    int num_light = popcount(pile & MASKS.light);
    if (num_light == 5) total_point += 10;
    else if (num_light == 4 && (pile & MASKS.rainman) == 0) total_point += 8;
    else if (num_light == 4) total_point += 7;
    else if (num_light == 3 && (pile & MASKS.rainman) == 0) total_point += 5;

    int num_seed = popcount(pile & MASKS.seed);
    if ((pile & MASKS.boar_deer_butterfly) == MASKS.boar_deer_butterfly) total_point += 5;
    if ((pile & MASKS.flower_sake) == MASKS.flower_sake) total_point += (koikoi_num == 0 ? 1 : 3);
    if ((pile & MASKS.moon_sake) == MASKS.moon_sake) total_point += (koikoi_num == 0 ? 1 : 3);
    if (num_seed >= 5) total_point += (num_seed - 4);

    int num_ribbon = popcount(pile & MASKS.ribbon);
    if ((pile & MASKS.red_blue_ribbon) == MASKS.red_blue_ribbon) total_point += 10;
    if ((pile & MASKS.red_ribbon) == MASKS.red_ribbon) total_point += 5;
    if ((pile & MASKS.blue_ribbon) == MASKS.blue_ribbon) total_point += 5;
    if (num_ribbon >= 5) total_point += (num_ribbon - 4);

    int num_dross = popcount(pile & MASKS.dross);
    if (num_dross >= 10) total_point += (num_dross - 9);

    if (koikoi_num <= 3) total_point += koikoi_num;
    else total_point *= (koikoi_num - 2);

    return total_point;
}

std::vector<std::tuple<int, std::string, int>> evaluate_yaku_by_bitboard(uint64_t pile, int koikoi_num) {
    std::vector<std::tuple<int, std::string, int>> yaku_list;
    int num_light = popcount(pile & MASKS.light);
    if (num_light == 5) yaku_list.emplace_back(1, "Five Lights", 10);
    else if (num_light == 4 && (pile & MASKS.rainman) == 0) yaku_list.emplace_back(2, "Four Lights", 8);
    else if (num_light == 4) yaku_list.emplace_back(3, "Rainy Four Lights", 7);
    else if (num_light == 3 && (pile & MASKS.rainman) == 0) yaku_list.emplace_back(4, "Three Lights", 5);

    int num_seed = popcount(pile & MASKS.seed);
    if ((pile & MASKS.boar_deer_butterfly) == MASKS.boar_deer_butterfly) yaku_list.emplace_back(5, "Boar-Deer-Butterfly", 5);
    if ((pile & MASKS.flower_sake) == MASKS.flower_sake) yaku_list.emplace_back(koikoi_num == 0 ? 6 : 7, "Flower Viewing Sake", koikoi_num == 0 ? 1 : 3);
    if ((pile & MASKS.moon_sake) == MASKS.moon_sake) yaku_list.emplace_back(koikoi_num == 0 ? 8 : 9, "Moon Viewing Sake", koikoi_num == 0 ? 1 : 3);
    if (num_seed >= 5) yaku_list.emplace_back(10, "Tane", num_seed - 4);

    int num_ribbon = popcount(pile & MASKS.ribbon);
    if ((pile & MASKS.red_blue_ribbon) == MASKS.red_blue_ribbon) yaku_list.emplace_back(11, "Red & Blue Ribbons", 10);
    if ((pile & MASKS.red_ribbon) == MASKS.red_ribbon) yaku_list.emplace_back(12, "Red Ribbons", 5);
    if ((pile & MASKS.blue_ribbon) == MASKS.blue_ribbon) yaku_list.emplace_back(13, "Blue Ribbons", 5);
    if (num_ribbon >= 5) yaku_list.emplace_back(14, "Tan", num_ribbon - 4);

    int num_dross = popcount(pile & MASKS.dross);
    if (num_dross >= 10) yaku_list.emplace_back(15, "Kasu", num_dross - 9);
    if (koikoi_num > 0) yaku_list.emplace_back(16, "Koi-Koi", koikoi_num);

    return yaku_list;
}

struct Snapshot {
    uint64_t bb_hand;
    uint64_t bb_unseen;
    uint64_t bb_my_pile;
    uint64_t bb_field;
    uint64_t bb_op_pile;
    
    // --- 新規追加: 信念状態・履歴トラッキング ---
    uint64_t bb_my_discard;
    uint16_t suit_op_played;
    uint16_t suit_op_ignored;

    int point_turn;
    int point_idle;
    int round_num;
    int turn_16;
    bool is_dealer;
    int koikoi_num_turn;
    int koikoi_num_idle;
    uint8_t state_type;
};

struct TraceSlot {
    int state_type; 
    int action;
    float q_val;
    Snapshot snap;
};

inline void write_feature_core(
    float* dest, const Snapshot& snap, int action_to_align, int max_round)
{
    // [24, 48] のゼロクリア
    std::memset(dest, 0, 24 * 48 * sizeof(float));

    auto set_row_bb = [&](int row, uint64_t bb, float val = 1.0f) {
        while (bb) {
            int c = ctz64(bb);
            dest[row * 48 + c] = val;
            bb &= bb - 1;
        }
    };

    // 1. 物理的なカード配置（C0〜C5）
    set_row_bb(0, snap.bb_hand);
    set_row_bb(1, snap.bb_field);
    set_row_bb(2, snap.bb_my_pile);
    set_row_bb(3, snap.bb_op_pile);
    set_row_bb(4, snap.bb_unseen);
    set_row_bb(5, snap.bb_my_discard);

    // 2. 札属性・役ポテンシャル（C6〜C14）
    set_row_bb(6, MASKS.light);
    set_row_bb(7, MASKS.seed);
    set_row_bb(8, MASKS.ribbon);
    set_row_bb(9, MASKS.dross);

    // C10: 月の残り枚数（未知の札を月ごとに集計して正規化）
    for (int m = 0; m < 12; ++m) {
        uint64_t month_bb = 0xFULL << (m * 4);
        int unseen_count = popcount(snap.bb_unseen & month_bb);
        float val = unseen_count / 4.0f;
        for (int i = 0; i < 4; ++i) dest[10 * 48 + (m * 4 + i)] = val;
    }

    // C11〜C14: 役リーチフラグ（場札を獲得したと仮定した場合の役ポイント変動をチェック）
    int current_pt = get_yaku_point_by_bitboard(snap.bb_my_pile, snap.koikoi_num_turn);
    uint64_t field_tmp = snap.bb_field;
    while (field_tmp) {
        int c = ctz64(field_tmp);
        int next_pt = get_yaku_point_by_bitboard(snap.bb_my_pile | (1ULL << c), snap.koikoi_num_turn);
        if (next_pt > current_pt) {
            if ((1ULL << c) & MASKS.light) dest[11 * 48 + c] = 1.0f;
            if ((1ULL << c) & MASKS.seed)  dest[12 * 48 + c] = 1.0f;
            if ((1ULL << c) & MASKS.ribbon) dest[13 * 48 + c] = 1.0f;
            if ((1ULL << c) & MASKS.dross) dest[14 * 48 + c] = 1.0f;
        }
        field_tmp &= field_tmp - 1;
    }

    // 3. 信念状態 / Belief State（C15, C16）
    for (int m = 0; m < 12; ++m) {
        // C15: 無視された月（相手が持っていない可能性が高い）
        if ((snap.suit_op_ignored >> m) & 1) {
            for (int i = 0; i < 4; ++i) dest[15 * 48 + (m * 4 + i)] = -1.0f;
        }
        // C16: 生出しされた月（相手が手札からプレイした）
        if ((snap.suit_op_played >> m) & 1) {
            for (int i = 0; i < 4; ++i) dest[16 * 48 + (m * 4 + i)] = 1.0f;
        }
    }

    // 4. グローバルコンテキスト（C17〜C23）
    float c17 = snap.point_turn / 60.0f;
    float c18 = snap.point_idle / 60.0f;
    int my_yaku = get_yaku_point_by_bitboard(snap.bb_my_pile, snap.koikoi_num_turn);
    int op_yaku = get_yaku_point_by_bitboard(snap.bb_op_pile, snap.koikoi_num_idle);
    float c19 = (float)my_yaku / (op_yaku + 1.0f);
    float c20 = (float)snap.koikoi_num_turn / (snap.koikoi_num_idle + 1.0f);
    float c21 = snap.turn_16 / 16.0f;
    float c22 = snap.is_dealer ? 1.0f : -1.0f;
    float c23 = (float)snap.round_num / max_round;

    for (int i = 0; i < 48; ++i) {
        dest[17 * 48 + i] = c17;
        dest[18 * 48 + i] = c18;
        dest[19 * 48 + i] = c19;
        dest[20 * 48 + i] = c20;
        dest[21 * 48 + i] = c21;
        dest[22 * 48 + i] = c22;
        dest[23 * 48 + i] = c23;
    }

    // [重要] Q-Target学習のために選択したアクションをColumn 0に寄せる既存仕様を継承
    // （アクション評価時のみ発動し、普段の推論時は action_to_align = -1 のため作動しません）
    if (action_to_align > 0 && action_to_align < 48) {
        float temp;
        for (int r = 0; r < 24; ++r) {
            temp = dest[r * 48];
            dest[r * 48] = dest[r * 48 + action_to_align];
            dest[r * 48 + action_to_align] = temp;
        }
    }
}

class KoiKoiStateManager {
public:
    uint64_t bb_hand[3] = {0};
    uint64_t bb_pile[3] = {0};
    uint64_t bb_field = 0;
    uint64_t bb_stock = 0;
    uint64_t bb_show = 0;
    uint64_t bb_pairing = 0;
    
    // --- 新仕様トラッキング ---
    uint64_t bb_discard_hist[3] = {0};
    uint16_t suit_played[3] = {0};
    uint16_t suit_ignored[3] = {0};

    KoiKoiStateManager() {}

    void init_board(uint64_t hand1, uint64_t hand2, uint64_t field, uint64_t stock) {
        bb_hand[1] = hand1; bb_hand[2] = hand2;
        bb_field = field; bb_stock = stock;
        bb_pile[1] = 0; bb_pile[2] = 0;
        bb_show = 0; bb_pairing = 0;
        
        bb_discard_hist[1] = 0; bb_discard_hist[2] = 0;
        suit_played[1] = 0; suit_played[2] = 0;
        suit_ignored[1] = 0; suit_ignored[2] = 0;
    }

    void discard(int p, int card, uint64_t pairing, int turn_16) {
        bb_hand[p] &= ~(1ULL << card);
        bb_show = (1ULL << card);
        bb_pairing = pairing;
        
        // 信念状態トラッキングの更新
        bb_discard_hist[p] |= (1ULL << card);
        int suit = card / 4;
        suit_played[p] |= (1 << suit);
        
        uint16_t field_suits = 0;
        uint64_t f = bb_field;
        while (f) {
            field_suits |= (1 << (ctz64(f) / 4));
            f &= f - 1;
        }
        // 場札があるのにあえて別の月を出した場合、その月の札は持っていない可能性が高いと推論
        suit_ignored[p] |= (field_suits & ~(1 << suit));
    }

    void discard_pick(int p, uint64_t collect, uint64_t field_rem, int turn_16) {
        bb_pile[p] |= collect;
        bb_field = field_rem;
        bb_show = 0;
        bb_pairing = 0;
    }

    void draw(int card, uint64_t pairing, int turn_16) {
        bb_stock &= ~(1ULL << card);
        bb_show = (1ULL << card);
        bb_pairing = pairing;
    }

    void draw_pick(int p, uint64_t collect, uint64_t field_rem, int turn_16) {
        bb_pile[p] |= collect;
        bb_field = field_rem;
        bb_show = 0;
        bb_pairing = 0;
    }

    int get_yaku_point(int p, int koikoi_num) {
        return get_yaku_point_by_bitboard(bb_pile[p], koikoi_num);
    }
    
    std::vector<std::tuple<int, std::string, int>> get_yaku_list(int p, int koikoi_num) {
        return evaluate_yaku_by_bitboard(bb_pile[p], koikoi_num);
    }

    // 引数のシグネチャは Python (koikoigame.py) との互換性を保つためにそのままにします
    py::array_t<float> get_feature(
        bool is_koikoi, int point_turn, int point_idle,
        int round_num, int turn_16, int dealer,
        int koikoi_num_turn, int koikoi_num_idle,
        int turn_p, int idle_p,
        int max_round) 
    {
        // 新仕様 [24, 48]
        auto result = py::array_t<float>({24, 48});
        
        Snapshot snap;
        snap.bb_hand = bb_hand[turn_p];
        snap.bb_unseen = bb_stock | bb_hand[idle_p];
        snap.bb_my_pile = bb_pile[turn_p];
        snap.bb_field = bb_field;
        snap.bb_op_pile = bb_pile[idle_p];
        
        snap.bb_my_discard = bb_discard_hist[turn_p];
        snap.suit_op_played = suit_played[idle_p];
        snap.suit_op_ignored = suit_ignored[idle_p];
        
        snap.point_turn = point_turn;
        snap.point_idle = point_idle;
        snap.round_num = round_num;
        snap.turn_16 = turn_16;
        snap.is_dealer = (dealer == turn_p);
        snap.koikoi_num_turn = koikoi_num_turn;
        snap.koikoi_num_idle = koikoi_num_idle;
        snap.state_type = is_koikoi ? 2 : 0;

        // 推論時は action_to_align は -1
        write_feature_core(result.mutable_data(), snap, -1, max_round);
        return result;
    }

    py::array_t<bool> get_action_mask(int state_type, int p) {
        int size = (state_type == 2) ? 2 : 48;
        auto result = py::array_t<bool>(size);
        auto buf = result.mutable_unchecked<1>();
        
        if (state_type == 2) {
            buf(0) = true; buf(1) = true;
            return result;
        }

        uint64_t mask_bb = (state_type == 0) ? bb_hand[p] : bb_pairing;
        for (int i = 0; i < 48; ++i) {
            buf(i) = ((mask_bb >> i) & 1) != 0;
        }
        return result;
    }

    int get_best_action(int state_type, int p, py::array_t<float>& logits) {
        auto log_buf = logits.unchecked<1>();
        int best_idx = -1;
        float max_val = -std::numeric_limits<float>::infinity();

        if (state_type == 2) {
            return log_buf(1) > log_buf(0) ? 1 : 0;
        }

        uint64_t mask_bb = (state_type == 0) ? bb_hand[p] : bb_pairing;
        for (int i = 0; i < 48; ++i) {
            if ((mask_bb >> i) & 1) {
                float val = log_buf(i);
                if (val > max_val) {
                    max_val = val;
                    best_idx = i;
                }
            }
        }
        return best_idx;
    }
};

class KoiKoiTraceBuffer {
public:
    int capacity;
    int rows;
    int cols;
    int current_size;
    
    torch::Tensor states_tensor;
    torch::Tensor actions_tensor;
    torch::Tensor rewards_tensor;

    float* states_ptr;
    int64_t* actions_ptr;
    float* rewards_ptr;

    KoiKoiTraceBuffer(int cap, int r, int c) : capacity(cap), rows(r), cols(c), current_size(0) {
        auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32);
        auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64);
        
        states_tensor = torch::empty({capacity, rows, cols}, opt_f32);
        actions_tensor = torch::empty({capacity}, opt_i64);
        rewards_tensor = torch::empty({capacity}, opt_f32);

        states_ptr = states_tensor.data_ptr<float>();
        actions_ptr = actions_tensor.data_ptr<int64_t>();
        rewards_ptr = rewards_tensor.data_ptr<float>();
    }

    void clear() { current_size = 0; }

    void push_reconstructed(const Snapshot& snap, int action, float reward, int max_round) {
        int idx;
        #pragma omp atomic capture
        {
            idx = current_size;
            current_size++;
        }
        
        if (idx >= capacity) {
            #pragma omp atomic write
            current_size = capacity;
            return;
        }

        float* dest = states_ptr + idx * rows * cols;
        actions_ptr[idx] = action;
        rewards_ptr[idx] = reward;

        // Q-Net学習用に選ばれたアクションの列を0にスワップする
        write_feature_core(dest, snap, action, max_round);
    }

    py::dict finalize() {
        py::dict result;
        if (current_size == 0) {
            result["states"] = torch::empty({0, rows, cols});
            result["actions"] = torch::empty({0});
            result["rewards"] = torch::empty({0});
        } else {
            result["states"] = states_tensor.slice(0, 0, current_size).clone();
            result["actions"] = actions_tensor.slice(0, 0, current_size).clone();
            result["rewards"] = rewards_tensor.slice(0, 0, current_size).clone();
        }
        return result;
    }
};

enum class GameState { INIT, DISCARD, DISCARD_PICK, DRAW, DRAW_PICK, KOIKOI, ROUND_OVER, GAME_OVER };

class KoiKoiEnv {
public:
    int round_num, dealer, winner, turn_16, turn_point;
    int point[3];
    int koikoi[3][8];
    bool exhausted, wait_action;
    GameState state;

    KoiKoiStateManager state_manager;
    std::mt19937 rng;

    FixedVec<int, 48> hand[3];
    FixedVec<int, 48> field_slot;
    FixedVec<int, 48> stock;
    FixedVec<int, 48> pile[3];
    FixedVec<int, 4> show;
    FixedVec<int, 8> collect;

    KoiKoiEnv(int seed = 42) : rng(seed) { reset_game(); }

    void reset_game() {
        round_num = 1; point[1] = 30; point[2] = 30;
        // ★ 修正: 初期ディーラーの完全ランダム化
        std::uniform_int_distribution<int> dist(1, 2);
        dealer = dist(rng);
        winner = dealer;
        state = GameState::ROUND_OVER;
    }

    void reset_round() {
        if (winner != 0 && winner != -1) dealer = winner;
        turn_16 = 1;
        for (int p = 1; p <= 2; ++p) {
            pile[p].clear();
            for (int i = 0; i < 8; ++i) koikoi[p][i] = 0;
        }
        show.clear(); collect.clear();
        winner = 0; exhausted = false; turn_point = 0;

        deal_card();

        uint64_t bb_h1 = 0, bb_h2 = 0, bb_f = 0, bb_s = 0;
        for (int c : hand[1]) bb_h1 |= (1ULL << c);
        for (int c : hand[2]) bb_h2 |= (1ULL << c);
        for (int c : field_slot) if (c != -1) bb_f |= (1ULL << c);
        for (int c : stock) bb_s |= (1ULL << c);

        state_manager.init_board(bb_h1, bb_h2, bb_f, bb_s);
        state = GameState::DISCARD; wait_action = true;
    }

    int turn_player() const { return ((turn_16 + dealer) % 2 == 0) ? 1 : 2; }
    int idle_player() const { return 3 - turn_player(); }
    int turn_8() const { return (turn_16 + 1) / 2; }
    int koikoi_num(int p) const { int s = 0; for (int i = 0; i < 8; ++i) s += koikoi[p][i]; return s; }

    int round_point(int p) {
        if (winner == 0) return 0;
        if (exhausted) return (dealer == p) ? 1 : -1;
        int pt = state_manager.get_yaku_point(winner, koikoi_num(winner));
        return (winner == p) ? pt : -pt;
    }

    bool needs_action() const {
        return wait_action && (state == GameState::DISCARD || state == GameState::DISCARD_PICK ||
                               state == GameState::DRAW_PICK || state == GameState::KOIKOI);
    }

    FixedVec<int, 4> get_pairing_cards() const {
        FixedVec<int, 4> pairs;
        if (show.empty()) return pairs;
        int target = show[0] / 4;
        for (int c : field_slot) if (c != -1 && c / 4 == target) pairs.push_back(c);
        return pairs;
    }
    
    void remove_from_field(int card) {
        for (int i = 0; i < field_slot.size(); ++i) {
            if (field_slot[i] == card) {
                field_slot[i] = -1;
                break;
            }
        }
    }

    void _collect_card(int card) {
        FixedVec<int, 4> pairing = get_pairing_cards();
        collect.clear();
        if (pairing.empty()) {
            for (int i = 0; i < field_slot.size(); ++i) {
                if (field_slot[i] == -1) {
                    field_slot[i] = show[0];
                    break;
                }
            }
        } else if (pairing.size() == 1 || pairing.size() == 3) {
            collect.insert_end(show.begin(), show.end());
            collect.insert_end(pairing.begin(), pairing.end());
            for (int p_card : pairing) remove_from_field(p_card);
            pile[turn_player()].insert_end(collect.begin(), collect.end());
        } else {
            collect.insert_end(show.begin(), show.end());
            collect.push_back(card);
            remove_from_field(card);
            pile[turn_player()].insert_end(collect.begin(), collect.end());
        }
    }

    void step(int action) {
        int p = turn_player();
        switch (state) {
            case GameState::DISCARD:      handle_discard(p, action); break;
            case GameState::DISCARD_PICK: handle_discard_pick(p, action); break;
            case GameState::DRAW:         handle_draw(p); break;
            case GameState::DRAW_PICK:    handle_draw_pick(p, action); break;
            case GameState::KOIKOI:       handle_koikoi(p, action); break;
            default: break;
        }
    }

private:
    void handle_discard(int p, int action) {
        turn_point = state_manager.get_yaku_point(p, koikoi_num(p));
        for (auto it = hand[p].begin(); it != hand[p].end(); ++it) {
            if (*it == action) {
                hand[p].erase(it);
                break;
            }
        }
        show.clear();
        show.push_back(action);

        FixedVec<int, 4> pairing = get_pairing_cards();
        uint64_t pair_bb = 0; for (int c : pairing) pair_bb |= (1ULL << c);
        
        state_manager.discard(p, action, pair_bb, turn_16);

        state = GameState::DISCARD_PICK; wait_action = (pairing.size() == 2);
    }

    void handle_discard_pick(int p, int action) {
        _collect_card(action);
        uint64_t coll_bb = 0; for (int c : collect) coll_bb |= (1ULL << c);
        uint64_t field_bb = 0; for (int c : field_slot) if (c != -1) field_bb |= (1ULL << c);
        
        state_manager.discard_pick(p, coll_bb, field_bb, turn_16);
        state = GameState::DRAW; wait_action = false;
    }

    void handle_draw(int p) {
        int drawn = stock.back(); stock.pop_back(); 
        show.clear();
        show.push_back(drawn);
        
        FixedVec<int, 4> pairing = get_pairing_cards();
        uint64_t pair_bb = 0; for (int c : pairing) pair_bb |= (1ULL << c);
        
        state_manager.draw(drawn, pair_bb, turn_16);
        state = GameState::DRAW_PICK; wait_action = (pairing.size() == 2);
    }

    void handle_draw_pick(int p, int action) {
        _collect_card(action);
        uint64_t coll_bb = 0; for (int c : collect) coll_bb |= (1ULL << c);
        uint64_t field_bb = 0; for (int c : field_slot) if (c != -1) field_bb |= (1ULL << c);
        
        state_manager.draw_pick(p, coll_bb, field_bb, turn_16);
        
        state = GameState::KOIKOI;
        int pt = state_manager.get_yaku_point(p, koikoi_num(p));
        wait_action = (pt > turn_point) && (turn_8() < 8);
    }

    void handle_koikoi(int p, int action) {
        int pt = state_manager.get_yaku_point(p, koikoi_num(p));
        bool is_koikoi = (action != 0);
        if ((pt > turn_point) && (turn_8() == 8)) is_koikoi = false;
        if (wait_action) koikoi[p][turn_8() - 1] = is_koikoi ? 1 : 0;

        if (!is_koikoi) { state = GameState::ROUND_OVER; wait_action = false; winner = p; }
        else if (turn_16 == 16) { state = GameState::ROUND_OVER; wait_action = false; exhausted = true; winner = dealer; }
        else { turn_16++; state = GameState::DISCARD; wait_action = true; }
    }

    void deal_card() {
        int cards[48];
        while (true) {
            for (int i = 0; i < 48; ++i) cards[i] = i;
            std::shuffle(cards, cards + 48, rng);
            
            hand[1].clear(); hand[2].clear(); field_slot.clear(); stock.clear();
            for (int i = 0; i < 8; ++i) hand[1].push_back(cards[i]);
            std::sort(hand[1].begin(), hand[1].end());
            
            for (int i = 8; i < 16; ++i) hand[2].push_back(cards[i]);
            std::sort(hand[2].begin(), hand[2].end());
            
            int init_f[8];
            for (int i = 16; i < 24; ++i) init_f[i - 16] = cards[i];
            std::sort(init_f, init_f + 8);
            for (int i = 0; i < 8; ++i) field_slot.push_back(init_f[i]);
            for (int i = 0; i < 8; ++i) field_slot.push_back(-1); 
            
            for (int i = 24; i < 48; ++i) stock.push_back(cards[i]);

            bool flag = true;
            for (int s = 0; s < 12; ++s) {
                int c1 = 0, c2 = 0, cf = 0;
                for (int c : hand[1]) if (c / 4 == s) c1++;
                for (int c : hand[2]) if (c / 4 == s) c2++;
                for (int i = 0; i < 8; ++i) if (init_f[i] / 4 == s) cf++;
                if (c1 == 4 || c2 == 4 || cf == 4) { flag = false; break; }
            }
            if (flag) break;
        }
    }
};

class BatchSimulator {
public: 
    int num_envs, target_games;
    float discount;
    std::vector<KoiKoiEnv> envs;
    std::vector<FixedVec<TraceSlot, 64>> traces[3];

    KoiKoiTraceBuffer buf_discard, buf_pick, buf_koikoi;
    const std::vector<float>& win_prob_mat;

    int finished_games = 0;
    torch::Device device;

    torch::Tensor feat_tensor[3];
    torch::Tensor mask_tensor[3];

    BatchSimulator(int n_envs, int target, int cap_d, int cap_p, int cap_k, float disc, const std::vector<float>& wp_mat, std::string dev_str)
        : num_envs(n_envs), target_games(target), discount(disc),
          // ★ 新仕様 [24, 48] に完全統一
          buf_discard(cap_d, 24, 48), buf_pick(cap_p, 24, 48), buf_koikoi(cap_k, 24, 48),
          win_prob_mat(wp_mat), device(dev_str)
    {
        envs.reserve(n_envs);
        std::random_device rd;
        for(int i = 0; i < n_envs; ++i) {
            envs.emplace_back(rd());
        }

        for(int p=1; p<=2; ++p) {
            traces[p].resize(n_envs);
        }

        auto opt_f32_pinned = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        auto opt_bool_pinned = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);

        feat_tensor[0] = torch::empty({num_envs, 24, 48}, opt_f32_pinned);
        feat_tensor[1] = torch::empty({num_envs, 24, 48}, opt_f32_pinned);
        feat_tensor[2] = torch::empty({num_envs, 24, 48}, opt_f32_pinned);

        mask_tensor[0] = torch::empty({num_envs, 48}, opt_bool_pinned);
        mask_tensor[1] = torch::empty({num_envs, 48}, opt_bool_pinned);
        mask_tensor[2] = torch::empty({num_envs, 2}, opt_bool_pinned);
    }

    void reset() {
        finished_games = 0;
        for (auto& env : envs) {
            env.reset_game();
            env.reset_round();
        }
        for (int p = 1; p <= 2; ++p) {
            for (auto& trace_vec : traces[p]) {
                trace_vec.clear();
            }
        }
        buf_discard.clear();
        buf_pick.clear();
        buf_koikoi.clear();
    }

    py::dict finalize_buffers() {
        py::dict res;
        res["discard"] = buf_discard.finalize();
        res["pick"] = buf_pick.finalize();
        res["koikoi"] = buf_koikoi.finalize();
        return res;
    }

    void build_feature_into(int i, float* feat, int type, int max_round) {
        auto& env = envs[i]; 
        int pt = env.turn_player(); 
        int pi = env.idle_player();
        Snapshot snap = capture_snapshot(env, pt, pi, type, max_round);
        write_feature_core(feat, snap, -1, max_round);
    }

    void build_mask_into(int i, bool* mk, int type) {
        auto& env = envs[i]; 
        int sz = (type == 2) ? 2 : 48;
        std::memset(mk, 0, sz * sizeof(bool));
        if (type == 2) { mk[0] = true; mk[1] = true; }
        else {
            uint64_t bb = (type == 0) ? env.state_manager.bb_hand[env.turn_player()] : env.state_manager.bb_pairing;
            for(int c=0; c<48; ++c) if((bb>>c)&1) mk[c] = true;
        }
    }

    float get_potential(int p_idx, int n_rnd, int n_pt, int dealer, int winner, bool exhausted, int max_round) {
        if (n_rnd > max_round || n_pt <= 0 || n_pt >= 60) {
            if (n_pt <= 0) return 0.0f;       
            if (n_pt >= 60) return 1.0f;      
            if (n_rnd > max_round) {
                if (n_pt == 30) return 0.5f;  
                return (n_pt > 30) ? 1.0f : 0.0f; 
            }
        }
        int mapped_rnd = n_rnd + (8 - max_round);
        int is_d = (exhausted ? dealer : winner) == p_idx ? 1 : 0;
        return win_prob_mat[(is_d * 9 + mapped_rnd) * 61 + n_pt];
    }

    void process_round_over(int i, int max_round) {
        auto& env = envs[i];
        for (int p_id = 1; p_id <= 2; ++p_id) {
            auto& trace = traces[p_id][i];
            if (trace.empty()) continue;

            int final_pt = env.point[p_id] + env.round_point(p_id);
            float final_potential = get_potential(p_id, env.round_num + 1, final_pt, env.dealer, env.winner, env.exhausted, max_round);

            for (size_t rs = 0; rs < trace.size(); ++rs) {
                int current_idx = static_cast<int>(trace.size() - 1 - rs);
                auto& slot = trace[current_idx];
                
                float reward = (current_idx == static_cast<int>(trace.size() - 1)) ? (final_potential * 10.0f) : 0.0f;
                float next_q = (current_idx == static_cast<int>(trace.size() - 1)) ? 0.0f : trace[current_idx + 1].q_val;
                float target_q = reward + discount * next_q;

                if (slot.state_type == 0) buf_discard.push_reconstructed(slot.snap, slot.action, target_q, max_round);
                else if (slot.state_type == 1) buf_pick.push_reconstructed(slot.snap, slot.action, target_q, max_round);
                else buf_koikoi.push_reconstructed(slot.snap, slot.action, target_q, max_round);
            }
            trace.clear();
        }
        env.point[1] += env.round_point(1); env.point[2] += env.round_point(2);
    }

    void play_games(int max_round, int current_loop) {
        std::vector<int> req_env_idx[3];
        float* feat_ptrs[3] = { feat_tensor[0].data_ptr<float>(), feat_tensor[1].data_ptr<float>(), feat_tensor[2].data_ptr<float>() };
        bool* mask_ptrs[3] = { mask_tensor[0].data_ptr<bool>(), mask_tensor[1].data_ptr<bool>(), mask_tensor[2].data_ptr<bool>() };

        while (finished_games < target_games) {
            step_environments_without_action(max_round);

            if (finished_games >= target_games) break;

            for (int p_id = 1; p_id <= 2; ++p_id) {
                for(int i=0; i<3; ++i) req_env_idx[i].clear();
                
                for (int i = 0; i < num_envs; ++i) {
                    auto& env = envs[i];
                    if (env.needs_action() && env.turn_player() == p_id) {
                        int type = (env.state == GameState::DISCARD) ? 0 : (env.state == GameState::KOIKOI ? 2 : 1);
                        req_env_idx[type].push_back(i);
                    }
                }

                torch::NoGradGuard no_grad; 
                for (int type = 0; type < 3; ++type) {
                    int n = static_cast<int>(req_env_idx[type].size());
                    if (n == 0) continue;

                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) {
                        int env_idx = req_env_idx[type][i];
                        int m_cols = (type == 2) ? 2 : 48;
                        build_feature_into(env_idx, feat_ptrs[type] + i * 24 * 48, type, max_round);
                        build_mask_into(env_idx, mask_ptrs[type] + i * m_cols, type);
                    }

                    torch::Tensor f_gpu = feat_tensor[type].slice(0, 0, n).to(device, true).to(torch::kFloat32);
                    
                    torch::Tensor output;
                    if (type == 0) output = g_models.discard.forward({f_gpu}).toTensor();
                    else if (type == 1) output = g_models.pick.forward({f_gpu}).toTensor();
                    else output = g_models.koikoi.forward({f_gpu}).toTensor();

                    torch::Tensor output_cpu = output.cpu();
                    const float* out_ptr = output_cpu.data_ptr<float>();
                    const bool* mask_ptr = mask_tensor[type].data_ptr<bool>(); 

                    int64_t act_ptr[8192]; 
                    float q_val_ptr[8192];

                    select_actions_epsilon_greedy(type, n, out_ptr, mask_ptr, act_ptr, q_val_ptr, current_loop);
                    apply_actions_and_record(p_id, type, req_env_idx[type], act_ptr, q_val_ptr, max_round);
                }
            }
        }
    }

private:
    Snapshot capture_snapshot(const KoiKoiEnv& env, int pt, int pi, int type, int max_round) {
        Snapshot snap;
        snap.bb_hand = env.state_manager.bb_hand[pt];
        snap.bb_unseen = env.state_manager.bb_stock | env.state_manager.bb_hand[pi];
        snap.bb_my_pile = env.state_manager.bb_pile[pt];
        snap.bb_field = env.state_manager.bb_field;
        snap.bb_op_pile = env.state_manager.bb_pile[pi];
        
        snap.bb_my_discard = env.state_manager.bb_discard_hist[pt];
        snap.suit_op_played = env.state_manager.suit_played[pi];
        snap.suit_op_ignored = env.state_manager.suit_ignored[pi];
        
        snap.point_turn = env.point[pt];
        snap.point_idle = env.point[pi];
        snap.round_num = env.round_num;
        snap.turn_16 = env.turn_16;
        snap.is_dealer = (env.dealer == pt);
        snap.koikoi_num_turn = env.koikoi_num(pt);
        snap.koikoi_num_idle = env.koikoi_num(pi);
        snap.state_type = type;
        return snap;
    }

    void step_environments_without_action(int max_round) {
        int local_finished = 0;
        #pragma omp parallel for reduction(+:local_finished)
        for (int i = 0; i < num_envs; ++i) {
            auto& env = envs[i];
            while (!env.needs_action() && env.state != GameState::GAME_OVER) {
                if (env.state == GameState::ROUND_OVER) {
                    process_round_over(i, max_round);
                    if (env.point[1] <= 0 || env.point[2] <= 0 || env.round_num == 8) {
                        local_finished++;
                        env.reset_game(); env.reset_round();
                    } else { env.round_num++; env.reset_round(); }
                } else { env.step(-1); }
            }
        }
        finished_games += local_finished;
    }

    void select_actions_epsilon_greedy(int type, int n, const float* out_ptr, const bool* mask_ptr, int64_t* act_ptr, float* q_val_ptr, int current_loop) {
        float epsilon = 0.05f + 0.95f * std::exp(-static_cast<float>(current_loop) / 200.0f);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        int cols = (type == 2) ? 2 : 48;

        for (int i = 0; i < n; ++i) {
            if (dist(envs[0].rng) < epsilon) {
                FixedVec<int, 48> valid_actions;
                for (int c = 0; c < cols; ++c) {
                    if (mask_ptr[i * cols + c]) valid_actions.push_back(c);
                }
                if (!valid_actions.empty()) {
                    std::uniform_int_distribution<int> action_dist(0, valid_actions.size() - 1);
                    act_ptr[i] = valid_actions[action_dist(envs[0].rng)];
                    q_val_ptr[i] = out_ptr[i * cols + act_ptr[i]]; 
                } else {
                    act_ptr[i] = 0; q_val_ptr[i] = 0.0f;
                }
            } else {
                float max_val = -1e9f;
                int best_a = 0;
                for (int c = 0; c < cols; ++c) {
                    if (mask_ptr[i * cols + c]) {
                        float val = out_ptr[i * cols + c];
                        if (val > max_val) {
                            max_val = val; best_a = c;
                        }
                    }
                }
                act_ptr[i] = best_a; q_val_ptr[i] = max_val;
            }
        }
    }

    void apply_actions_and_record(int p_id, int type, const std::vector<int>& req_idx, const int64_t* act_ptr, const float* q_val_ptr, int max_round) {
        int n = static_cast<int>(req_idx.size());
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            int env_idx = req_idx[i];
            int action = static_cast<int>(act_ptr[i]);
            auto& env = envs[env_idx];

            TraceSlot slot;
            slot.state_type = type;
            slot.action = action;
            slot.q_val = q_val_ptr[i];
            int idle_id = (p_id == 1) ? 2 : 1;
            slot.snap = capture_snapshot(env, p_id, idle_id, type, max_round);
            traces[p_id][env_idx].push_back(slot);

            env.step(action);
        }
    }
};

class KoiKoiTrainer {
private:
    torch::jit::script::Module model;
    std::shared_ptr<torch::optim::Adam> optimizer;
    torch::Device device;

    torch::Tensor batch_states[2];
    torch::Tensor batch_rewards[2];
    int64_t micro_batch_size = 1024;
    int current_cols = 0;

    void copy_parameters(torch::jit::script::Module& src, torch::jit::script::Module& dst) {
        auto src_params = src.parameters();
        auto dst_params = dst.parameters();
        auto src_it = src_params.begin();
        auto dst_it = dst_params.begin();
        while (src_it != src_params.end() && dst_it != dst_params.end()) {
            (*dst_it).copy_(*src_it);
            ++src_it;
            ++dst_it;
        }
    }

public:
    KoiKoiTrainer(torch::jit::Module py_model, float lr, const std::string& dev_str) 
        : device(dev_str) {
        
        model = py_model;
        model.train();
        
        std::vector<torch::Tensor> parameters;
        for (const auto& p : model.parameters()) {
            if (p.requires_grad()) {
                parameters.push_back(p);
            }
        }
        optimizer = std::make_shared<torch::optim::Adam>(
            parameters, torch::optim::AdamOptions(lr));
    }

    float train_epoch(torch::Tensor states, torch::Tensor rewards, int batch_size) {
        
        states = states.contiguous();
        rewards = rewards.contiguous();
        
        int64_t num_samples = states.size(0);
        int rows = states.size(1);
        int cols = states.size(2);
        
        if (current_cols != cols || !batch_states[0].defined()) {
            auto opt_pinned = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
            batch_states[0] = torch::empty({micro_batch_size, rows, cols}, opt_pinned);
            batch_states[1] = torch::empty({micro_batch_size, rows, cols}, opt_pinned);
            batch_rewards[0] = torch::empty({micro_batch_size}, opt_pinned);
            batch_rewards[1] = torch::empty({micro_batch_size}, opt_pinned);
            current_cols = cols;
        }

        auto indices = torch::randperm(num_samples, torch::TensorOptions().dtype(torch::kLong));
        const int64_t* idx_ptr = indices.data_ptr<int64_t>();
        
        float total_loss = 0.0f;
        int num_batches = 0;
        
        const float* src_states_ptr = states.data_ptr<float>();
        const float* src_rewards_ptr = rewards.data_ptr<float>();

        py::gil_scoped_release release;
        int buf_idx = 0;

        for (int64_t i = 0; i < num_samples; i += batch_size) {
            int64_t current_batch_size = std::min(static_cast<int64_t>(batch_size), num_samples - i);
            optimizer->zero_grad();
            
            float batch_loss = 0.0f;

            for (int64_t j = 0; j < current_batch_size; j += micro_batch_size) {
                int64_t m_size = std::min(micro_batch_size, current_batch_size - j);
                
                float* dst_states_ptr = batch_states[buf_idx].data_ptr<float>();
                float* dst_rewards_ptr = batch_rewards[buf_idx].data_ptr<float>();

                for (int64_t b = 0; b < m_size; ++b) {
                    int64_t src_idx = idx_ptr[i + j + b];
                    std::memcpy(dst_states_ptr + b * rows * cols, 
                                src_states_ptr + src_idx * rows * cols, 
                                rows * cols * sizeof(float));
                    dst_rewards_ptr[b] = src_rewards_ptr[src_idx];
                }

                auto state_gpu = batch_states[buf_idx].slice(0, 0, m_size).to(torch::kFloat32).to(device, true);
                auto reward_gpu = batch_rewards[buf_idx].slice(0, 0, m_size).to(device, true);

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(state_gpu);
                
                torch::Tensor q_values = model.forward(inputs).toTensor().view({-1});
                torch::Tensor reward_flat = reward_gpu.view({-1});
                
                auto loss = torch::nn::functional::smooth_l1_loss(
                    q_values,
                    reward_flat, 
                    torch::nn::functional::SmoothL1LossFuncOptions().beta(30.0)
                );

                float scale = static_cast<float>(m_size) / current_batch_size;
                auto scaled_loss = loss * scale;
                scaled_loss.backward();

                batch_loss += loss.item<float>() * scale;
                buf_idx = 1 - buf_idx;
            }

            optimizer->step();
            total_loss += batch_loss;
            num_batches++;
        }
        
        return num_batches > 0 ? total_loss / num_batches : 0.0f;
    }

    void sync_and_save_action_model(const std::string& action_model_path) {
        try {
            auto action_model = torch::jit::load(action_model_path, device);
            torch::NoGradGuard no_grad;
            copy_parameters(model, action_model);
            action_model.save(action_model_path);
        } catch (const c10::Error& e) {
            (void)e;
            std::cerr << "Error syncing weights to " << action_model_path << "\n";
        }
    }

    void sync_to_inference_model(const std::string& type) {
        torch::NoGradGuard no_grad;
        std::lock_guard<std::mutex> lock(g_models.mtx);
        
        torch::jit::script::Module* target_model = nullptr;
        if (type == "discard") target_model = &g_models.discard;
        else if (type == "pick") target_model = &g_models.pick;
        else if (type == "koikoi") target_model = &g_models.koikoi;
        
        if (target_model && g_models.loaded) {
            copy_parameters(model, *target_model);
        }
    }
};

struct SimConfig {
    int num_threads, n_envs, target, cap_d, cap_p, cap_k;
    float disc;
    std::string dev_str;
    int max_round;

    bool operator==(const SimConfig& o) const {
        return num_threads == o.num_threads && n_envs == o.n_envs &&
               target == o.target && cap_d == o.cap_d && cap_p == o.cap_p &&
               cap_k == o.cap_k && disc == o.disc && dev_str == o.dev_str && max_round == o.max_round;
    }
    bool operator!=(const SimConfig& o) const { return !(*this == o); }
};

class SimulationManager {
public:
    std::vector<std::unique_ptr<BatchSimulator>> sims;
    int current_loop_exec = 0;
    SimConfig current_config;

    std::vector<std::thread> workers;
    std::mutex pool_mtx;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    
    int generation = 0;
    bool shutdown = false;
    int done_workers = 0;
    
    std::vector<std::exception_ptr> worker_exceptions;
    std::vector<float> shared_win_prob_mat;

    SimulationManager(const SimConfig& config, py::array_t<float>& wp_mat) 
        : current_config(config), worker_exceptions(config.num_threads, nullptr) 
    {
        auto wp = wp_mat.unchecked<3>();
        shared_win_prob_mat.resize(2 * 9 * 61, 0.0f);
        for(int i=0; i<2; ++i) for(int j=0; j<9; ++j) for(int k=0; k<61; ++k)
            shared_win_prob_mat[(i*9+j)*61+k] = static_cast<float>(wp(i, j, k));

        for (int i = 0; i < config.num_threads; ++i) {
            sims.push_back(std::make_unique<BatchSimulator>(
                config.n_envs, config.target, config.cap_d, config.cap_p, config.cap_k, 
                config.disc, shared_win_prob_mat, config.dev_str
            ));
        }

        for (int i = 0; i < config.num_threads; ++i) {
            workers.emplace_back(&SimulationManager::worker_loop, this, i);
        }
    }

    ~SimulationManager() {
        {
            std::lock_guard<std::mutex> lock(pool_mtx);
            shutdown = true;
        }
        cv_start.notify_all(); 
        
        for (auto& t : workers) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    void reset_all() {
        for (auto& sim : sims) {
            sim->reset();
        }
    }

    void run_simulation_parallel(int loop_num) {
        {
            std::lock_guard<std::mutex> lock(pool_mtx);
            current_loop_exec = loop_num;
            std::fill(worker_exceptions.begin(), worker_exceptions.end(), nullptr);
            done_workers = 0;
            generation++; 
        }
        cv_start.notify_all();

        {
            std::unique_lock<std::mutex> lock(pool_mtx);
            cv_done.wait(lock, [this]() { return this->done_workers == this->current_config.num_threads; });
        }

        for (auto& e : worker_exceptions) {
            if (e) {
                std::rethrow_exception(e);
            }
        }
    }

private:
    void worker_loop(int worker_id) {
        int local_generation = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(pool_mtx);
                cv_start.wait(lock, [this, local_generation]() { 
                    return generation > local_generation || shutdown; 
                });
                
                if (shutdown && generation <= local_generation) {
                    break;
                }
                local_generation = generation;
            }

            try {
                sims[worker_id]->play_games(current_config.max_round, current_loop_exec);
            } catch (...) {
                std::lock_guard<std::mutex> lock(pool_mtx);
                worker_exceptions[worker_id] = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(pool_mtx);
                done_workers++;
                if (done_workers == current_config.num_threads) {
                    cv_done.notify_one();
                }
            }
        }
    }
};

static std::unique_ptr<SimulationManager> g_sim_manager = nullptr;

py::list run_parallel_simulations(
    int num_threads, int n_envs_per_thread, int target_games_per_thread,
    int cap_d, int cap_p, int cap_k, float disc, py::array_t<float> wp_mat,
    torch::jit::Module discard_model, torch::jit::Module pick_model, torch::jit::Module koikoi_model,
    std::string dev_str, int max_round, int current_loop) 
{
    torch::Device device(dev_str);

    {
        std::lock_guard<std::mutex> lock(g_models.mtx);
        if (!g_models.loaded) {
            g_models.discard = discard_model;
            g_models.pick = pick_model;
            g_models.koikoi = koikoi_model;
            g_models.discard.eval(); 
            g_models.pick.eval(); 
            g_models.koikoi.eval();
            g_models.loaded = true;
        }
    }

    SimConfig config = {num_threads, n_envs_per_thread, target_games_per_thread, cap_d, cap_p, cap_k, disc, dev_str, max_round};
    
    if (g_sim_manager && g_sim_manager->current_config != config) {
        g_sim_manager.reset();
    }

    if (!g_sim_manager) {
        g_sim_manager = std::make_unique<SimulationManager>(config, wp_mat);
    } else {
        g_sim_manager->reset_all();
    }

    {
        py::gil_scoped_release release;
        g_sim_manager->run_simulation_parallel(current_loop);
    }

    py::list results;
    for (int i = 0; i < num_threads; ++i) {
        results.append(g_sim_manager->sims[i]->finalize_buffers());
    }
    return results;
}

PYBIND11_MODULE(koikoicore, m) {
    m.doc() = "C++ Core Engine with Bitboards for KoiKoi AI";
    
    m.def("get_yaku_point_by_bitboard", &get_yaku_point_by_bitboard);
    m.def("evaluate_yaku_by_bitboard", &evaluate_yaku_by_bitboard);
    m.def("run_parallel_simulations", &run_parallel_simulations);
    m.def("destroy_sim_manager", []() {
        if (g_sim_manager) {
            g_sim_manager.reset();
            std::cout << "[C++] SimulationManager and Pinned Tensors safely destroyed.\n";
        }
    });

    py::class_<KoiKoiStateManager>(m, "KoiKoiStateManager")
        .def(py::init<>())
        .def("init_board", &KoiKoiStateManager::init_board)
        .def("discard", &KoiKoiStateManager::discard)
        .def("discard_pick", &KoiKoiStateManager::discard_pick)
        .def("draw", &KoiKoiStateManager::draw)
        .def("draw_pick", &KoiKoiStateManager::draw_pick)
        .def("get_yaku_point", &KoiKoiStateManager::get_yaku_point)
        .def("get_yaku_list", &KoiKoiStateManager::get_yaku_list)
        .def("get_feature", &KoiKoiStateManager::get_feature)
        .def("get_action_mask", &KoiKoiStateManager::get_action_mask)
        .def("get_best_action", &KoiKoiStateManager::get_best_action);

    py::class_<KoiKoiTraceBuffer>(m, "KoiKoiTraceBuffer")
        .def(py::init<int, int, int>(), py::arg("capacity"), py::arg("rows"), py::arg("cols"))
        .def("clear", &KoiKoiTraceBuffer::clear)
        .def("finalize", &KoiKoiTraceBuffer::finalize);

    py::class_<BatchSimulator>(m, "BatchSimulator")
        .def(py::init<int, int, int, int, int, float, const std::vector<float>&, std::string>())
        .def("play_games", &BatchSimulator::play_games)
        .def("finalize_buffers", &BatchSimulator::finalize_buffers);

    py::class_<KoiKoiTrainer>(m, "KoiKoiTrainer")
        .def(py::init<torch::jit::Module, float, std::string>())
        .def("train_epoch", &KoiKoiTrainer::train_epoch)
        .def("sync_and_save_action_model", &KoiKoiTrainer::sync_and_save_action_model)
        .def("sync_to_inference_model", &KoiKoiTrainer::sync_to_inference_model);
}