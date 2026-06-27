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

// Design Intent: マジックナンバーを排除し、盤面チャネル仕様を一元管理する
constexpr int NUM_CARDS = 48;
constexpr int NUM_FEAT_ROWS = 24;

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

        const int seeds[][2] = {{2,1},{4,1},{5,1},{6,1},{7,1},{8,2},{9,1},{10,1},{11,2}};
        for (const auto& s : seeds) add(seed, s[0], s[1]);

        const int ribbons[][2] = {{1,2},{2,2},{3,2},{4,2},{5,2},{6,2},{7,2},{9,2},{10,2},{11,3}};
        for (const auto& r : ribbons) add(ribbon, r[0], r[1]);

        const int dross_list[][2] = {
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

struct KoiKoiRuleConfig {
    int five_lights = 10;
    int four_lights = 8;
    int rainy_four_lights = 7;
    int three_lights = 5;
    int bdb = 5;
    bool enable_flower_sake = true;
    bool enable_moon_sake = true;
    int flower_moon_base_pt = 1;
    int flower_moon_koikoi_pt = 3;
    int red_blue_ribbon = 10;
    int red_ribbon = 5;
    int blue_ribbon = 5;
};
static KoiKoiRuleConfig g_rules;

void set_rules(py::dict rules) {
    if (rules.contains("five_lights")) g_rules.five_lights = rules["five_lights"].cast<int>();
    if (rules.contains("four_lights")) g_rules.four_lights = rules["four_lights"].cast<int>();
    if (rules.contains("rainy_four_lights")) g_rules.rainy_four_lights = rules["rainy_four_lights"].cast<int>();
    if (rules.contains("three_lights")) g_rules.three_lights = rules["three_lights"].cast<int>();
    if (rules.contains("bdb")) g_rules.bdb = rules["bdb"].cast<int>();
    if (rules.contains("enable_flower_sake")) g_rules.enable_flower_sake = rules["enable_flower_sake"].cast<bool>();
    if (rules.contains("enable_moon_sake")) g_rules.enable_moon_sake = rules["enable_moon_sake"].cast<bool>();
    if (rules.contains("flower_moon_base_pt")) g_rules.flower_moon_base_pt = rules["flower_moon_base_pt"].cast<int>();
    if (rules.contains("flower_moon_koikoi_pt")) g_rules.flower_moon_koikoi_pt = rules["flower_moon_koikoi_pt"].cast<int>();
    if (rules.contains("red_blue_ribbon")) g_rules.red_blue_ribbon = rules["red_blue_ribbon"].cast<int>();
    if (rules.contains("red_ribbon")) g_rules.red_ribbon = rules["red_ribbon"].cast<int>();
    if (rules.contains("blue_ribbon")) g_rules.blue_ribbon = rules["blue_ribbon"].cast<int>();
}

template<typename T, int Capacity>
struct FixedVec {
    std::array<T, Capacity> data_{}; // Design Intent: 生配列からstd::arrayへ変更し安全なイテレータを確保
    int size_ = 0;

    void clear() { size_ = 0; }
    
    void push_back(const T& val) { 
        if (size_ < Capacity) [[likely]] {
            data_[size_++] = val; 
        } else {
            // OpenMPスレッド内でのthrow即死を回避するため、上限到達時は安全に無視する
            return; 
        }
    }
    
    void pop_back() { if (size_ > 0) size_--; }
    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }
    int size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T& operator[](int i) { return data_[i]; }
    const T& operator[](int i) const { return data_[i]; }

    T* begin() { return data_.data(); }
    T* end() { return data_.data() + size_; }
    const T* begin() const { return data_.data(); }
    const T* end() const { return data_.data() + size_; }

    void erase(T* it) {
        const int idx = static_cast<int>(it - data_.data());
        for (int i = idx; i < size_ - 1; ++i) {
            data_[i] = data_[i + 1];
        }
        size_--;
    }

    void insert_end(const T* first, const T* last) {
        const int n = static_cast<int>(last - first);
        if (size_ + n > Capacity) {
            throw std::out_of_range("FixedVec insert_end capacity exceeded"); // Validation
        }
        for (int i = 0; i < n; ++i) {
            data_[size_++] = first[i];
        }
    }
};

// --- ビット演算ヘルパー関数 ---
inline int popcount(uint64_t bb) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bb);
#elif defined(_MSC_VER)
    return static_cast<int>(__popcnt64(bb));
#else
    bb =           bb - ((bb >> 1) & 0x5555555555555555ULL);
    bb = (bb & 0x3333333333333333ULL) + ((bb >> 2) & 0x3333333333333333ULL);
    return static_cast<int>((((bb + (bb >> 4)) & 0xF0F0F0F0F0F0F0FULL) * 0x101010101010101ULL) >> 56);
#endif
}

inline int ctz64(uint64_t bb) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bb);
#elif defined(_MSC_VER)
    unsigned long c; _BitScanForward64(&c, bb); return static_cast<int>(c);
#else
    int c = 0; while ((bb & 1) == 0) { bb >>= 1; c++; } return c;
#endif
}
// ----------------------------

int get_yaku_point_by_bitboard(uint64_t pile, int koikoi_num) {
    int total_point = 0;
    const int num_light = popcount(pile & MASKS.light);
    if (num_light == 5) total_point += g_rules.five_lights;
    else if (num_light == 4 && (pile & MASKS.rainman) == 0) total_point += g_rules.four_lights;
    else if (num_light == 4) total_point += g_rules.rainy_four_lights;
    else if (num_light == 3 && (pile & MASKS.rainman) == 0) total_point += g_rules.three_lights;

    const int num_seed = popcount(pile & MASKS.seed);
    if ((pile & MASKS.boar_deer_butterfly) == MASKS.boar_deer_butterfly) total_point += g_rules.bdb;
    
    // 花見酒・月見酒の有効/無効判定と得点
    const int sake_pt = (koikoi_num == 0 ? g_rules.flower_moon_base_pt : g_rules.flower_moon_koikoi_pt);
    if (g_rules.enable_flower_sake && (pile & MASKS.flower_sake) == MASKS.flower_sake) total_point += sake_pt;
    if (g_rules.enable_moon_sake && (pile & MASKS.moon_sake) == MASKS.moon_sake) total_point += sake_pt;
    
    if (num_seed >= 5) total_point += (num_seed - 4);

    const int num_ribbon = popcount(pile & MASKS.ribbon);
    if ((pile & MASKS.red_blue_ribbon) == MASKS.red_blue_ribbon) total_point += g_rules.red_blue_ribbon;
    if ((pile & MASKS.red_ribbon) == MASKS.red_ribbon) total_point += g_rules.red_ribbon;
    if ((pile & MASKS.blue_ribbon) == MASKS.blue_ribbon) total_point += g_rules.blue_ribbon;
    if (num_ribbon >= 5) total_point += (num_ribbon - 4);

    const int num_dross = popcount(pile & MASKS.dross);
    if (num_dross >= 10) total_point += (num_dross - 9);

    if (koikoi_num <= 3) total_point += koikoi_num;
    else total_point *= (koikoi_num - 2);

    return total_point;
}

std::vector<std::tuple<int, std::string, int>> evaluate_yaku_by_bitboard(uint64_t pile, int koikoi_num) {
    std::vector<std::tuple<int, std::string, int>> yaku_list;
    const int num_light = popcount(pile & MASKS.light);
    if (num_light == 5) yaku_list.emplace_back(1, "Five Lights", g_rules.five_lights);
    else if (num_light == 4 && (pile & MASKS.rainman) == 0) yaku_list.emplace_back(2, "Four Lights", g_rules.four_lights);
    else if (num_light == 4) yaku_list.emplace_back(3, "Rainy Four Lights", g_rules.rainy_four_lights);
    else if (num_light == 3 && (pile & MASKS.rainman) == 0) yaku_list.emplace_back(4, "Three Lights", g_rules.three_lights);

    const int num_seed = popcount(pile & MASKS.seed);
    if ((pile & MASKS.boar_deer_butterfly) == MASKS.boar_deer_butterfly) yaku_list.emplace_back(5, "Boar-Deer-Butterfly", g_rules.bdb);
    
    const int sake_pt = (koikoi_num == 0 ? g_rules.flower_moon_base_pt : g_rules.flower_moon_koikoi_pt);
    if (g_rules.enable_flower_sake && (pile & MASKS.flower_sake) == MASKS.flower_sake) 
        yaku_list.emplace_back(koikoi_num == 0 ? 6 : 7, "Flower Viewing Sake", sake_pt);
    if (g_rules.enable_moon_sake && (pile & MASKS.moon_sake) == MASKS.moon_sake) 
        yaku_list.emplace_back(koikoi_num == 0 ? 8 : 9, "Moon Viewing Sake", sake_pt);
        
    if (num_seed >= 5) yaku_list.emplace_back(10, "Tane", num_seed - 4);

    const int num_ribbon = popcount(pile & MASKS.ribbon);
    if ((pile & MASKS.red_blue_ribbon) == MASKS.red_blue_ribbon) yaku_list.emplace_back(11, "Red & Blue Ribbons", g_rules.red_blue_ribbon);
    if ((pile & MASKS.red_ribbon) == MASKS.red_ribbon) yaku_list.emplace_back(12, "Red Ribbons", g_rules.red_ribbon);
    if ((pile & MASKS.blue_ribbon) == MASKS.blue_ribbon) yaku_list.emplace_back(13, "Blue Ribbons", g_rules.blue_ribbon);
    if (num_ribbon >= 5) yaku_list.emplace_back(14, "Tan", num_ribbon - 4);

    const int num_dross = popcount(pile & MASKS.dross);
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
    float old_value;            // 収集時のValue (Critic) 出力
    float old_action_prob;      // 収集時の挙動方策確率 μ(a_t|s_t)
    float old_logits[NUM_CARDS];       // 収集時のPolicy (Actor) 出力
    bool legal_mask[NUM_CARDS];        // 合法手マスク
    Snapshot snap;
};

struct ExplorationParams {
    float temperature;
    float epsilon;
    float uniform_noise_rate;

    bool operator==(const ExplorationParams& o) const {
        return temperature == o.temperature && epsilon == o.epsilon && uniform_noise_rate == o.uniform_noise_rate;
    }
    bool operator!=(const ExplorationParams& o) const { return !(*this == o); }
};

inline void write_feature_core(float* dest, const Snapshot& snap, int max_round) {
    std::memset(dest, 0, NUM_FEAT_ROWS * NUM_CARDS * sizeof(float));

    // Design Intent: 行アドレスを固定し、ループ内の乗算コストをゼロにする
    auto set_row_bb = [dest](int row, uint64_t bb, float val = 1.0f) {
        float* row_ptr = dest + row * NUM_CARDS;
        while (bb) {
            row_ptr[ctz64(bb)] = val;
            bb &= (bb - 1);
        }
    };

    // 1. 物理的なカード配置（C0〜C5）
    set_row_bb(0, snap.bb_hand);
    set_row_bb(1, snap.bb_field);
    set_row_bb(2, snap.bb_my_pile);
    set_row_bb(3, snap.bb_op_pile);
    set_row_bb(4, snap.bb_unseen);
    set_row_bb(5, snap.bb_my_discard);

    // 2. 札属性・役ポテンシャル（C6〜C9）
    set_row_bb(6, MASKS.light);
    set_row_bb(7, MASKS.seed);
    set_row_bb(8, MASKS.ribbon);
    set_row_bb(9, MASKS.dross);

    // C10: 月の残り枚数（Design Intent: ループ内判定を避け4枚単位の連続メモリ展開で超高速化）
    for (int m = 0; m < 12; ++m) {
        const uint64_t month_bb = 0xFULL << (m * 4);
        const float val = popcount(snap.bb_unseen & month_bb) * 0.25f; // /4.0f
        float* row10 = dest + 10 * NUM_CARDS + (m * 4);
        row10[0] = val; row10[1] = val; row10[2] = val; row10[3] = val;
    }

    // C11〜C14: 役リーチフラグ
    const int current_pt = get_yaku_point_by_bitboard(snap.bb_my_pile, snap.koikoi_num_turn);
    uint64_t field_tmp = snap.bb_field;
    while (field_tmp) {
        const int c = ctz64(field_tmp);
        const uint64_t card_bit = 1ULL << c;
        if (get_yaku_point_by_bitboard(snap.bb_my_pile | card_bit, snap.koikoi_num_turn) > current_pt) {
            if (card_bit & MASKS.light)  dest[11 * NUM_CARDS + c] = 1.0f;
            if (card_bit & MASKS.seed)   dest[12 * NUM_CARDS + c] = 1.0f;
            if (card_bit & MASKS.ribbon) dest[13 * NUM_CARDS + c] = 1.0f;
            if (card_bit & MASKS.dross)  dest[14 * NUM_CARDS + c] = 1.0f;
        }
        field_tmp &= (field_tmp - 1);
    }

    // 3. 信念状態 / Belief State（C15, C16）
    for (int m = 0; m < 12; ++m) {
        if ((snap.suit_op_ignored >> m) & 1) {
            float* p = dest + 15 * NUM_CARDS + (m * 4);
            p[0] = -1.0f; p[1] = -1.0f; p[2] = -1.0f; p[3] = -1.0f;
        }
        if ((snap.suit_op_played >> m) & 1) {
            float* p = dest + 16 * NUM_CARDS + (m * 4);
            p[0] = 1.0f; p[1] = 1.0f; p[2] = 1.0f; p[3] = 1.0f;
        }
    }

    // 4. グローバルコンテキスト（C17〜C23）
    const float c17 = snap.point_turn / 60.0f;
    const float c18 = snap.point_idle / 60.0f;
    const int my_yaku = get_yaku_point_by_bitboard(snap.bb_my_pile, snap.koikoi_num_turn);
    const int op_yaku = get_yaku_point_by_bitboard(snap.bb_op_pile, snap.koikoi_num_idle);
    const float c19 = static_cast<float>(my_yaku) / (op_yaku + 1.0f);
    const float c20 = static_cast<float>(snap.koikoi_num_turn) / (snap.koikoi_num_idle + 1.0f);
    const float c21 = snap.turn_16 / 16.0f;
    const float c22 = snap.is_dealer ? 1.0f : -1.0f;
    const float c23 = static_cast<float>(snap.round_num) / (max_round > 0 ? max_round : 1); // Validation: ゼロ除算ガード

    // Design Intent: std::fill_nによりSIMDベクトル転送命令を一括生成させる
    std::fill_n(dest + 17 * NUM_CARDS, NUM_CARDS, c17);
    std::fill_n(dest + 18 * NUM_CARDS, NUM_CARDS, c18);
    std::fill_n(dest + 19 * NUM_CARDS, NUM_CARDS, c19);
    std::fill_n(dest + 20 * NUM_CARDS, NUM_CARDS, c20);
    std::fill_n(dest + 21 * NUM_CARDS, NUM_CARDS, c21);
    std::fill_n(dest + 22 * NUM_CARDS, NUM_CARDS, c22);
    std::fill_n(dest + 23 * NUM_CARDS, NUM_CARDS, c23);
}

class KoiKoiStateManager {
public:
    uint64_t bb_hand[3]{};
    uint64_t bb_pile[3]{};
    uint64_t bb_field = 0;
    uint64_t bb_stock = 0;
    uint64_t bb_show = 0;
    uint64_t bb_pairing = 0;
    
    // --- 新仕様トラッキング ---
    uint64_t bb_discard_hist[3]{};
    uint16_t suit_played[3]{};
    uint16_t suit_ignored[3]{};

    KoiKoiStateManager() = default;

    void init_board(uint64_t hand1, uint64_t hand2, uint64_t field, uint64_t stock) {
        bb_hand[1] = hand1; bb_hand[2] = hand2;
        bb_field = field; bb_stock = stock;
        bb_pile[1] = 0; bb_pile[2] = 0;
        bb_show = 0; bb_pairing = 0;
        
        bb_discard_hist[1] = 0; bb_discard_hist[2] = 0;
        suit_played[1] = 0; suit_played[2] = 0;
        suit_ignored[1] = 0; suit_ignored[2] = 0;
    }

    void set_state_from_vision(
        uint64_t hand1, uint64_t hand2, uint64_t field, uint64_t stock,
        uint64_t pile1, uint64_t pile2,
        uint64_t my_discard, uint16_t op_played, uint16_t op_ignored) 
    {
        bb_hand[1] = hand1; bb_hand[2] = hand2;
        bb_field = field; bb_stock = stock;
        bb_pile[1] = pile1; bb_pile[2] = pile2;
        
        bb_discard_hist[1] = my_discard; bb_discard_hist[2] = 0;
        suit_played[1] = 0; suit_played[2] = op_played;
        suit_ignored[1] = 0; suit_ignored[2] = op_ignored;
    }

    void discard(int p, int card, uint64_t pairing, int turn_16) {
        bb_hand[p] &= ~(1ULL << card);
        bb_show = (1ULL << card);
        bb_pairing = pairing;
        
        // 信念状態トラッキングの更新
        bb_discard_hist[p] |= (1ULL << card);
        const int suit = card / 4;
        suit_played[p] |= (1 << suit);
        
        uint16_t field_suits = 0;
        uint64_t f = bb_field;
        while (f) {
            field_suits |= (1 << (ctz64(f) / 4));
            f &= (f - 1);
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

    py::array_t<float> get_feature(
        bool is_koikoi, int point_turn, int point_idle,
        int round_num, int turn_16, int dealer,
        int koikoi_num_turn, int koikoi_num_idle,
        int turn_p, int idle_p,
        int max_round) 
    {
        auto result = py::array_t<float>({NUM_FEAT_ROWS, NUM_CARDS});
        
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

        write_feature_core(result.mutable_data(), snap, max_round);
        return result;
    }

    py::array_t<bool> get_action_mask(int state_type, int p) {
        const int size = (state_type == 2) ? 2 : NUM_CARDS;
        auto result = py::array_t<bool>(size);
        auto buf = result.mutable_unchecked<1>();
        
        if (state_type == 2) {
            buf(0) = true; buf(1) = true;
            return result;
        }

        // Design Intent: 48回の全インデックス判定を廃止し、存在するビットのみをピンポイントでTrue化
        std::memset(result.mutable_data(), 0, NUM_CARDS * sizeof(bool));
        uint64_t bb = (state_type == 0) ? bb_hand[p] : bb_pairing;
        while (bb) {
            buf(ctz64(bb)) = true;
            bb &= (bb - 1);
        }
        return result;
    }

    int get_best_action(int state_type, int p, py::array_t<float>& logits) {
        auto log_buf = logits.unchecked<1>();
        if (state_type == 2) {
            return log_buf(1) > log_buf(0) ? 1 : 0;
        }

        int best_idx = -1;
        float max_val = -std::numeric_limits<float>::infinity();
        uint64_t bb = (state_type == 0) ? bb_hand[p] : bb_pairing;

        // Design Intent: O(48)をO(ポップカウント数)に短縮
        while (bb) {
            const int idx = ctz64(bb);
            const float val = log_buf(idx);
            if (val > max_val) {
                max_val = val;
                best_idx = idx;
            }
            bb &= (bb - 1);
        }
        return best_idx;
    }
};

class KoiKoiTraceBuffer {
public:
    int capacity, rows, cols, current_size;
    
    torch::Tensor states_tensor, actions_tensor, rewards_tensor;
    torch::Tensor old_logits_tensor, old_values_tensor, legal_masks_tensor;
    torch::Tensor old_action_probs_tensor; // ★追加

    float* states_ptr;
    int64_t* actions_ptr;
    float *rewards_ptr, *old_logits_ptr, *old_values_ptr;
    bool* legal_masks_ptr;
    float* old_action_probs_ptr;

    KoiKoiTraceBuffer(int cap, int r, int c) 
        : capacity(cap), rows(r), cols(c), current_size(0) 
    {
        if (cap <= 0 || r <= 0 || c <= 0) {
            throw std::invalid_argument("KoiKoiTraceBuffer dimensions must be positive."); // Validation
        }

        auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32);
        auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64);
        auto opt_bool = torch::TensorOptions().dtype(torch::kBool);
        
        states_tensor = torch::empty({capacity, rows, cols}, opt_f32);
        actions_tensor = torch::empty({capacity}, opt_i64);
        rewards_tensor = torch::empty({capacity}, opt_f32);
        
        old_logits_tensor = torch::empty({capacity, NUM_CARDS}, opt_f32);
        old_values_tensor = torch::empty({capacity}, opt_f32);
        legal_masks_tensor = torch::empty({capacity, NUM_CARDS}, opt_bool);
        old_action_probs_tensor = torch::empty({capacity}, opt_f32);

        states_ptr = states_tensor.data_ptr<float>();
        actions_ptr = actions_tensor.data_ptr<int64_t>();
        rewards_ptr = rewards_tensor.data_ptr<float>();
        
        old_logits_ptr = old_logits_tensor.data_ptr<float>();
        old_values_ptr = old_values_tensor.data_ptr<float>();
        legal_masks_ptr = legal_masks_tensor.data_ptr<bool>();
        old_action_probs_ptr = old_action_probs_tensor.data_ptr<float>();
    }

    void clear() { current_size = 0; }

    void push_reconstructed(const TraceSlot& slot, float mc_return, int max_round) {
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

        float* dest_state = states_ptr + idx * rows * cols;
        actions_ptr[idx] = slot.action;
        rewards_ptr[idx] = mc_return; // MC Return
        old_values_ptr[idx] = slot.old_value;
        old_action_probs_ptr[idx] = slot.old_action_prob;

        std::memcpy(old_logits_ptr + idx * NUM_CARDS, slot.old_logits, NUM_CARDS * sizeof(float));
        std::memcpy(legal_masks_ptr + idx * NUM_CARDS, slot.legal_mask, NUM_CARDS * sizeof(bool));

        write_feature_core(dest_state, slot.snap, max_round);
    }

    py::dict finalize() {
        py::dict result;
        if (current_size == 0) {
            result["states"] = torch::empty({0, rows, cols});
            result["actions"] = torch::empty({0});
            result["rewards"] = torch::empty({0});
            result["old_logits"] = torch::empty({0, NUM_CARDS});
            result["old_values"] = torch::empty({0});
            result["legal_masks"] = torch::empty({0, NUM_CARDS}, torch::kBool);
            result["old_action_probs"] = torch::empty({0}, torch::kFloat32);
        } else {
            result["states"] = states_tensor.slice(0, 0, current_size).clone();
            result["actions"] = actions_tensor.slice(0, 0, current_size).clone();
            result["rewards"] = rewards_tensor.slice(0, 0, current_size).clone();
            result["old_logits"] = old_logits_tensor.slice(0, 0, current_size).clone();
            result["old_values"] = old_values_tensor.slice(0, 0, current_size).clone();
            result["legal_masks"] = legal_masks_tensor.slice(0, 0, current_size).clone();
            result["old_action_probs"] = old_action_probs_tensor.slice(0, 0, current_size).clone();
        }
        return result;
    }
};

enum class GameState : uint8_t { 
    INIT, DISCARD, DISCARD_PICK, DRAW, DRAW_PICK, KOIKOI, ROUND_OVER, GAME_OVER 
};

class KoiKoiEnv {
public:
    int round_num = 1;
    int dealer = 1;
    int winner = 1;
    int turn_16 = 1;
    int turn_point = 0;
    int point[3]{};       // [1]=P1, [2]=P2
    int koikoi[3][8]{};   // [1]=P1, [2]=P2
    bool exhausted = false;
    bool wait_action = false;
    GameState state = GameState::ROUND_OVER;

    KoiKoiStateManager state_manager;
    std::mt19937 rng;

    FixedVec<int, NUM_CARDS> hand[3];
    FixedVec<int, NUM_CARDS> field_slot;
    FixedVec<int, NUM_CARDS> stock;
    FixedVec<int, NUM_CARDS> pile[3];
    FixedVec<int, 4> show;
    FixedVec<int, 8> collect;

    explicit KoiKoiEnv(int seed = 42) : rng(seed) { reset_game(); }

    void reset_game() {
        round_num = 1; point[1] = 30; point[2] = 30;
        std::uniform_int_distribution<int> dist(1, 2);
        dealer = dist(rng);
        winner = dealer;
        state = GameState::ROUND_OVER;
    }

    void reset_game_with_dealer(int initial_dealer) {
        round_num = 1; point[1] = 30; point[2] = 30;
        dealer = initial_dealer;
        winner = dealer;
        state = GameState::ROUND_OVER;
    }

    void reset_round() {
        if (winner != 0 && winner != -1) dealer = winner;
        turn_16 = 1;
        for (int p = 1; p <= 2; ++p) {
            pile[p].clear();
            std::memset(koikoi[p], 0, sizeof(koikoi[p]));
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
    
    int koikoi_num(int p) const { 
        int s = 0; 
        for (int i = 0; i < 8; ++i) s += koikoi[p][i]; 
        return s; 
    }

    int round_point(int p) {
        if (winner == 0) return 0;
        if (exhausted) return (dealer == p) ? 1 : -1;
        const int pt = state_manager.get_yaku_point(winner, koikoi_num(winner));
        return (winner == p) ? pt : -pt;
    }

    bool needs_action() const {
        return wait_action && (state == GameState::DISCARD || state == GameState::DISCARD_PICK ||
                               state == GameState::DRAW_PICK || state == GameState::KOIKOI);
    }

    FixedVec<int, 4> get_pairing_cards() const {
        FixedVec<int, 4> pairs;
        if (show.empty()) return pairs;
        const int target = show[0] / 4;
        for (int c : field_slot) {
            if (c != -1 && c / 4 == target) pairs.push_back(c);
        }
        return pairs;
    }
    
    void remove_from_field(int card) {
        auto it = std::find(field_slot.begin(), field_slot.end(), card);
        if (it != field_slot.end()) *it = -1;
    }

    void _collect_card(int card) {
        FixedVec<int, 4> pairing = get_pairing_cards();
        collect.clear();
        if (pairing.empty()) {
            auto it = std::find(field_slot.begin(), field_slot.end(), -1);
            if (it != field_slot.end()) *it = show[0];
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
        const int p = turn_player();
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
        auto it = std::find(hand[p].begin(), hand[p].end(), action);
        if (it != hand[p].end()) hand[p].erase(it);

        show.clear();
        show.push_back(action);

        FixedVec<int, 4> pairing = get_pairing_cards();
        uint64_t pair_bb = 0; 
        for (int c : pairing) pair_bb |= (1ULL << c);
        
        state_manager.discard(p, action, pair_bb, turn_16);
        state = GameState::DISCARD_PICK; 
        wait_action = (pairing.size() == 2);
    }

    void handle_discard_pick(int p, int action) {
        _collect_card(action);
        uint64_t coll_bb = 0;  for (int c : collect) coll_bb |= (1ULL << c);
        uint64_t field_bb = 0; for (int c : field_slot) if (c != -1) field_bb |= (1ULL << c);
        
        state_manager.discard_pick(p, coll_bb, field_bb, turn_16);
        state = GameState::DRAW; wait_action = false;
    }

    void handle_draw(int p) {
        const int drawn = stock.back(); stock.pop_back(); 
        show.clear();
        show.push_back(drawn);
        
        FixedVec<int, 4> pairing = get_pairing_cards();
        uint64_t pair_bb = 0; 
        for (int c : pairing) pair_bb |= (1ULL << c);
        
        state_manager.draw(drawn, pair_bb, turn_16);
        state = GameState::DRAW_PICK; wait_action = (pairing.size() == 2);
    }

    void handle_draw_pick(int p, int action) {
        _collect_card(action);
        uint64_t coll_bb = 0;  for (int c : collect) coll_bb |= (1ULL << c);
        uint64_t field_bb = 0; for (int c : field_slot) if (c != -1) field_bb |= (1ULL << c);
        
        state_manager.draw_pick(p, coll_bb, field_bb, turn_16);
        state = GameState::KOIKOI;
        const int pt = state_manager.get_yaku_point(p, koikoi_num(p));
        wait_action = (pt > turn_point) && (turn_8() < 8);
    }

    void handle_koikoi(int p, int action) {
        const int pt = state_manager.get_yaku_point(p, koikoi_num(p));
        bool is_koikoi = (action != 0);
        if ((pt > turn_point) && (turn_8() == 8)) is_koikoi = false;
        if (wait_action) koikoi[p][turn_8() - 1] = is_koikoi ? 1 : 0;

        if (!is_koikoi) { state = GameState::ROUND_OVER; wait_action = false; winner = p; }
        else if (turn_16 == 16) { state = GameState::ROUND_OVER; wait_action = false; exhausted = true; winner = dealer; }
        else { turn_16++; state = GameState::DISCARD; wait_action = true; }
    }

    void deal_card() {
        std::array<int, NUM_CARDS> cards{};
        for (int i = 0; i < NUM_CARDS; ++i) cards[i] = i;

        while (true) {
            std::shuffle(cards.begin(), cards.end(), rng);
            const int d_idx = dealer;
            const int nd_idx = (dealer == 1) ? 2 : 1;
            
            hand[d_idx].clear(); hand[nd_idx].clear(); field_slot.clear(); stock.clear();
            
            for (int i = 0; i < 8; ++i) hand[d_idx].push_back(cards[i]);
            std::sort(hand[d_idx].begin(), hand[d_idx].end());
            
            for (int i = 8; i < 16; ++i) hand[nd_idx].push_back(cards[i]);
            std::sort(hand[nd_idx].begin(), hand[nd_idx].end());
            
            std::array<int, 8> init_f{};
            for (int i = 16; i < 24; ++i) init_f[i - 16] = cards[i];
            std::sort(init_f.begin(), init_f.end());
            
            for (int i = 0; i < 8; ++i) field_slot.push_back(init_f[i]);
            for (int i = 0; i < 8; ++i) field_slot.push_back(-1); 
            for (int i = 24; i < NUM_CARDS; ++i) stock.push_back(cards[i]);

            // Design Intent: 手四・場四チェックをビット論理積へ置換しO(1)化
            uint64_t bb_h1 = 0, bb_h2 = 0, bb_f = 0;
            for (int c : hand[1]) bb_h1 |= (1ULL << c);
            for (int c : hand[2]) bb_h2 |= (1ULL << c);
            for (int i = 0; i < 8; ++i) bb_f |= (1ULL << init_f[i]);

            bool valid_deal = true;
            for (int s = 0; s < 12; ++s) {
                const uint64_t m_mask = 0xFULL << (s * 4);
                if (popcount(bb_h1 & m_mask) == 4 || popcount(bb_h2 & m_mask) == 4 || popcount(bb_f & m_mask) == 4) { 
                    valid_deal = false; 
                    break; 
                }
            }
            if (valid_deal) break;
        }
    }
};

class BatchSimulator {
public: 
    int num_envs, target_games;
    float discount;
    float gae_lambda;
    std::vector<KoiKoiEnv> envs;
    std::vector<FixedVec<TraceSlot, 512>> traces[3];

    KoiKoiTraceBuffer buf_discard, buf_pick, buf_koikoi;
    int finished_games = 0;
    torch::Device device;

    torch::Tensor feat_tensor[3];
    torch::Tensor mask_tensor[3];
    ExplorationParams exp_params;

private:
    // Design Intent: スタック溢れ回避のため推論結果の作業バッファをメンバ常駐化
    std::vector<int64_t> act_buf_;
    std::vector<float> prob_buf_;

public:
    BatchSimulator(int n_envs, int target, int cap_d, int cap_p, int cap_k, float disc, float lambda_val, const std::string& dev_str, ExplorationParams ep)
        : num_envs(n_envs), target_games(target), discount(disc), gae_lambda(lambda_val),
          buf_discard(cap_d, NUM_FEAT_ROWS, NUM_CARDS), 
          buf_pick(cap_p, NUM_FEAT_ROWS, NUM_CARDS), 
          buf_koikoi(cap_k, NUM_FEAT_ROWS, NUM_CARDS),
          device(dev_str), exp_params(ep),
          act_buf_(n_envs), prob_buf_(n_envs)
    {
        envs.reserve(n_envs);
        std::random_device rd;
        for (int i = 0; i < n_envs; ++i) envs.emplace_back(rd());
        for (int p = 1; p <= 2; ++p) traces[p].resize(n_envs);

        auto opt_f32_pin = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        auto opt_bool_pin = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);

        for (int i = 0; i < 3; ++i) {
            feat_tensor[i] = torch::empty({num_envs, NUM_FEAT_ROWS, NUM_CARDS}, opt_f32_pin);
        }
        mask_tensor[0] = torch::empty({num_envs, NUM_CARDS}, opt_bool_pin);
        mask_tensor[1] = torch::empty({num_envs, NUM_CARDS}, opt_bool_pin);
        mask_tensor[2] = torch::empty({num_envs, 2}, opt_bool_pin);
    }

    void reset() {
        finished_games = 0;
        for (auto& env : envs) { env.reset_game(); env.reset_round(); }
        for (int p = 1; p <= 2; ++p) {
            for (auto& trace_vec : traces[p]) trace_vec.clear();
        }
        buf_discard.clear(); buf_pick.clear(); buf_koikoi.clear();
    }

    py::dict finalize_buffers() {
        py::dict res;
        res["discard"] = buf_discard.finalize();
        res["pick"]    = buf_pick.finalize();
        res["koikoi"]  = buf_koikoi.finalize();
        return res;
    }

    void build_feature_into(int i, float* feat, int type, int max_round) {
        const auto& env = envs[i]; 
        write_feature_core(feat, capture_snapshot(env, env.turn_player(), env.idle_player(), type, max_round), max_round);
    }

    void build_mask_into(int i, bool* mk, int type) {
        const auto& env = envs[i]; 
        const int sz = (type == 2) ? 2 : NUM_CARDS;
        std::memset(mk, 0, sz * sizeof(bool));
        if (type == 2) { mk[0] = true; mk[1] = true; }
        else {
            uint64_t bb = (type == 0) ? env.state_manager.bb_hand[env.turn_player()] : env.state_manager.bb_pairing;
            while (bb) { mk[ctz64(bb)] = true; bb &= (bb - 1); } // 跳躍走査
        }
    }

    void process_game_over(int i, int max_round) {
        auto& env = envs[i];
        int final_winner = 0;
        if (env.point[1] > env.point[2])      final_winner = 1;
        else if (env.point[2] > env.point[1]) final_winner = 2;

        for (int p_id = 1; p_id <= 2; ++p_id) {
            auto& trace = traces[p_id][i];
            if (trace.empty()) continue;

            float term_reward = (final_winner == p_id) ? 1.0f : (final_winner != 0 ? -1.0f : 0.0f);
            const int T = trace.size();
            
            // Design Intent: ヒープ確保を全廃。TraceSlot上限(64)に合わせたスタック確保
            std::array<float, 512> gae_returns{};
            float gae_a = 0.0f;

            for (int t = T - 1; t >= 0; --t) {
                const float r_t = (t == T - 1) ? term_reward : 0.0f;
                const float v_t = trace[t].old_value;
                const float v_next = (t == T - 1) ? 0.0f : trace[t + 1].old_value;

                const float delta = r_t + discount * v_next - v_t;
                gae_a = delta + (discount * gae_lambda) * gae_a;
                gae_returns[t] = gae_a + v_t;
            }

            for (int t = 0; t < T; ++t) {
                const auto& slot = trace[t];
                if (slot.state_type == 0)      buf_discard.push_reconstructed(slot, gae_returns[t], max_round);
                else if (slot.state_type == 1) buf_pick.push_reconstructed(slot, gae_returns[t], max_round);
                else                           buf_koikoi.push_reconstructed(slot, gae_returns[t], max_round);
            }
            trace.clear(); 
        }
    }

    void play_games(int max_round) {
        std::array<std::vector<int>, 3> req_env_idx{};
        float* feat_ptrs[3] = { feat_tensor[0].data_ptr<float>(), feat_tensor[1].data_ptr<float>(), feat_tensor[2].data_ptr<float>() };
        bool* mask_ptrs[3]  = { mask_tensor[0].data_ptr<bool>(),  mask_tensor[1].data_ptr<bool>(),  mask_tensor[2].data_ptr<bool>() };

        while (finished_games < target_games) {
            step_environments_without_action(max_round);
            if (finished_games >= target_games) break;

            for (int p_id = 1; p_id <= 2; ++p_id) {
                for (int i = 0; i < 3; ++i) req_env_idx[i].clear();
                
                for (int i = 0; i < num_envs; ++i) {
                    auto& env = envs[i];
                    if (env.needs_action() && env.turn_player() == p_id) {
                        const int type = (env.state == GameState::DISCARD) ? 0 : (env.state == GameState::KOIKOI ? 2 : 1);
                        req_env_idx[type].push_back(i);
                    }
                }

                torch::NoGradGuard no_grad; 
                for (int type = 0; type < 3; ++type) {
                    const int n = static_cast<int>(req_env_idx[type].size());
                    if (n == 0) continue;

                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) {
                        const int env_idx = req_env_idx[type][i];
                        build_feature_into(env_idx, feat_ptrs[type] + i * NUM_FEAT_ROWS * NUM_CARDS, type, max_round);
                        build_mask_into(env_idx, mask_ptrs[type] + i * ((type == 2) ? 2 : NUM_CARDS), type);
                    }

                    torch::Tensor f_gpu = feat_tensor[type].slice(0, 0, n).to(device, true).to(torch::kFloat32);
                    auto out_tuple = ((type == 0) ? g_models.discard.forward({f_gpu})
                                    : (type == 1) ? g_models.pick.forward({f_gpu})
                                    :               g_models.koikoi.forward({f_gpu})).toTuple();
                                    
                    torch::Tensor logits_cpu = out_tuple->elements()[0].toTensor().cpu();
                    torch::Tensor value_cpu  = out_tuple->elements()[1].toTensor().cpu();
                    
                    const float* log_ptr = logits_cpu.data_ptr<float>();
                    const float* val_ptr = value_cpu.data_ptr<float>();
                    const bool*  msk_ptr = mask_tensor[type].data_ptr<bool>(); 

                    select_actions_neurd(type, n, req_env_idx[type].data(), log_ptr, msk_ptr, act_buf_.data(), prob_buf_.data(), exp_params);

                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) {
                        const int env_idx = req_env_idx[type][i];
                        auto& env = envs[env_idx];

                        TraceSlot slot;
                        slot.state_type = type;
                        slot.action = act_buf_[i];
                        slot.old_value = val_ptr[i];
                        slot.old_action_prob = prob_buf_[i];
                        
                        const int cols = (type == 2) ? 2 : NUM_CARDS;
                        std::memset(slot.old_logits, 0, NUM_CARDS * sizeof(float));
                        std::memset(slot.legal_mask, 0, NUM_CARDS * sizeof(bool));
                        
                        for (int c = 0; c < cols; ++c) {
                            slot.old_logits[c] = log_ptr[i * cols + c];
                            slot.legal_mask[c] = msk_ptr[i * cols + c];
                        }

                        slot.snap = capture_snapshot(env, p_id, (p_id == 1) ? 2 : 1, type, max_round);
                        traces[p_id][env_idx].push_back(slot);

                        env.step(slot.action);
                    }
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
        snap.state_type = static_cast<uint8_t>(type);
        return snap;
    }

    void step_environments_without_action(int max_round) {
        int local_finished = 0;
        #pragma omp parallel for reduction(+:local_finished)
        for (int i = 0; i < num_envs; ++i) {
            auto& env = envs[i];
            while (!env.needs_action() && env.state != GameState::GAME_OVER) {
                if (env.state == GameState::ROUND_OVER) {
                    env.point[1] += env.round_point(1);
                    env.point[2] += env.round_point(2);

                    if (env.point[1] <= 0 || env.point[2] <= 0 || env.round_num == max_round) {
                        process_game_over(i, max_round);
                        local_finished++;
                        env.reset_game(); env.reset_round();
                    } else {
                        env.round_num++; env.reset_round(); 
                    }
                } else { 
                    env.step(-1); 
                }
            }
        }
        finished_games += local_finished;
    }

    void select_actions_neurd(int type, int n, const int* env_idx_list, const float* out_ptr, const bool* mask_ptr, int64_t* act_ptr, float* prob_ptr, const ExplorationParams& params) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        const int cols = (type == 2) ? 2 : NUM_CARDS;

        for (int i = 0; i < n; ++i) {
            FixedVec<int, NUM_CARDS> valid_actions;
            for (int c = 0; c < cols; ++c) {
                if (mask_ptr[i * cols + c]) valid_actions.push_back(c);
            }

            if (valid_actions.empty()) {
                act_ptr[i] = 0; prob_ptr[i] = 1.0f;
                continue;
            }

            const int num_legals = valid_actions.size();
            const float p_noise = params.uniform_noise_rate + params.epsilon;
            const float uniform_part = p_noise / num_legals;
            const float policy_weight = 1.0f - p_noise;

            std::array<float, NUM_CARDS> mu{};

            if (params.temperature <= 0.0f) {
                float max_val = -1e9f;
                for (int a : valid_actions) if (out_ptr[i * cols + a] > max_val) max_val = out_ptr[i * cols + a];
                
                int best_count = 0;
                for (int a : valid_actions) if (out_ptr[i * cols + a] == max_val) best_count++;
                
                for (int a : valid_actions) {
                    mu[a] = uniform_part + policy_weight * ((out_ptr[i * cols + a] == max_val) ? (1.0f / best_count) : 0.0f);
                }
            } else {
                float max_logit = -1e9f;
                for (int a : valid_actions) if (out_ptr[i * cols + a] > max_logit) max_logit = out_ptr[i * cols + a];
                
                float sum_exp = 0.0f;
                std::array<float, NUM_CARDS> exps{};
                for (int a : valid_actions) {
                    exps[a] = std::exp((out_ptr[i * cols + a] - max_logit) / params.temperature);
                    sum_exp += exps[a];
                }
                for (int a : valid_actions) {
                    mu[a] = uniform_part + policy_weight * (exps[a] / sum_exp);
                }
            }

            // Design Intent: バッチ内各環境の独立したRNGからサンプリング (Bug Fix)
            const float r = dist(envs[env_idx_list[i]].rng);
            float acc = 0.0f;
            int chosen = valid_actions.back();
            for (int a : valid_actions) {
                acc += mu[a];
                if (r <= acc) { chosen = a; break; }
            }

            act_ptr[i] = chosen;
            prob_ptr[i] = mu[chosen];
        }
    }
};

struct SimConfig {
    int num_threads, n_envs, target, cap_d, cap_p, cap_k;
    float disc;
    float gae_lambda;
    std::string dev_str;
    int max_round;
    ExplorationParams exp_params;

    // C++17互換の明示的比較（全コンパイラ環境で100%通ります）
    bool operator==(const SimConfig& o) const {
        return num_threads == o.num_threads &&
               n_envs == o.n_envs &&
               target == o.target &&
               cap_d == o.cap_d &&
               cap_p == o.cap_p &&
               cap_k == o.cap_k &&
               disc == o.disc &&
               gae_lambda == o.gae_lambda &&
               dev_str == o.dev_str &&
               max_round == o.max_round &&
               exp_params == o.exp_params;
    }

    bool operator!=(const SimConfig& o) const { 
        return !(*this == o); 
    }
};

class SimulationManager {
public:
    std::vector<std::unique_ptr<BatchSimulator>> sims;
    SimConfig current_config;

    std::vector<std::thread> workers;
    std::mutex pool_mtx;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    
    int generation = 0;
    bool shutdown = false;
    int done_workers = 0;
    std::vector<std::exception_ptr> worker_exceptions;

    explicit SimulationManager(const SimConfig& config) 
        : current_config(config), worker_exceptions(config.num_threads, nullptr) 
    {
        sims.reserve(config.num_threads);
        workers.reserve(config.num_threads);

        for (int i = 0; i < config.num_threads; ++i) {
            sims.push_back(std::make_unique<BatchSimulator>(
                config.n_envs, config.target, config.cap_d, config.cap_p, config.cap_k, 
                config.disc, config.gae_lambda, config.dev_str, config.exp_params
            ));
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
            if (t.joinable()) t.join();
        }
    }

    void reset_all() {
        for (auto& sim : sims) sim->reset();
    }

    void run_simulation_parallel() {
        {
            std::lock_guard<std::mutex> lock(pool_mtx);
            std::fill(worker_exceptions.begin(), worker_exceptions.end(), nullptr);
            done_workers = 0;
            generation++; 
        }
        cv_start.notify_all();

        {
            std::unique_lock<std::mutex> lock(pool_mtx);
            cv_done.wait(lock, [this]() { return done_workers == current_config.num_threads; });
        }

        for (const auto& e : worker_exceptions) {
            if (e) std::rethrow_exception(e);
        }
    }

private:
    void worker_loop(int worker_id) {
        int local_gen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(pool_mtx);
                cv_start.wait(lock, [this, local_gen]() { 
                    return generation > local_gen || shutdown; 
                });
                if (shutdown && generation <= local_gen) break;
                local_gen = generation;
            }

            // Design Intent: RAIIガードにより例外発生時も確実にcv_doneを発火させハングを防ぐ
            struct WorkerDoneGuard {
                SimulationManager& mgr;
                ~WorkerDoneGuard() {
                    std::lock_guard<std::mutex> lock(mgr.pool_mtx);
                    mgr.done_workers++;
                    if (mgr.done_workers == mgr.current_config.num_threads) {
                        mgr.cv_done.notify_one();
                    }
                }
            } guard{*this};

            try {
                sims[worker_id]->play_games(current_config.max_round);
            } catch (...) {
                std::lock_guard<std::mutex> lock(pool_mtx);
                worker_exceptions[worker_id] = std::current_exception();
            }
        }
    }
};

// ---------------------------------------------------------
// アリーナ（評価フェーズ）専用シミュレータ
// ---------------------------------------------------------
class ArenaBatchSimulator {
public: 
    int num_envs, target_games;
    std::vector<KoiKoiEnv> envs;
    int next_match_to_start = 0;
    int finished_games = 0;
    std::vector<int> match_seeds;

    // Design Intent: C++標準のロックフリーアトミック集計へ変更
    std::atomic<int> wins_p1{0};
    std::atomic<int> wins_p2{0};
    std::atomic<int> draws{0};
    std::atomic<int64_t> total_score_p1{0};

    torch::Device device;
    torch::Tensor feat_tensor[3];
    torch::Tensor mask_tensor[3];

private:
    std::array<torch::jit::Module, 3> p1_models_;
    std::array<torch::jit::Module, 3> p2_models_;
    std::vector<int64_t> act_buf_; // スタック溢れ回避

public:
    ArenaBatchSimulator(int n_envs, const std::string& dev_str, 
                        std::array<torch::jit::Module, 3> p1_models,
                        std::array<torch::jit::Module, 3> p2_models)
        : num_envs(n_envs), target_games(n_envs), device(dev_str),
          p1_models_(std::move(p1_models)), p2_models_(std::move(p2_models)),
          act_buf_(n_envs)
    {
        const int target_pairs = n_envs / 2;
        std::random_device rd;
        for (int i = 0; i < target_pairs; ++i) match_seeds.push_back(rd());

        auto opt_f32_pin = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        auto opt_bool_pin = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);

        for (int i = 0; i < 3; ++i) {
            feat_tensor[i] = torch::empty({num_envs, NUM_FEAT_ROWS, NUM_CARDS}, opt_f32_pin);
        }
        mask_tensor[0] = torch::empty({num_envs, NUM_CARDS}, opt_bool_pin);
        mask_tensor[1] = torch::empty({num_envs, NUM_CARDS}, opt_bool_pin);
        mask_tensor[2] = torch::empty({num_envs, 2}, opt_bool_pin);

        envs.reserve(n_envs);
        for (int i = 0; i < n_envs; ++i) {
            envs.emplace_back(rd());
            start_next_match(i);
        }
    }

    void start_next_match(int env_idx) {
        if (next_match_to_start < target_games) {
            const int seed = match_seeds[next_match_to_start / 2];
            const int dealer = (next_match_to_start % 2) + 1; 
            envs[env_idx].rng.seed(seed);
            envs[env_idx].reset_game_with_dealer(dealer);
            envs[env_idx].reset_round();
            next_match_to_start++;
        } else {
            envs[env_idx].state = GameState::GAME_OVER;
        }
    }

    void process_game_over(int i) {
        const auto& env = envs[i];
        const int p1_pt = env.point[1];
        const int p2_pt = env.point[2];

        if (p1_pt > p2_pt)      wins_p1.fetch_add(1, std::memory_order_relaxed);
        else if (p2_pt > p1_pt) wins_p2.fetch_add(1, std::memory_order_relaxed);
        else                    draws.fetch_add(1, std::memory_order_relaxed);

        total_score_p1.fetch_add(p1_pt, std::memory_order_relaxed);
    }

    void build_feature_into(int i, float* feat, int type, int max_round) {
        const auto& env = envs[i]; 
        Snapshot snap;
        const int pt = env.turn_player(), pi = env.idle_player();
        snap.bb_hand = env.state_manager.bb_hand[pt];
        snap.bb_unseen = env.state_manager.bb_stock | env.state_manager.bb_hand[pi];
        snap.bb_my_pile = env.state_manager.bb_pile[pt];
        snap.bb_field = env.state_manager.bb_field;
        snap.bb_op_pile = env.state_manager.bb_pile[pi];
        snap.bb_my_discard = env.state_manager.bb_discard_hist[pt];
        snap.suit_op_played = env.state_manager.suit_played[pi];
        snap.suit_op_ignored = env.state_manager.suit_ignored[pi];
        snap.point_turn = env.point[pt]; snap.point_idle = env.point[pi];
        snap.round_num = env.round_num; snap.turn_16 = env.turn_16;
        snap.is_dealer = (env.dealer == pt);
        snap.koikoi_num_turn = env.koikoi_num(pt); snap.koikoi_num_idle = env.koikoi_num(pi);
        snap.state_type = static_cast<uint8_t>(type);
        
        write_feature_core(feat, snap, max_round);
    }

    void build_mask_into(int i, bool* mk, int type) {
        const auto& env = envs[i]; 
        const int sz = (type == 2) ? 2 : NUM_CARDS;
        std::memset(mk, 0, sz * sizeof(bool));
        if (type == 2) { mk[0] = true; mk[1] = true; }
        else {
            uint64_t bb = (type == 0) ? env.state_manager.bb_hand[env.turn_player()] : env.state_manager.bb_pairing;
            while (bb) { mk[ctz64(bb)] = true; bb &= (bb - 1); }
        }
    }

    void step_environments_without_action(int max_round) {
        int local_finished = 0;
        #pragma omp parallel for reduction(+:local_finished)
        for (int i = 0; i < num_envs; ++i) {
            auto& env = envs[i];
            while (!env.needs_action() && env.state != GameState::GAME_OVER) {
                if (env.state == GameState::ROUND_OVER) {
                    env.point[1] += env.round_point(1);
                    env.point[2] += env.round_point(2);

                    if (env.point[1] <= 0 || env.point[2] <= 0 || env.round_num == max_round) {
                        process_game_over(i);
                        local_finished++;
                        env.state = GameState::GAME_OVER; 
                    } else {
                        env.round_num++; env.reset_round(); 
                    }
                } else { 
                    env.step(-1); 
                }
            }
        }
        if (local_finished > 0) {
            finished_games += local_finished;
            for (int i = 0; i < num_envs; ++i) {
                if (envs[i].state == GameState::GAME_OVER && next_match_to_start < target_games) {
                    start_next_match(i);
                }
            }
        }
    }

    void select_actions_argmax(int type, int n, const float* out_ptr, const bool* mask_ptr, int64_t* act_ptr) {
        const int cols = (type == 2) ? 2 : NUM_CARDS;
        for (int i = 0; i < n; ++i) {
            float max_val = -1e9f;
            int best_a = 0;
            bool found = false;
            for (int c = 0; c < cols; ++c) {
                if (mask_ptr[i * cols + c] && (!found || out_ptr[i * cols + c] > max_val)) {
                    max_val = out_ptr[i * cols + c];
                    best_a = c;
                    found = true;
                }
            }
            act_ptr[i] = best_a;
        }
    }

    void play_games(int max_round) {
        std::array<std::vector<int>, 3> req_env_idx{};
        float* feat_ptrs[3] = { feat_tensor[0].data_ptr<float>(), feat_tensor[1].data_ptr<float>(), feat_tensor[2].data_ptr<float>() };
        bool* mask_ptrs[3]  = { mask_tensor[0].data_ptr<bool>(),  mask_tensor[1].data_ptr<bool>(),  mask_tensor[2].data_ptr<bool>() };

        while (finished_games < target_games) {
            step_environments_without_action(max_round);
            if (finished_games >= target_games) break;

            for (int p_id = 1; p_id <= 2; ++p_id) {
                for (int i = 0; i < 3; ++i) req_env_idx[i].clear();
                for (int i = 0; i < num_envs; ++i) {
                    if (envs[i].needs_action() && envs[i].turn_player() == p_id) {
                        const int type = (envs[i].state == GameState::DISCARD) ? 0 : (envs[i].state == GameState::KOIKOI ? 2 : 1);
                        req_env_idx[type].push_back(i);
                    }
                }

                torch::NoGradGuard no_grad; 
                for (int type = 0; type < 3; ++type) {
                    const int n = static_cast<int>(req_env_idx[type].size());
                    if (n == 0) continue;

                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) {
                        const int env_idx = req_env_idx[type][i];
                        build_feature_into(env_idx, feat_ptrs[type] + i * NUM_FEAT_ROWS * NUM_CARDS, type, max_round);
                        build_mask_into(env_idx, mask_ptrs[type] + i * ((type == 2) ? 2 : NUM_CARDS), type);
                    }

                    torch::Tensor f_gpu = feat_tensor[type].slice(0, 0, n).to(device, true).to(torch::kFloat32);
                    auto& model = (p_id == 1) ? p1_models_[type] : p2_models_[type];
                    auto out_tuple = model.forward({f_gpu}).toTuple();
                                    
                    torch::Tensor logits_cpu = out_tuple->elements()[0].toTensor().cpu();
                    select_actions_argmax(type, n, logits_cpu.data_ptr<float>(), mask_tensor[type].data_ptr<bool>(), act_buf_.data());

                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) {
                        envs[req_env_idx[type][i]].step(act_buf_[i]);
                    }
                }
            }
        }
    }
};

static std::unique_ptr<SimulationManager> g_sim_manager = nullptr;

py::dict run_arena_batch_simulations(
    int total_games,
    torch::jit::Module p1_d, torch::jit::Module p1_p, torch::jit::Module p1_k,
    torch::jit::Module p2_d, torch::jit::Module p2_p, torch::jit::Module p2_k,
    const std::string& dev_str, int max_round)
{
    if (total_games % 2 != 0) total_games++;

    // Design Intent: グローバル変数を経由せず直接注入することでスレッドセーフ化
    ArenaBatchSimulator sim(total_games, dev_str, {p1_d, p1_p, p1_k}, {p2_d, p2_p, p2_k});
    sim.play_games(max_round);

    py::dict res;
    res["win"]   = sim.wins_p1.load();
    res["lose"]  = sim.wins_p2.load();
    res["draw"]  = sim.draws.load();
    res["score"] = static_cast<double>(sim.total_score_p1.load());
    return res;
}

py::list run_parallel_simulations(
    int num_threads, int n_envs_per_thread, int target_games_per_thread,
    int cap_d, int cap_p, int cap_k, float disc,
    torch::jit::Module discard_model, torch::jit::Module pick_model, torch::jit::Module koikoi_model,
    const std::string& dev_str, int max_round,
    float temperature, float epsilon, float uniform_noise,
    float gae_lambda)
{
    {
        std::lock_guard<std::mutex> lock(g_models.mtx);
        if (!g_models.loaded) {
            g_models.discard = discard_model; g_models.pick = pick_model; g_models.koikoi = koikoi_model;
            g_models.discard.eval(); g_models.pick.eval(); g_models.koikoi.eval();
            g_models.loaded = true;
        }
    }

    ExplorationParams ep{temperature, epsilon, uniform_noise};
    SimConfig config{num_threads, n_envs_per_thread, target_games_per_thread, cap_d, cap_p, cap_k, disc, gae_lambda, dev_str, max_round, ep};
    
    if (g_sim_manager && g_sim_manager->current_config != config) g_sim_manager.reset();
    if (!g_sim_manager) g_sim_manager = std::make_unique<SimulationManager>(config);
    else                g_sim_manager->reset_all();

    {
        py::gil_scoped_release release;
        g_sim_manager->run_simulation_parallel();
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
    m.def("run_arena_batch_simulations", &run_arena_batch_simulations);
    m.def("set_rules", &set_rules);
    m.def("destroy_sim_manager", []() {
        if (g_sim_manager) {
            g_sim_manager.reset();
            std::cout << "[C++] SimulationManager safely destroyed.\n";
        }
    });

    py::class_<KoiKoiStateManager>(m, "KoiKoiStateManager")
        .def(py::init<>())
        .def("init_board", &KoiKoiStateManager::init_board)
        .def("set_state_from_vision", &KoiKoiStateManager::set_state_from_vision)
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
        .def("play_games", &BatchSimulator::play_games)
        .def("finalize_buffers", &BatchSimulator::finalize_buffers);
}