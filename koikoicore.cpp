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

inline constexpr int card_index(int suit, int rank) {
    return (suit - 1) * 4 + (rank - 1);
}

uint64_t cards_to_bitboard(const std::vector<std::vector<int>>& cards) {
    uint64_t bb = 0;
    for (const auto& c : cards) {
        if (c.size() == 2) {
            int idx = card_index(c[0], c[1]);
            if (idx >= 0 && idx < 48) bb |= (1ULL << idx);
        }
    }
    return bb;
}

// ---------------------------------------------------------
// モデルを1度だけGPUにロードし、全スレッドで共有するためのグローバル管理
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
        int idx = it - data_;
        for(int i = idx; i < size_ - 1; ++i) {
            data_[i] = data_[i+1];
        }
        size_--;
    }

    void insert_end(const T* first, const T* last) {
        int n = last - first;
        for(int i = 0; i < n && size_ < Capacity; ++i) {
            data_[size_++] = first[i];
        }
    }

    bool contains(const T& val) const {
        for(int i = 0; i < size_; ++i) if (data_[i] == val) return true;
        return false;
    }
};

static const KoiKoiMasks MASKS;

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

py::array_t<float> cards_to_multi_hot_np(const std::vector<std::vector<int>>& card_list) {
    auto result = py::array_t<float>(48);
    auto buf = result.mutable_unchecked<1>();
    for(py::ssize_t i=0; i<48; i++) buf(i) = 0.0f;
    for (const auto& card : card_list) {
        if (card.size() == 2) {
            int index = card_index(card[0], card[1]);
            if (index >= 0 && index < 48) buf(index) = 1.0f;
        }
    }
    return result;
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
    uint64_t bb_init_board;
    uint64_t bb_unseen;
    uint64_t bb_my_pile;
    uint64_t bb_field;
    uint64_t bb_op_pile;
    uint64_t bb_show;
    uint64_t bb_pairing;
    int point_turn;
    int point_idle;
    int round_num;
    int turn_16;
    int dealer;
    int koikoi_num_turn;
    int koikoi_num_idle;
    uint8_t f_turn;
    uint8_t f_idle;
    uint8_t state_type;
    uint8_t is_draw_pick;
    float potential;
};

struct TraceSlot {
    int state_type; 
    int action;
    Snapshot snap;
};

inline void write_feature_core(
    float* dest, int cols, int off,
    const Snapshot& snap,
    const float* log_buf_ptr, const int* order, const float* cache_ptr,
    bool apply_mask, int action_to_align)
{
    std::memset(dest, 0, 300 * cols * sizeof(float));

    auto sgn = [](float x) { return (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f); };
    float t55[55] = {0.0f};

    // 1. 数学演算の最適化 (std::powの排除とabsの整理)
    float df = static_cast<float>(snap.point_turn - snap.point_idle) * 0.5f;
    float sq_df = std::sqrt(std::abs(df));
    t55[0] = sq_df * sgn(df);
    t55[1] = df * 0.5f;
    t55[2] = df * sq_df * 0.1f;

    float yp_t = static_cast<float>(get_yaku_point_by_bitboard(snap.bb_my_pile, snap.koikoi_num_turn));
    float sq_yp_t = std::sqrt(std::abs(yp_t));
    t55[3] = sq_yp_t * sgn(yp_t);
    t55[4] = yp_t * 0.5f;
    t55[5] = yp_t * sq_yp_t * 0.1f;

    float yp_i = static_cast<float>(get_yaku_point_by_bitboard(snap.bb_op_pile, snap.koikoi_num_idle));
    float sq_yp_i = std::sqrt(std::abs(yp_i));
    t55[6] = sq_yp_i * sgn(yp_i);
    t55[7] = yp_i * 0.5f;
    t55[8] = yp_i * sq_yp_i * 0.1f;

    if(snap.round_num >= 1 && snap.round_num <= 8) t55[9 + snap.round_num - 1] = 1.0f;
    if(snap.turn_16 >= 1 && snap.turn_16 <= 16) t55[17 + snap.turn_16 - 1] = 1.0f;
    if(snap.dealer >= 1 && snap.dealer <= 2) t55[33 + snap.dealer - 1] = 1.0f;

    float k1 = static_cast<float>(snap.koikoi_num_turn); float k2 = static_cast<float>(snap.koikoi_num_idle);
    t55[35] = std::abs(k1) * sgn(k1); t55[36] = (k1 * k1) * sgn(k1);
    t55[37] = std::abs(k2) * sgn(k2); t55[38] = (k2 * k2) * sgn(k2);

    for (int i = 0; i < 8; ++i) {
        t55[39 + i] = static_cast<float>((snap.f_turn >> i) & 1);
        t55[47 + i] = static_cast<float>((snap.f_idle >> i) & 1);
    }

    float yaku_feat[65] = {0.0f};
    const uint64_t m[13] = {
        MASKS.crane, MASKS.curtain, MASKS.moon, MASKS.rainman, MASKS.phoenix, MASKS.sake,
        MASKS.boar_deer_butterfly, MASKS.seed, MASKS.red_ribbon, MASKS.blue_ribbon,
        MASKS.red_blue_ribbon, MASKS.ribbon, MASKS.dross
    };

    for (int i = 0; i < 13; ++i) {
        yaku_feat[i]      = static_cast<float>(popcount(m[i] & snap.bb_hand));
        yaku_feat[i + 13] = static_cast<float>(popcount(m[i] & snap.bb_field));
        yaku_feat[i + 26] = static_cast<float>(popcount(m[i] & snap.bb_my_pile));
        yaku_feat[i + 39] = static_cast<float>(popcount(m[i] & snap.bb_op_pile));
        yaku_feat[i + 52] = static_cast<float>(popcount(m[i] & snap.bb_unseen));
    }

    // 1. t55 特徴量の埋め立て (std::fill_n)
    for (int r = 17; r < 72; ++r) {
        float val = t55[r - 17];
        if (val != 0.0f) {
            std::fill_n(dest + r * cols + off, 48, val);
        }
    }

    // 2. 役特徴量の埋め立て (std::fill_n)
    for (int r = 72; r < 137; ++r) {
        float val = yaku_feat[r - 72];
        if (val != 0.0f) {
            std::fill_n(dest + r * cols + off, 48, val);
        }
    }

    // 3. cache_ptr からの転記 (std::copy_n)
    for (int r = 0; r < 25; ++r) {
        std::copy_n(cache_ptr + r * 48, 48, dest + (137 + r) * cols + off);
    }

    // 4. set_bitsのラムダ式をポインタ直接アクセスに最適化
    auto set_bits = [&](int row, uint64_t bb) {
        float* row_ptr = dest + row * cols + off;
        while (bb) {
#if defined(__GNUC__) || defined(__clang__)
            int c = __builtin_ctzll(bb);
#elif defined(_MSC_VER)
            unsigned long c; _BitScanForward64(&c, bb);
#else
            int c = 0; uint64_t temp = bb; while((temp & 1) == 0) { temp >>= 1; c++; }
#endif
            row_ptr[c] = 1.0f; 
            bb &= bb - 1; 
        }
    };

    set_bits(162, snap.bb_hand); set_bits(165, snap.bb_hand);
    set_bits(163, snap.bb_init_board);
    set_bits(164, snap.bb_unseen); set_bits(169, snap.bb_unseen);
    set_bits(166, snap.bb_my_pile);
    set_bits(167, snap.bb_field);
    set_bits(168, snap.bb_op_pile);
    set_bits(170, snap.bb_show);
    set_bits(171, snap.bb_pairing);

    int out_r = 172;
    for (int j = 0; j < 16; ++j) {
        int t = order[j];
        
        if (apply_mask && t > snap.turn_16) {
            out_r += 8; 
            continue;
        }

        int max_sr = 8;
        if (apply_mask && t == snap.turn_16) {
            if (snap.state_type == 0) {
                max_sr = 0; 
            } else if (snap.state_type == 1) {
                max_sr = snap.is_draw_pick ? 6 : 2;
            }
        }

        for (int sr = 0; sr < 8; ++sr) {
            if (sr < max_sr) {
                std::copy_n(log_buf_ptr + (t * 384) + (sr * 48), 48, dest + (out_r * cols) + off);
            }
            out_r++;
        }
    }

    if (snap.state_type == 2) {
        for(int r = 0; r < 137; ++r) { 
            dest[r * cols + 0] = dest[r * cols + 2]; 
            dest[r * cols + 1] = dest[r * cols + 3]; 
        }
        dest[0 * cols + 0] = 1.0f; dest[1 * cols + 1] = 1.0f;
    }

    // 5. Action Alignment (動的メモリ確保 std::vector の排除)
    if (action_to_align > 0 && action_to_align < cols) {
        float temp[50]; // colsは最大50と保証されているため、高速なスタック配列を使用
        for (int r = 0; r < 300; ++r) {
            float* row_ptr = dest + r * cols;
            std::memcpy(temp, row_ptr, cols * sizeof(float));
            row_ptr[0] = temp[action_to_align];
            std::memcpy(row_ptr + 1, temp, action_to_align * sizeof(float));
        }
    }
}

py::array_t<float> build_feature_fast(
    bool is_koikoi, int point_turn, int point_idle,
    int round_num, int turn_16, int dealer,
    int koikoi_num_turn, int koikoi_num_idle,
    uint8_t f_turn, uint8_t f_idle,
    uint64_t bb_hand, uint64_t bb_init_board, uint64_t bb_unseen,
    uint64_t bb_my_pile, uint64_t bb_field, uint64_t bb_op_pile,
    uint64_t bb_show, uint64_t bb_pairing,
    py::array_t<float>& card_log_buf, 
    py::array_t<py::ssize_t>& order,
    py::array_t<float>& cache_buf) 
{
    int cols = is_koikoi ? 50 : 48;
    int offset = is_koikoi ? 2 : 0;
    
    auto result = py::array_t<float>({300, cols});
    
    Snapshot snap;
    snap.bb_hand = bb_hand; snap.bb_init_board = bb_init_board; snap.bb_unseen = bb_unseen;
    snap.bb_my_pile = bb_my_pile; snap.bb_field = bb_field; snap.bb_op_pile = bb_op_pile;
    snap.bb_show = bb_show; snap.bb_pairing = bb_pairing;
    snap.point_turn = point_turn; snap.point_idle = point_idle;
    snap.round_num = round_num; snap.turn_16 = turn_16; snap.dealer = dealer;
    snap.koikoi_num_turn = koikoi_num_turn; snap.koikoi_num_idle = koikoi_num_idle;
    snap.f_turn = f_turn; snap.f_idle = f_idle;
    snap.state_type = is_koikoi ? 2 : 0;
    snap.is_draw_pick = 0; // python側推論ではリアルタイムなためマスク不要で安全

    auto ord_buf = order.unchecked<1>();
    int c_ord[16];
    for (int i = 0; i < 16; ++i) c_ord[i] = ord_buf(i);

    write_feature_core(
        result.mutable_data(), cols, offset, snap,
        card_log_buf.data(), c_ord, cache_buf.data(),
        false, -1 // マスクなし、アライメントなし
    );

    return result;
}

class KoiKoiStateManager {
public:
    uint64_t bb_hand[3] = {0};
    uint64_t bb_pile[3] = {0};
    uint64_t bb_field = 0;
    uint64_t bb_stock = 0;
    uint64_t bb_initBoard = 0;
    uint64_t bb_show = 0;
    uint64_t bb_pairing = 0;

    KoiKoiStateManager() {}

    void init_board(uint64_t hand1, uint64_t hand2, uint64_t field, uint64_t stock) {
        bb_hand[1] = hand1;
        bb_hand[2] = hand2;
        bb_field = field;
        bb_initBoard = field;
        bb_stock = stock;
        bb_pile[1] = 0;
        bb_pile[2] = 0;
        bb_show = 0;
        bb_pairing = 0;
    }

    void discard(int p, int card, uint64_t pairing) {
        bb_hand[p] &= ~(1ULL << card);
        bb_show = (1ULL << card);
        bb_pairing = pairing;
    }

    void discard_pick(int p, uint64_t collect, uint64_t field_rem) {
        bb_pile[p] |= collect;
        bb_field = field_rem;
        bb_show = 0;
        bb_pairing = 0;
    }

    void draw(int card, uint64_t pairing) {
        bb_stock &= ~(1ULL << card);
        bb_show = (1ULL << card);
        bb_pairing = pairing;
    }

    void draw_pick(int p, uint64_t collect, uint64_t field_rem) {
        bb_pile[p] |= collect;
        bb_field = field_rem;
        bb_show = 0;
        bb_pairing = 0;
    }

    int get_yaku_point(int p, int koikoi_num) {
        return get_yaku_point_by_bitboard(bb_pile[p], koikoi_num);
    }

    py::array_t<float> get_feature(
        bool is_koikoi, int point_turn, int point_idle,
        int round_num, int turn_16, int dealer,
        int koikoi_num_turn, int koikoi_num_idle,
        uint8_t f_turn, uint8_t f_idle,
        int turn_p, int idle_p,
        py::array_t<float>& card_log_buf, 
        py::array_t<py::ssize_t>& order,
        py::array_t<float>& cache_buf) 
    {
        uint64_t un = bb_stock | bb_hand[idle_p];
        return build_feature_fast(
            is_koikoi, point_turn, point_idle,
            round_num, turn_16, dealer,
            koikoi_num_turn, koikoi_num_idle,
            f_turn, f_idle,
            bb_hand[turn_p], bb_initBoard, un,
            bb_pile[turn_p], bb_field, bb_pile[idle_p],
            bb_show, bb_pairing,
            card_log_buf, order, cache_buf
        );
    }

    py::array_t<bool> get_action_mask(int state_type, int p) {
        int size = (state_type == 2) ? 2 : 48;
        auto result = py::array_t<bool>(size);
        auto buf = result.mutable_unchecked<1>();
        
        if (state_type == 2) {
            buf(0) = true;
            buf(1) = true;
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
    
    // std::vector を廃止し、最初から PyTorch テンソルとして保持する
    torch::Tensor states_tensor;
    torch::Tensor actions_tensor;
    torch::Tensor rewards_tensor;

    // 高速アクセス用の生ポインタ
    float* states_ptr;
    int64_t* actions_ptr;
    float* rewards_ptr;

    KoiKoiTraceBuffer(int cap, int r, int c) : capacity(cap), rows(r), cols(c), current_size(0) {
        // ★ メモリ削減＆高速化の要：Pinned Memory でテンソルを1回だけ確保
        auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        auto opt_i64 = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
        
        states_tensor = torch::empty({capacity, rows, cols}, opt_f32);
        actions_tensor = torch::empty({capacity}, opt_i64);
        rewards_tensor = torch::empty({capacity}, opt_f32);

        states_ptr = states_tensor.data_ptr<float>();
        actions_ptr = actions_tensor.data_ptr<int64_t>();
        rewards_ptr = rewards_tensor.data_ptr<float>();
    }

    void clear() { current_size = 0; }

    void push_reconstructed(const Snapshot& snap, int action, float reward,
                            const float* log_buf_ptr, const int* order, const float* cache_ptr) {
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

        // テンソルの生ポインタに直接特徴量を書き込む
        float* dest = states_ptr + idx * rows * cols;
        actions_ptr[idx] = action;
        rewards_ptr[idx] = reward;

        write_feature_core(dest, cols, snap.state_type == 2 ? 2 : 0, snap,
                           log_buf_ptr, order, cache_ptr,
                           true, action);
    }

    py::dict finalize() {
        py::dict result;
        if (current_size == 0) {
            result["states"] = torch::empty({0, rows, cols});
            result["actions"] = torch::empty({0});
            result["rewards"] = torch::empty({0});
        } else {
            // ★ メモリ削減の究極手：Numpy配列を作ってコピーするのをやめ、
            // テンソルの View (スライス) を返すだけにする。
            // これにより、コピーにかかる時間と追加のメモリ消費が「ゼロ」になります。
            result["states"] = states_tensor.slice(0, 0, current_size);
            result["actions"] = actions_tensor.slice(0, 0, current_size);
            result["rewards"] = rewards_tensor.slice(0, 0, current_size);
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

    // 全ての std::vector を FixedVec に置き換え、クリティカルパス上のアロケーションをゼロ化
    FixedVec<int, 48> hand[3];
    FixedVec<int, 48> field_slot;
    FixedVec<int, 48> stock;
    FixedVec<int, 48> pile[3];
    FixedVec<int, 4> show;
    FixedVec<int, 8> collect;

    float card_log_buf[17][8][48];
    FixedVec<int, 4> turn_pairCard[17];
    FixedVec<int, 4> turn_pairCard2[17];

    KoiKoiEnv(int seed = 42) : rng(seed) { reset_game(); }

    void reset_game() {
        round_num = 1; point[1] = 30; point[2] = 30;
        dealer = 1; winner = dealer;
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
        std::memset(card_log_buf, 0, sizeof(card_log_buf));

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

    template<typename Container>
    void set_log_buf(int t16, int idx, const Container& cards) {
        if (t16 < 1 || t16 > 16) return;
        std::memset(card_log_buf[t16][idx], 0, 48 * sizeof(float));
        for (int c : cards) if (c >= 0 && c < 48) card_log_buf[t16][idx][c] = 1.0f;
    }

    FixedVec<int, 4> get_pairing_cards() const {
        FixedVec<int, 4> pairs;
        if (show.empty()) return pairs;
        int target = show[0] / 4;
        for (int c : field_slot) if (c != -1 && c / 4 == target) pairs.push_back(c);
        return pairs;
    }

    FixedVec<int, 8> field_collect() const {
        FixedVec<int, 8> fc = collect;
        for (auto it = fc.begin(); it != fc.end(); ++it) {
            if (*it == show[0]) {
                fc.erase(it);
                break;
            }
        }
        return fc;
    }

    FixedVec<int, 4> get_uncollect(const FixedVec<int, 4>& pairing, const FixedVec<int, 8>& fc) const {
        FixedVec<int, 4> un;
        for (int c : pairing) {
            if (!fc.contains(c)) un.push_back(c);
        }
        return un;
    }

    void _collect_card(int card) {
        FixedVec<int, 4> pairing = get_pairing_cards();
        if (pairing.empty()) {
            collect.clear();
            for (int i = 0; i < field_slot.size(); ++i) {
                if (field_slot[i] == -1) {
                    field_slot[i] = show[0];
                    break;
                }
            }
        } else if (pairing.size() == 1 || pairing.size() == 3) {
            collect.clear();
            collect.insert_end(show.begin(), show.end());
            collect.insert_end(pairing.begin(), pairing.end());
            for (int p_card : pairing) {
                for (int i = 0; i < field_slot.size(); ++i) {
                    if (field_slot[i] == p_card) {
                        field_slot[i] = -1;
                        break;
                    }
                }
            }
            pile[turn_player()].insert_end(collect.begin(), collect.end());
        } else {
            collect.clear();
            collect.insert_end(show.begin(), show.end());
            collect.push_back(card);
            for (int i = 0; i < field_slot.size(); ++i) {
                if (field_slot[i] == card) {
                    field_slot[i] = -1;
                    break;
                }
            }
            pile[turn_player()].insert_end(collect.begin(), collect.end());
        }
    }

    void step(int action) {
        int p = turn_player();
        if (state == GameState::DISCARD) {
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
            state_manager.discard(p, action, pair_bb);

            set_log_buf(turn_16, pairing.empty() ? 1 : 0, show);
            turn_pairCard[turn_16] = pairing;
            state = GameState::DISCARD_PICK; wait_action = (pairing.size() == 2);
        }
        else if (state == GameState::DISCARD_PICK) {
            _collect_card(action);
            uint64_t coll_bb = 0; for (int c : collect) coll_bb |= (1ULL << c);
            uint64_t field_bb = 0; for (int c : field_slot) if (c != -1) field_bb |= (1ULL << c);
            state_manager.discard_pick(p, coll_bb, field_bb);

            if (!collect.empty()) {
                FixedVec<int, 8> fc = field_collect();
                set_log_buf(turn_16, 2, fc);
                set_log_buf(turn_16, 3, get_uncollect(turn_pairCard[turn_16], fc));
            }
            state = GameState::DRAW; wait_action = false;
        }
        else if (state == GameState::DRAW) {
            int drawn = stock.back(); stock.pop_back(); 
            show.clear();
            show.push_back(drawn);
            
            FixedVec<int, 4> pairing = get_pairing_cards();
            uint64_t pair_bb = 0; for (int c : pairing) pair_bb |= (1ULL << c);
            state_manager.draw(drawn, pair_bb);

            set_log_buf(turn_16, pairing.empty() ? 5 : 4, show);
            turn_pairCard2[turn_16] = pairing;
            state = GameState::DRAW_PICK; wait_action = (pairing.size() == 2);
        }
        else if (state == GameState::DRAW_PICK) {
            _collect_card(action);
            uint64_t coll_bb = 0; for (int c : collect) coll_bb |= (1ULL << c);
            uint64_t field_bb = 0; for (int c : field_slot) if (c != -1) field_bb |= (1ULL << c);
            state_manager.draw_pick(p, coll_bb, field_bb);

            if (!collect.empty()) {
                FixedVec<int, 8> fc = field_collect();
                set_log_buf(turn_16, 6, fc);
                set_log_buf(turn_16, 7, get_uncollect(turn_pairCard2[turn_16], fc));
            }
            state = GameState::KOIKOI;
            int pt = state_manager.get_yaku_point(p, koikoi_num(p));
            wait_action = (pt > turn_point) && (turn_8() < 8);
        }
        else if (state == GameState::KOIKOI) {
            int pt = state_manager.get_yaku_point(p, koikoi_num(p));
            bool is_koikoi = (action != 0);
            if ((pt > turn_point) && (turn_8() == 8)) is_koikoi = false;
            if (wait_action) koikoi[p][turn_8() - 1] = is_koikoi ? 1 : 0;

            if (!is_koikoi) { state = GameState::ROUND_OVER; wait_action = false; winner = p; }
            else if (turn_16 == 16) { state = GameState::ROUND_OVER; wait_action = false; exhausted = true; winner = dealer; }
            else { turn_16++; state = GameState::DISCARD; wait_action = true; }
        }
    }

private:
    void deal_card() {
        int cards[48]; // スタック確保
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
            for (int i = 0; i < 8; ++i) field_slot.push_back(-1); // -1で空きスロットを表現
            
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
    std::vector<float> win_prob_mat;
    float cache_buf[25][48];
    int order[17][16];

    int finished_games = 0;
    torch::Device device;

    torch::Tensor feat_tensor[3];
    torch::Tensor mask_tensor[3];

    BatchSimulator(int n_envs, int target, int cap_d, int cap_p, int cap_k, float disc, py::array_t<float> wp_mat, std::string dev_str)
        : num_envs(n_envs), target_games(target), discount(disc),
          buf_discard(cap_d, 300, 48), buf_pick(cap_p, 300, 48), buf_koikoi(cap_k, 300, 50),
          device(dev_str)
    {
        envs.resize(n_envs);
        for(int p=1; p<=2; ++p) {
            traces[p].resize(n_envs);
        }

        auto wp = wp_mat.unchecked<3>();
        win_prob_mat.resize(2 * 9 * 61, 0.0f);
        for(int i=0; i<2; ++i) for(int j=0; j<9; ++j) for(int k=0; k<61; ++k)
            win_prob_mat[(i*9+j)*61+k] = static_cast<float>(wp(i, j, k));

        std::memset(cache_buf, 0, sizeof(cache_buf));
        std::vector<std::vector<std::pair<int,int>>> dict = {
            {{1,1}}, {{3,1}}, {{8,1}}, {{11,1}}, {{12,1}}, {{9,1}},
            {{6,1},{7,1},{10,1}},
            {{2,1},{4,1},{5,1},{6,1},{7,1},{8,2},{9,1},{10,1},{11,2}},
            {{1,2},{2,2},{3,2}}, {{6,2},{9,2},{10,2}},
            {{1,2},{2,2},{3,2},{6,2},{9,2},{10,2}},
            {{1,2},{2,2},{3,2},{4,2},{5,2},{6,2},{7,2},{9,2},{10,2},{11,3}},
            {{1,3},{1,4},{2,3},{2,4},{3,3},{3,4},{4,3},{4,4},{5,3},{5,4},{6,3},{6,4},{7,3},{7,4},{8,3},{8,4},{9,3},{9,4},{10,3},{10,4},{11,4},{12,2},{12,3},{12,4},{9,1}}
        };
        for(size_t i=0; i<dict.size(); ++i) for(auto c : dict[i]) cache_buf[i][(c.first-1)*4 + (c.second-1)] = 1.0f;
        for(int i=0; i<12; ++i) for(int j=0; j<4; ++j) cache_buf[13+i][i*4+j] = 1.0f;

        for(int i=1; i<=16; ++i) {
            int idx = 0;
            for(int x=i; x>=1; --x) order[i][idx++] = x;
            for(int x=i+1; x<=16; ++x) order[i][idx++] = x;
        }

        auto opt_f32_pinned = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        auto opt_bool_pinned = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);

        feat_tensor[0] = torch::empty({num_envs, 300, 48}, opt_f32_pinned);
        feat_tensor[1] = torch::empty({num_envs, 300, 48}, opt_f32_pinned);
        feat_tensor[2] = torch::empty({num_envs, 300, 50}, opt_f32_pinned);

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

    void build_feature_into(int i, float* feat, int type) {
        auto& env = envs[i]; 
        bool is_k = (type == 2);
        int cols = is_k ? 50 : 48; 
        int off = is_k ? 2 : 0;
        
        std::memset(feat, 0, 300 * cols * sizeof(float));

        int pt = env.turn_player(); int pi = env.idle_player();
        auto sgn = [](float x) { return (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f); };
        
        float t55[55] = {0.0f};

        // 1. 点数差 (df) の計算
        float df = static_cast<float>(env.point[pt] - env.point[pi]) * 0.5f;
        float sq_df = std::sqrt(std::abs(df)); // 1度だけ計算して使い回す
        t55[0] = sq_df * sgn(df);
        t55[1] = df * 0.5f;                    // |df| * sgn(df) は df と等価
        t55[2] = df * sq_df * 0.1f;            // pow(abs, 1.5) * sgn は df * sqrt(abs) と等価

        // 2. ターンプレイヤーの役ポイント (yp_t) の計算（都度計算）
        float yp_t = static_cast<float>(env.state_manager.get_yaku_point(pt, env.koikoi_num(pt)));
        float sq_yp_t = std::sqrt(std::abs(yp_t));
        t55[3] = sq_yp_t * sgn(yp_t);
        t55[4] = yp_t * 0.5f;
        t55[5] = yp_t * sq_yp_t * 0.1f;

        // 3. アイドルプレイヤーの役ポイント (yp_i) の計算（都度計算）
        float yp_i = static_cast<float>(env.state_manager.get_yaku_point(pi, env.koikoi_num(pi)));
        float sq_yp_i = std::sqrt(std::abs(yp_i));
        t55[6] = sq_yp_i * sgn(yp_i);
        t55[7] = yp_i * 0.5f;
        t55[8] = yp_i * sq_yp_i * 0.1f;

        if(env.round_num <= 8) t55[9 + env.round_num - 1] = 1.0f;
        if(env.turn_16 <= 16) t55[17 + env.turn_16 - 1] = 1.0f;
        t55[33 + env.dealer - 1] = 1.0f;

        t55[35] = static_cast<float>(env.koikoi_num(pt)); t55[36] = t55[35]*t55[35];
        t55[37] = static_cast<float>(env.koikoi_num(pi)); t55[38] = t55[37]*t55[37];
        for(int j=0; j<8; ++j) { t55[39+j] = static_cast<float>(env.koikoi[pt][j]); t55[47+j] = static_cast<float>(env.koikoi[pi][j]); }

        uint64_t bb_h = env.state_manager.bb_hand[pt]; uint64_t bb_f = env.state_manager.bb_field;
        uint64_t bb_m = env.state_manager.bb_pile[pt]; uint64_t bb_o = env.state_manager.bb_pile[pi];
        uint64_t bb_u = env.state_manager.bb_stock | env.state_manager.bb_hand[pi];
        float y_f[65] = {0.0f}; const uint64_t ms[13] = { MASKS.crane, MASKS.curtain, MASKS.moon, MASKS.rainman, MASKS.phoenix, MASKS.sake, MASKS.boar_deer_butterfly, MASKS.seed, MASKS.red_ribbon, MASKS.blue_ribbon, MASKS.red_blue_ribbon, MASKS.ribbon, MASKS.dross };
        for(int j=0; j<13; ++j) { y_f[j]=static_cast<float>(popcount(ms[j]&bb_h)); y_f[j+13]=static_cast<float>(popcount(ms[j]&bb_f)); y_f[j+26]=static_cast<float>(popcount(ms[j]&bb_m)); y_f[j+39]=static_cast<float>(popcount(ms[j]&bb_o)); y_f[j+52]=static_cast<float>(popcount(ms[j]&bb_u)); }

        auto set_v = [&](int r, int c, float v) { feat[r * cols + c + off] = v; };
        for(int r=17; r<72; ++r) {
            if(t55[r-17] != 0.0f) {
                std::fill_n(feat + r * cols + off, 48, t55[r-17]);
            }
        }
        for(int r=72; r<137; ++r) {
            if(y_f[r-72] != 0.0f) {
                std::fill_n(feat + r * cols + off, 48, y_f[r-72]);
            }
        }
        // cache_buf[r] からのコピーは std::copy_n の方が高速かつ安全です
        for(int r=0; r<25; ++r) {
            std::memcpy(feat + (137 + r) * cols + off, cache_buf[r], 48 * sizeof(float));
        }

        // ラムダ式を排し、直接ポインタに書き込むマクロまたはインライン関数化
        auto set_b_fast = [&](int row, uint64_t bb) {
            float* row_ptr = feat + row * cols + off;
            while(bb) {
#if defined(__GNUC__) || defined(__clang__)
                int c = __builtin_ctzll(bb);
#elif defined(_MSC_VER)
                unsigned long c; _BitScanForward64(&c, bb);
#else
                int c=0; uint64_t t=bb; while((t&1)==0){ t>>=1; c++; }
#endif
                row_ptr[c] = 1.0f; // 2次元インデックス計算を排除
                bb &= bb-1;
            }
        };

        set_b_fast(162, bb_h); set_b_fast(165, bb_h); set_b_fast(163, env.state_manager.bb_initBoard);
        set_b_fast(164, bb_u); set_b_fast(169, bb_u); set_b_fast(166, bb_m); set_b_fast(167, bb_f);
        set_b_fast(168, bb_o); set_b_fast(170, env.state_manager.bb_show); set_b_fast(171, env.state_manager.bb_pairing);

        int out_r = 172, t16 = std::max(1, std::min(16, env.turn_16));
        for(int j=0; j<16; ++j) {
            int t = order[t16][j];
            for(int sr=0; sr<8; ++sr) {
                // 修正: ログバッファからの転写も std::memcpy に統一
                std::memcpy(feat + out_r * cols + off, &env.card_log_buf[t][sr][0], 48 * sizeof(float));
                out_r++;
            }
        }
        if(is_k) {
            for(int r=0; r<137; ++r) { feat[r*cols+0] = feat[r*cols+2]; feat[r*cols+1] = feat[r*cols+3]; }
            feat[0*cols+0] = 1.0f; feat[1*cols+1] = 1.0f;
        }
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

    float get_potential(int p_idx, int n_rnd, int n_pt, int dealer, int winner, bool exhausted) {
        // ラウンド終了時またはゲーム終了時
        if (n_rnd > 8 || n_pt <= 0 || n_pt >= 60) {
            if (n_pt <= 0) return 0.0f;       // 負け確定
            if (n_pt >= 60) return 1.0f;      // 勝ち確定
            if (n_rnd > 8) {
                if (n_pt == 30) return 0.5f;  // 引き分け
                return (n_pt > 30) ? 1.0f : 0.0f; // 点数が多い方が勝ち
            }
        }
        // ゲーム継続中：勝率テーブル（win_prob_mat）を参照
        int is_d = (exhausted ? dealer : winner) == p_idx ? 1 : 0;
        return win_prob_mat[(is_d * 9 + n_rnd) * 61 + n_pt];
    }

    void process_round_over(int i) {
        auto& env = envs[i];
        
        // P1の経験のみを抽出
        auto& trace = traces[1][i];
        int final_pt = env.point[1] + env.round_point(1);
        float final_potential = get_potential(1, env.round_num + 1, final_pt, env.dealer, env.winner, env.exhausted);
        
        // 修正点: モンテカルロリターンとして最終ポテンシャルから開始
        float mc_return = final_potential; 
        for (size_t rs = 0; rs < trace.size(); ++rs) {
            auto& slot = trace[trace.size() - 1 - rs];
            float current_potential = slot.snap.potential;

            // 修正点: 現在のポテンシャルから、最終結果（割引済み）への差分を報酬とする
            float pbrs_reward = (mc_return - current_potential) * 10.0f;
        //    float pbrs_reward = mc_return * 10.0f; # 最終結果だけの評価
            
            // 削除してしまっていた変数の定義を復元
            int t16_limit = std::max<int>(1, std::min<int>(16, slot.snap.turn_16));
            const float* log_ptr = &env.card_log_buf[0][0][0];
            const float* cache_ptr = &cache_buf[0][0];

            if (slot.state_type == 0) buf_discard.push_reconstructed(slot.snap, slot.action, pbrs_reward, log_ptr, order[t16_limit], cache_ptr);
            else if (slot.state_type == 1) buf_pick.push_reconstructed(slot.snap, slot.action, pbrs_reward, log_ptr, order[t16_limit], cache_ptr);
            else buf_koikoi.push_reconstructed(slot.snap, slot.action, pbrs_reward, log_ptr, order[t16_limit], cache_ptr);
            
            // 修正点: 時間を遡るごとに割引を適用
            mc_return = discount * mc_return;
        }
        traces[1][i].clear();
        traces[2][i].clear(); // P2の経験は学習しないため捨てる
        
        env.point[1] += env.round_point(1); env.point[2] += env.round_point(2);
    }

    void play_games() {
        std::vector<int> req_env_idx[3];
        float* feat_ptrs[3] = { feat_tensor[0].data_ptr<float>(), feat_tensor[1].data_ptr<float>(), feat_tensor[2].data_ptr<float>() };
        bool* mask_ptrs[3] = { mask_tensor[0].data_ptr<bool>(), mask_tensor[1].data_ptr<bool>(), mask_tensor[2].data_ptr<bool>() };

        while (finished_games < target_games) {
            int local_finished = 0;

            // 1. 環境のステップ進行を OpenMP で全コア並列処理！
            #pragma omp parallel for reduction(+:local_finished)
            for (int i = 0; i < num_envs; ++i) {
                auto& env = envs[i];
                while (!env.needs_action() && env.state != GameState::GAME_OVER) {
                    if (env.state == GameState::ROUND_OVER) {
                        process_round_over(i);
                        if (env.point[1] <= 0 || env.point[2] <= 0 || env.round_num == 8) {
                            local_finished++;
                            // ターゲット到達の判定は後で行うため、ここでは無条件でリセットする
                            env.reset_game(); env.reset_round();
                        } else { env.round_num++; env.reset_round(); }
                    } else { env.step(-1); }
                }
            }

            finished_games += local_finished;
            if (finished_games >= target_games) break;

            // P1とP2の独立した推論ループ
            for (int p_id = 1; p_id <= 2; ++p_id) {
                for(int i=0; i<3; ++i) req_env_idx[i].clear();
                
                // 2. 対象となる環境インデックスの収集 (非常に軽い処理なので直列)
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

                    // 3. 重い特徴量・マスクの構築を OpenMP で全コア並列処理！
                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) {
                        int env_idx = req_env_idx[type][i];
                        int cols = (type == 2) ? 50 : 48;
                        int m_cols = (type == 2) ? 2 : 48;
                        build_feature_into(env_idx, feat_ptrs[type] + i * 300 * cols, type);
                        build_mask_into(env_idx, mask_ptrs[type] + i * m_cols, type);
                    }

                    // 4. GPU推論はただ1つのスレッドがまとめて担当！
                //    torch::Tensor f_gpu = feat_tensor[type].slice(0, 0, n).to(device, true).to(torch::kBFloat16);
                    torch::Tensor f_gpu = feat_tensor[type].slice(0, 0, n).to(device, true).to(torch::kFloat32);
                    
                    // GPUにはモデルの forward のみを任せる
                    torch::Tensor output;
                    if (type == 0) output = g_models.discard.forward({f_gpu}).toTensor();
                    else if (type == 1) output = g_models.pick.forward({f_gpu}).toTensor();
                    else output = g_models.koikoi.forward({f_gpu}).toTensor();

                    // ★ CPU負荷削減の要: 推論結果を直ちにCPU(Float32)に持ってくる
                    // これ以降の細かい分岐や乱数生成は、GPUカーネルを起動するよりCPUで行った方が圧倒的に速い
                    torch::Tensor output_cpu = output.cpu();
                    float* out_ptr = output_cpu.data_ptr<float>();
                    bool* mask_ptr = mask_tensor[type].data_ptr<bool>(); // すでにPinnedメモリ上にある

                    // 結果を格納するスタック上の軽量配列
                    int64_t act_ptr[8192]; 
                    float epsilon = 0.1f;
                    
                    // 乱数生成器は環境0のものを借用 (CPU上で高速に乱数を生成)
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

                    for (int i = 0; i < n; ++i) {
                        int cols = (type == 2) ? 2 : 48;
                        
                        if (dist(envs[0].rng) < epsilon) {
                            // 探索 (Explore): 有効なアクションからランダム選択
                            FixedVec<int, 48> valid_actions;
                            for (int c = 0; c < cols; ++c) {
                                if (mask_ptr[i * cols + c]) valid_actions.push_back(c);
                            }
                            if (!valid_actions.empty()) {
                                std::uniform_int_distribution<int> action_dist(0, valid_actions.size() - 1);
                                act_ptr[i] = valid_actions[action_dist(envs[0].rng)];
                            } else {
                                act_ptr[i] = 0; // フェールセーフ
                            }
                        } else {
                            // 活用 (Exploit): Argmax
                            float max_val = -1e9f;
                            int best_a = 0;
                            for (int c = 0; c < cols; ++c) {
                                if (mask_ptr[i * cols + c]) {
                                    float val = out_ptr[i * cols + c];
                                    if (val > max_val) {
                                        max_val = val;
                                        best_a = c;
                                    }
                                }
                            }
                            act_ptr[i] = best_a;
                        }
                    }

                    // 5. アクションの適用と軌跡の記録 (std::vector への push_back があるため直列が安全)
                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) {
                        int env_idx = req_env_idx[type][i];
                        int action = static_cast<int>(act_ptr[i]);
                        auto& env = envs[env_idx];

                        // P1 (学習側) のみ軌跡を記録する
                        if (p_id == 1) {
                            TraceSlot slot;
                            slot.state_type = type;
                            slot.action = action;
                            Snapshot& snap = slot.snap;
                            snap.bb_hand = env.state_manager.bb_hand[1];
                            snap.bb_init_board = env.state_manager.bb_initBoard;
                            snap.bb_unseen = env.state_manager.bb_stock | env.state_manager.bb_hand[2];
                            snap.bb_my_pile = env.state_manager.bb_pile[1];
                            snap.bb_field = env.state_manager.bb_field;
                            snap.bb_op_pile = env.state_manager.bb_pile[2];
                            snap.bb_show = env.state_manager.bb_show;
                            snap.bb_pairing = env.state_manager.bb_pairing;
                            snap.point_turn = env.point[1];
                            snap.point_idle = env.point[2];
                            snap.round_num = env.round_num;
                            snap.turn_16 = env.turn_16;
                            snap.dealer = env.dealer;
                            snap.koikoi_num_turn = env.koikoi_num(1);
                            snap.koikoi_num_idle = env.koikoi_num(2);
                            snap.f_turn = 0; for(int j=0; j<8; ++j) if(env.koikoi[1][j]) snap.f_turn |= (1 << j);
                            snap.f_idle = 0; for(int j=0; j<8; ++j) if(env.koikoi[2][j]) snap.f_idle |= (1 << j);
                            snap.state_type = type;
                            snap.is_draw_pick = (env.state == GameState::DRAW_PICK) ? 1 : 0;
                            snap.potential = get_potential(1, env.round_num, env.point[1], env.dealer, env.winner, env.exhausted);
                            traces[1][env_idx].push_back(slot);
                        }
                        env.step(action);
                    }
                }
            }
        }
    }
};

class KoiKoiTrainer {
private:
    torch::jit::script::Module model;
    std::shared_ptr<torch::optim::Adam> optimizer;
    torch::Device device;

public:
    KoiKoiTrainer(const std::string& model_path, float lr, const std::string& dev_str) 
        : device(dev_str) {
        try {
            model = torch::jit::load(model_path, device);
            model.train();
            
            std::vector<torch::Tensor> parameters;
            for (const auto& p : model.parameters()) {
                if (p.requires_grad()) {
                    parameters.push_back(p);
                }
            }
            optimizer = std::make_shared<torch::optim::Adam>(
                parameters, torch::optim::AdamOptions(lr));
        } catch (const c10::Error& e) {
            (void)e;
            std::cerr << "Error loading JIT model for training: " << model_path << "\n";
            throw;
        }
    }

    float train_epoch(torch::Tensor states, torch::Tensor rewards, int batch_size) {
        int64_t num_samples = states.size(0);
        int rows = states.size(1);
        int cols = states.size(2);
        
        auto indices = torch::randperm(num_samples, torch::TensorOptions().dtype(torch::kLong));
        const int64_t* idx_ptr = indices.data_ptr<int64_t>();
        
        float total_loss = 0.0f;
        int num_batches = 0;

        // ★ VRAMスワップ撲滅の要：マイクロバッチ化
        // 4096をそのままGPUに送るのではなく、512ずつに分割して処理・累積します
        int64_t micro_batch_size = 1024; 
        
        auto opt_pinned = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        
        // ★ Pinned Memory のサイズも 1/8 に激減し、メインメモリ・共有VRAMの占有が消滅します
        torch::Tensor batch_states[2] = {
            torch::empty({micro_batch_size, rows, cols}, opt_pinned),
            torch::empty({micro_batch_size, rows, cols}, opt_pinned)
        };
        torch::Tensor batch_rewards[2] = {
            torch::empty({micro_batch_size}, opt_pinned),
            torch::empty({micro_batch_size}, opt_pinned)
        };
        
        const float* src_states_ptr = states.data_ptr<float>();
        const float* src_rewards_ptr = rewards.data_ptr<float>();

        py::gil_scoped_release release;
        int buf_idx = 0;

        for (int64_t i = 0; i < num_samples; i += batch_size) {
            int64_t current_batch_size = std::min(static_cast<int64_t>(batch_size), num_samples - i);
            optimizer->zero_grad();
            
            float batch_loss = 0.0f;

            // 巨大なバッチをマイクロバッチに分割してループ
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

                auto state_gpu = batch_states[buf_idx].slice(0, 0, m_size)
                                             .to(torch::kFloat32)
                                             .to(device, /*non_blocking=*/true);

            //    auto state_gpu = batch_states[buf_idx].slice(0, 0, m_size)
            //                                 .to(torch::kBFloat16)
            //                                 .to(device, /*non_blocking=*/true);
                                             
                auto reward_gpu = batch_rewards[buf_idx].slice(0, 0, m_size)
                                               .to(device, /*non_blocking=*/true);

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(state_gpu);
                
                torch::Tensor q_values = model.forward(inputs).toTensor().view({-1});
                torch::Tensor reward_flat = reward_gpu.view({-1});
                
                auto loss = torch::nn::functional::smooth_l1_loss(
                    q_values,
                    reward_flat, 
                    torch::nn::functional::SmoothL1LossFuncOptions().beta(30.0)
                );

                // 全体バッチに対するこのマイクロバッチの割合を計算して勾配をスケール
                // （数学的に 4096を一括処理したのと全く同じ更新量になります）
                float scale = static_cast<float>(m_size) / current_batch_size;
                auto scaled_loss = loss * scale;
                
                // backward() を呼ぶと、計算が終わった不要な中間データが即座にVRAMから解放されます
                scaled_loss.backward();

                batch_loss += loss.item<float>() * scale;
                buf_idx = 1 - buf_idx; // ダブルバッファの切り替え
            }

            // マイクロバッチ全ての累積が終わってから1回だけ重みを更新
            optimizer->step();
            total_loss += batch_loss;
            num_batches++;
        }
        
        return num_batches > 0 ? total_loss / num_batches : 0.0f;
    }

    void save_model(const std::string& save_path) {
        model.save(save_path);
    }

    float train_from_results(py::list results, const std::string& key, int batch_size) {
        std::vector<torch::Tensor> state_list;
        std::vector<torch::Tensor> reward_list;
        
        for (auto item : results) {
            py::dict res_dict = item.cast<py::dict>();
            if (!res_dict.contains(key.c_str())) continue;
            
            py::dict data = res_dict[key.c_str()].cast<py::dict>();
            
            // ★ 高速化: Numpy (py::array_t) を経由せず、最初から torch::Tensor として受け取る
            torch::Tensor s_tensor = data["states"].cast<torch::Tensor>();
            torch::Tensor r_tensor = data["rewards"].cast<torch::Tensor>();
            
            if (s_tensor.size(0) > 0) {
                // clone() を削除。すでに Pinned Memory に乗っているテンソルをそのままリストに入れる
                state_list.push_back(s_tensor);
                reward_list.push_back(r_tensor);
            }
        }
        
        if (state_list.empty()) return 0.0f;
        
        // 分割されている View を結合して1つの学習用バッチにする
        // ここで初めて1回だけメモリ結合が走りますが、全体を通したピークメモリは圧倒的に下がります
        auto all_states = torch::cat(state_list, 0);
        auto all_rewards = torch::cat(reward_list, 0);
        
        return train_epoch(all_states, all_rewards, batch_size);
    }

    void sync_and_save_action_model(const std::string& action_model_path) {
        try {
            auto action_model = torch::jit::load(action_model_path, device);
            torch::NoGradGuard no_grad;
            
            auto src_params = model.parameters();
            auto dst_params = action_model.parameters();
            
            auto src_it = src_params.begin();
            auto dst_it = dst_params.begin();
            while (src_it != src_params.end() && dst_it != dst_params.end()) {
                (*dst_it).copy_(*src_it);
                ++src_it;
                ++dst_it;
            }
            
            action_model.save(action_model_path);
        } catch (const c10::Error& e) {
            (void)e;
            std::cerr << "Error syncing weights to " << action_model_path << "\n";
        }
    }

    void sync_to_inference_model(const std::string& type) {
        torch::NoGradGuard no_grad;
        // 推論用モデルのロックを取得
        std::lock_guard<std::mutex> lock(g_models.mtx);
        
        torch::jit::script::Module* target_model = nullptr;
        if (type == "discard") target_model = &g_models.discard;
        else if (type == "pick") target_model = &g_models.pick;
        else if (type == "koikoi") target_model = &g_models.koikoi;
        
        if (target_model && g_models.loaded) {
            auto src_params = model.parameters();
            auto dst_params = target_model->parameters();
            auto src_it = src_params.begin();
            auto dst_it = dst_params.begin();
            while (src_it != src_params.end() && dst_it != dst_params.end()) {
                // ★VRAM内での直接コピー。ディスクI/Oゼロで一瞬で終わります。
                (*dst_it).copy_(*src_it);
                ++src_it;
                ++dst_it;
            }
        }
    }
};


// ==========================================
// SimulationManager による Thread Pool の永続化・再利用
// ==========================================
struct SimConfig {
    int num_threads, n_envs, target, cap_d, cap_p, cap_k;
    float disc;
    std::string dev_str;

    bool operator==(const SimConfig& o) const {
        return num_threads == o.num_threads && n_envs == o.n_envs &&
               target == o.target && cap_d == o.cap_d && cap_p == o.cap_p &&
               cap_k == o.cap_k && disc == o.disc && dev_str == o.dev_str;
    }
    bool operator!=(const SimConfig& o) const { return !(*this == o); }
};

class SimulationManager {
public:
    std::vector<std::unique_ptr<BatchSimulator>> sims;
    SimConfig current_config;

    // Thread Pool 用の同期変数
    std::vector<std::thread> workers;
    std::mutex pool_mtx;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    
    int generation = 0;
    bool shutdown = false;
    int done_workers = 0;
    
    std::vector<std::exception_ptr> worker_exceptions;

    SimulationManager(const SimConfig& config, py::array_t<float>& wp_mat) 
        : current_config(config), worker_exceptions(config.num_threads, nullptr) 
    {
        for (int i = 0; i < config.num_threads; ++i) {
            sims.push_back(std::make_unique<BatchSimulator>(
                config.n_envs, config.target, config.cap_d, config.cap_p, config.cap_k, 
                config.disc, wp_mat, config.dev_str
            ));
        }

        for (int i = 0; i < config.num_threads; ++i) {
            workers.emplace_back([this, i]() {
                int local_generation = 0;
                while (true) {
                    {
                        std::unique_lock<std::mutex> lock(this->pool_mtx);
                        this->cv_start.wait(lock, [this, local_generation]() { 
                            return this->generation > local_generation || this->shutdown; 
                        });
                        
                        if (this->shutdown && this->generation <= local_generation) {
                            break;
                        }
                        local_generation = this->generation;
                    }

                    try {
                        this->sims[i]->play_games();
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(this->pool_mtx);
                        this->worker_exceptions[i] = std::current_exception();
                    }

                    {
                        std::lock_guard<std::mutex> lock(this->pool_mtx);
                        this->done_workers++;
                        if (this->done_workers == this->current_config.num_threads) {
                            this->cv_done.notify_one();
                        }
                    }
                }
            });
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
            cv_done.wait(lock, [this]() { return this->done_workers == this->current_config.num_threads; });
        }

        for (auto& e : worker_exceptions) {
            if (e) {
                std::rethrow_exception(e);
            }
        }
    }
};

static std::unique_ptr<SimulationManager> g_sim_manager = nullptr;

py::list run_parallel_simulations(
    int num_threads, int n_envs_per_thread, int target_games_per_thread,
    int cap_d, int cap_p, int cap_k, float disc, py::array_t<float> wp_mat,
    std::string path_discard, std::string path_pick, std::string path_koikoi, std::string dev_str) 
{
    torch::Device device(dev_str);

    {
        std::lock_guard<std::mutex> lock(g_models.mtx);
        // ★修正: loaded が false の時（初回）だけロードするように変更
        if (!g_models.loaded) {
            g_models.discard = torch::jit::load(path_discard, device);
            g_models.pick = torch::jit::load(path_pick, device);
            g_models.koikoi = torch::jit::load(path_koikoi, device);
            g_models.discard.eval(); 
            g_models.pick.eval(); 
            g_models.koikoi.eval();
            g_models.loaded = true;
        }
    }

    SimConfig config = {num_threads, n_envs_per_thread, target_games_per_thread, cap_d, cap_p, cap_k, disc, dev_str};
    
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
    
    m.def("cards_to_bitboard", &cards_to_bitboard);
    m.def("get_yaku_point_by_bitboard", &get_yaku_point_by_bitboard);
    m.def("evaluate_yaku_by_bitboard", &evaluate_yaku_by_bitboard);
    m.def("cards_to_multi_hot_np", &cards_to_multi_hot_np);
    m.def("build_feature_fast", &build_feature_fast);
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
        .def("get_feature", &KoiKoiStateManager::get_feature)
        .def("get_action_mask", &KoiKoiStateManager::get_action_mask)
        .def("get_best_action", &KoiKoiStateManager::get_best_action);

    py::class_<KoiKoiTraceBuffer>(m, "KoiKoiTraceBuffer")
        .def(py::init<int, int, int>(), py::arg("capacity"), py::arg("rows"), py::arg("cols"))
        .def("clear", &KoiKoiTraceBuffer::clear)
        .def("finalize", &KoiKoiTraceBuffer::finalize);

    py::class_<BatchSimulator>(m, "BatchSimulator")
        .def(py::init<int, int, int, int, int, float, py::array_t<float>, std::string>())
        .def("play_games", &BatchSimulator::play_games)
        .def("finalize_buffers", &BatchSimulator::finalize_buffers);

    py::class_<KoiKoiTrainer>(m, "KoiKoiTrainer")
        .def(py::init<std::string, float, std::string>())
        .def("train_epoch", &KoiKoiTrainer::train_epoch)
        .def("save_model", &KoiKoiTrainer::save_model)
        .def("train_from_results", &KoiKoiTrainer::train_from_results)
        .def("sync_and_save_action_model", &KoiKoiTrainer::sync_and_save_action_model)
        .def("sync_to_inference_model", &KoiKoiTrainer::sync_to_inference_model);
}