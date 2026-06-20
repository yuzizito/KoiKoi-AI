#include <torch/extension.h>
#include <torch/script.h>
#include <torch/optim.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
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

py::array_t<float> get_yaku_status_features_np(uint64_t mh, uint64_t b, uint64_t mc, uint64_t oc, uint64_t unseen) {
    auto result = py::array_t<float>(65);
    auto buf = result.mutable_unchecked<1>();

    // 役を判定するためのマスクビットボード
    const uint64_t m[13] = {
        MASKS.crane, MASKS.curtain, MASKS.moon, MASKS.rainman, MASKS.phoenix, MASKS.sake,
        MASKS.boar_deer_butterfly, MASKS.seed, MASKS.red_ribbon, MASKS.blue_ribbon,
        MASKS.red_blue_ribbon, MASKS.ribbon, MASKS.dross
    };

    // 各役の popcount をバッファに格納
    for (int i = 0; i < 13; ++i) {
        buf(i)      = static_cast<float>(popcount(m[i] & mh)); // My Hand
        buf(i + 13) = static_cast<float>(popcount(m[i] & b));  // Board
        buf(i + 26) = static_cast<float>(popcount(m[i] & mc)); // My Collect
        buf(i + 39) = static_cast<float>(popcount(m[i] & oc)); // Op Collect
        buf(i + 52) = static_cast<float>(popcount(m[i] & unseen));  // Unseen
    }
    return result;
}

std::vector<int> card_to_multi_hot(const std::vector<std::vector<int>>& card_list) {
    std::vector<int> multi_hot(48, 0);
    for (const auto& card : card_list) {
        if (card.size() == 2) {
            int index = card_index(card[0], card[1]);
            if (index >= 0 && index < 48) multi_hot[index] = 1;
        }
    }
    return multi_hot;
}

int get_yaku_point(const std::vector<std::vector<int>>& pile_cards, int koikoi_num) {
    return get_yaku_point_by_bitboard(cards_to_bitboard(pile_cards), koikoi_num);
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

std::vector<std::tuple<int, std::string, int>> evaluate_yaku(
    const std::vector<std::vector<int>>& pile_cards, int koikoi_num) {
    return evaluate_yaku_by_bitboard(cards_to_bitboard(pile_cards), koikoi_num);
}

std::vector<std::pair<int, int>> evaluate_yaku_id_by_bitboard(uint64_t pile, int koikoi_num) {
    std::vector<std::pair<int, int>> yaku_list;
    return yaku_list; 
}

std::vector<std::vector<int>> get_yaku_status_features_by_bitboard(
    uint64_t mh, uint64_t b, uint64_t mc, uint64_t oc, uint64_t u) { return {}; }

std::vector<std::vector<int>> get_yaku_status_features(
    const std::vector<std::vector<int>>& my_hand,
    const std::vector<std::vector<int>>& board,
    const std::vector<std::vector<int>>& my_collect,
    const std::vector<std::vector<int>>& op_collect,
    const std::vector<std::vector<int>>& unseen) { return {}; }

void build_feature_inplace(
    py::array_t<float>& feat_buf, 
    bool is_koikoi,
    float point_diff_half,
    float yp_t, float yp_i,
    int round_num, int turn_16, int dealer,
    int koikoi_num_turn, int koikoi_num_idle,
    uint8_t koikoi_turn_flags,
    uint8_t koikoi_idle_flags,
    uint64_t hand,
    uint64_t init_board,
    uint64_t unseen,
    uint64_t my_pile,
    uint64_t field,
    uint64_t op_pile,
    uint64_t show,
    uint64_t pairing,
    py::array_t<float>& card_log_buf, 
    py::array_t<py::ssize_t>& order) 
{
    auto buf = feat_buf.mutable_unchecked<2>();
    auto log_buf = card_log_buf.unchecked<3>();
    auto ord = order.unchecked<1>();

    int offset = is_koikoi ? 2 : 0;
    auto sgn = [](float x) { return (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f); };

    float t55[55] = {0.0f};

    float ax0 = std::abs(point_diff_half); float sgn0 = sgn(point_diff_half);
    t55[0] = std::sqrt(ax0) * sgn0; t55[1] = ax0 * sgn0 * 0.5f; t55[2] = std::pow(ax0, 1.5f) * sgn0 * 0.1f;

    float ax1 = std::abs(yp_t); float sgn1 = sgn(yp_t);
    t55[3] = std::sqrt(ax1) * sgn1; t55[4] = ax1 * sgn1 * 0.5f; t55[5] = std::pow(ax1, 1.5f) * sgn1 * 0.1f;

    float ax2 = std::abs(yp_i); float sgn2 = sgn(yp_i);
    t55[6] = std::sqrt(ax2) * sgn2; t55[7] = ax2 * sgn2 * 0.5f; t55[8] = std::pow(ax2, 1.5f) * sgn2 * 0.1f;

    t55[9 + round_num - 1] = 1.0f;
    t55[17 + turn_16 - 1] = 1.0f;
    t55[33 + dealer - 1] = 1.0f;

    float k1 = static_cast<float>(koikoi_num_turn); float k2 = static_cast<float>(koikoi_num_idle);
    t55[35] = std::abs(k1) * sgn(k1); t55[36] = (k1 * k1) * sgn(k1);
    t55[37] = std::abs(k2) * sgn(k2); t55[38] = (k2 * k2) * sgn(k2);

    for (int i = 0; i < 8; ++i) {
        t55[39 + i] = static_cast<float>((koikoi_turn_flags >> i) & 1);
        t55[47 + i] = static_cast<float>((koikoi_idle_flags >> i) & 1);
    }

    float yaku_feat[65] = {0.0f};
    const uint64_t m[13] = {
        MASKS.crane, MASKS.curtain, MASKS.moon, MASKS.rainman, MASKS.phoenix, MASKS.sake,
        MASKS.boar_deer_butterfly, MASKS.seed, MASKS.red_ribbon, MASKS.blue_ribbon,
        MASKS.red_blue_ribbon, MASKS.ribbon, MASKS.dross
    };

    for (int i = 0; i < 13; ++i) {
        yaku_feat[i]      = static_cast<float>(popcount(m[i] & hand));
        yaku_feat[i + 13] = static_cast<float>(popcount(m[i] & field));
        yaku_feat[i + 26] = static_cast<float>(popcount(m[i] & my_pile));
        yaku_feat[i + 39] = static_cast<float>(popcount(m[i] & op_pile));
        yaku_feat[i + 52] = static_cast<float>(popcount(m[i] & unseen));
    }

    for (int r = 17; r < 72; ++r) {
        float val = t55[r - 17];
        for (int c = 0; c < 48; ++c) buf(r, c + offset) = val;
    }
    for (int r = 72; r < 137; ++r) {
        float val = yaku_feat[r - 72];
        for (int c = 0; c < 48; ++c) buf(r, c + offset) = val;
    }

    auto set_bits = [&](int row, uint64_t bb) {
        while (bb) {
#if defined(__GNUC__) || defined(__clang__)
            int c = __builtin_ctzll(bb);
#elif defined(_MSC_VER)
            unsigned long c;
            _BitScanForward64(&c, bb);
#else
            int c = 0; uint64_t temp = bb;
            while((temp & 1) == 0) { temp >>= 1; c++; }
#endif
            buf(row, c + offset) = 1.0f;
            bb &= bb - 1; 
        }
    };

    for (int r = 162; r < 172; ++r) {
        for (int c = 0; c < 48; ++c) buf(r, c + offset) = 0.0f;
    }
    set_bits(162, hand); set_bits(165, hand);
    set_bits(163, init_board);
    set_bits(164, unseen); set_bits(169, unseen);
    set_bits(166, my_pile);
    set_bits(167, field);
    set_bits(168, op_pile);
    set_bits(170, show);
    set_bits(171, pairing);

    int out_r = 172;
    for (int i = 0; i < 16; ++i) {
        py::ssize_t t = ord(i); 
        for (int sub_r = 0; sub_r < 8; ++sub_r) {
            for (int c = 0; c < 48; ++c) buf(out_r, c + offset) = log_buf(t, sub_r, c);
            out_r++;
        }
    }

    if (is_koikoi) {
        for(int r = 0; r < 300; ++r) { buf(r, 0) = 0.0f; buf(r, 1) = 0.0f; }
        for(int r = 0; r < 137; ++r) { buf(r, 0) = buf(r, 2); buf(r, 1) = buf(r, 3); }
        buf(0, 0) = 1.0f;
        buf(1, 1) = 1.0f;
    }
}

inline uint64_t list_to_bb(const py::list& l) {
    uint64_t bb = 0;
    for (size_t i = 0; i < l.size(); ++i) {
        int val = l[i].cast<int>();
        if (val >= 0 && val < 48) bb |= (1ULL << val);
    }
    return bb;
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

    float df = static_cast<float>(snap.point_turn - snap.point_idle) * 0.5f;
    float ax0 = std::abs(df); float sgn0 = sgn(df);
    t55[0] = std::sqrt(ax0) * sgn0; t55[1] = ax0 * sgn0 * 0.5f; t55[2] = std::pow(ax0, 1.5f) * sgn0 * 0.1f;

    float yp_t = static_cast<float>(get_yaku_point_by_bitboard(snap.bb_my_pile, snap.koikoi_num_turn));
    float ax1 = std::abs(yp_t); float sgn1 = sgn(yp_t);
    t55[3] = std::sqrt(ax1) * sgn1; t55[4] = ax1 * sgn1 * 0.5f; t55[5] = std::pow(ax1, 1.5f) * sgn1 * 0.1f;

    float yp_i = static_cast<float>(get_yaku_point_by_bitboard(snap.bb_op_pile, snap.koikoi_num_idle));
    float ax2 = std::abs(yp_i); float sgn2 = sgn(yp_i);
    t55[6] = std::sqrt(ax2) * sgn2; t55[7] = ax2 * sgn2 * 0.5f; t55[8] = std::pow(ax2, 1.5f) * sgn2 * 0.1f;

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

    auto set_v = [&](int r, int c, float v) { dest[r * cols + c + off] = v; };

    for (int r = 17; r < 72; ++r) {
        float val = t55[r - 17];
        if (val != 0.0f) { for (int c = 0; c < 48; ++c) set_v(r, c, val); }
    }
    for (int r = 72; r < 137; ++r) {
        float val = yaku_feat[r - 72];
        if (val != 0.0f) { for (int c = 0; c < 48; ++c) set_v(r, c, val); }
    }
    for (int r = 0; r < 25; ++r) {
        for (int c = 0; c < 48; ++c) {
            float val = cache_ptr[r * 48 + c];
            if (val != 0.0f) set_v(137 + r, c, val);
        }
    }

    auto set_bits = [&](int row, uint64_t bb) {
        while (bb) {
#if defined(__GNUC__) || defined(__clang__)
            int c = __builtin_ctzll(bb);
#elif defined(_MSC_VER)
            unsigned long c; _BitScanForward64(&c, bb);
#else
            int c = 0; uint64_t temp = bb; while((temp & 1) == 0) { temp >>= 1; c++; }
#endif
            set_v(row, c, 1.0f); bb &= bb - 1; 
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
        for (int sr = 0; sr < 8; ++sr) {
            bool ignore = false;
            // 同一 turn_16 内の未確定 sr マスク処理
            if (apply_mask) {
                if (t > snap.turn_16) ignore = true;
                else if (t == snap.turn_16) {
                    if (snap.state_type == 0) ignore = true;
                    else if (snap.state_type == 1) {
                        if (!snap.is_draw_pick && sr >= 2) ignore = true;
                        if (snap.is_draw_pick && sr >= 6) ignore = true;
                    }
                }
            }
            if (!ignore) {
                for (int c = 0; c < 48; ++c) {
                    float val = log_buf_ptr[t * 8 * 48 + sr * 48 + c];
                    if (val != 0.0f) set_v(out_r, c, val);
                }
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

    // Action Alignment の統合 (0-copy シフト)
    if (action_to_align > 0 && action_to_align < cols) {
        std::vector<float> temp(cols);
        for (int r = 0; r < 300; ++r) {
            float* row_ptr = dest + r * cols;
            std::memcpy(temp.data(), row_ptr, cols * sizeof(float));
            row_ptr[0] = temp[action_to_align];
            std::memcpy(row_ptr + 1, temp.data(), action_to_align * sizeof(float));
            // action+1 以降は相対位置が変わらないためコピー不要
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

py::array_t<float> adjust_feature(py::array_t<float>& feature, int index) {
    auto buf = feature.unchecked<2>();
    py::ssize_t rows = buf.shape(0);
    py::ssize_t cols = buf.shape(1);

    auto result = py::array_t<float>({rows, cols});
    float* dest = result.mutable_data();
    const float* src = feature.data();

    if (index < 0 || index >= cols) {
        std::memcpy(dest, src, rows * cols * sizeof(float));
        return result;
    }

    for (py::ssize_t r = 0; r < rows; ++r) {
        dest[r * cols + 0] = src[r * cols + index];
        if (index > 0) {
            std::memcpy(&dest[r * cols + 1], &src[r * cols + 0], index * sizeof(float));
        }
        if (index < cols - 1) {
            std::memcpy(&dest[r * cols + 1 + index], &src[r * cols + index + 1], (cols - 1 - index) * sizeof(float));
        }
    }
    return result;
}

py::array_t<float> adjust_features_batched(py::array_t<float>& states, py::array_t<int>& actions) {
    auto buf_s = states.unchecked<3>();
    auto buf_a = actions.unchecked<1>();
    
    py::ssize_t B = buf_s.shape(0);
    py::ssize_t rows = buf_s.shape(1);
    py::ssize_t cols = buf_s.shape(2);

    auto result = py::array_t<float>({B, rows, cols});
    float* dest = result.mutable_data();
    const float* src = states.data();

    for (py::ssize_t b = 0; b < B; ++b) {
        int index = buf_a(b);
        py::ssize_t b_offset = b * rows * cols;
        
        if (index < 0 || index >= cols) {
            std::memcpy(&dest[b_offset], &src[b_offset], rows * cols * sizeof(float));
            continue;
        }

        for (py::ssize_t r = 0; r < rows; ++r) {
            py::ssize_t r_offset = b_offset + r * cols;
            dest[r_offset + 0] = src[r_offset + index];
            if (index > 0) {
                std::memcpy(&dest[r_offset + 1], &src[r_offset + 0], index * sizeof(float));
            }
            if (index < cols - 1) {
                std::memcpy(&dest[r_offset + 1 + index], &src[r_offset + index + 1], (cols - 1 - index) * sizeof(float));
            }
        }
    }
    return result;
}

py::array_t<float> build_feature_packed(
    py::array_t<uint64_t>& state_arr,
    py::array_t<float>& card_log_buf, 
    py::array_t<py::ssize_t>& order,
    py::array_t<float>& cache_buf) 
{
    auto st = state_arr.unchecked<1>();
    
    bool is_koikoi = (st(0) != 0);
    int point_turn = static_cast<int>(st(1));
    int point_idle = static_cast<int>(st(2));
    int round_num = static_cast<int>(st(3));
    int turn_16 = static_cast<int>(st(4));
    int dealer = static_cast<int>(st(5));
    int koikoi_num_turn = static_cast<int>(st(6));
    int koikoi_num_idle = static_cast<int>(st(7));
    uint8_t f_turn = static_cast<uint8_t>(st(8));
    uint8_t f_idle = static_cast<uint8_t>(st(9));
    
    uint64_t bb_hand = st(10);
    uint64_t bb_init_board = st(11);
    uint64_t bb_unseen = st(12);
    uint64_t bb_my_pile = st(13);
    uint64_t bb_field = st(14);
    uint64_t bb_op_pile = st(15);
    uint64_t bb_show = st(16);
    uint64_t bb_pairing = st(17);

    int cols = is_koikoi ? 50 : 48;
    int offset = is_koikoi ? 2 : 0;
    
    auto result = py::array_t<float>({300, cols});
    float* buf_ptr = result.mutable_data();
    std::memset(buf_ptr, 0, 300 * cols * sizeof(float));

    auto log_buf = card_log_buf.unchecked<3>();
    auto ord = order.unchecked<1>();
    auto cache = cache_buf.unchecked<2>();
    auto buf = [&](int r, int c) -> float& { return buf_ptr[r * cols + c]; };

    float yp_t = static_cast<float>(get_yaku_point_by_bitboard(bb_my_pile, koikoi_num_turn));
    float yp_i = static_cast<float>(get_yaku_point_by_bitboard(bb_op_pile, koikoi_num_idle));
    float point_diff_half = static_cast<float>(point_turn - point_idle) * 0.5f;

    auto sgn = [](float x) { return (x > 0) ? 1.0f : ((x < 0) ? -1.0f : 0.0f); };

    float t55[55] = {0.0f};
    float ax0 = std::abs(point_diff_half); float sgn0 = sgn(point_diff_half);
    t55[0] = std::sqrt(ax0) * sgn0; t55[1] = ax0 * sgn0 * 0.5f; t55[2] = std::pow(ax0, 1.5f) * sgn0 * 0.1f;

    float ax1 = std::abs(yp_t); float sgn1 = sgn(yp_t);
    t55[3] = std::sqrt(ax1) * sgn1; t55[4] = ax1 * sgn1 * 0.5f; t55[5] = std::pow(ax1, 1.5f) * sgn1 * 0.1f;

    float ax2 = std::abs(yp_i); float sgn2 = sgn(yp_i);
    t55[6] = std::sqrt(ax2) * sgn2; t55[7] = ax2 * sgn2 * 0.5f; t55[8] = std::pow(ax2, 1.5f) * sgn2 * 0.1f;

    if(round_num >= 1 && round_num <= 8) t55[9 + round_num - 1] = 1.0f;
    if(turn_16 >= 1 && turn_16 <= 16) t55[17 + turn_16 - 1] = 1.0f;
    if(dealer >= 1 && dealer <= 2) t55[33 + dealer - 1] = 1.0f;

    float k1 = static_cast<float>(koikoi_num_turn); float k2 = static_cast<float>(koikoi_num_idle);
    t55[35] = std::abs(k1) * sgn(k1); t55[36] = (k1 * k1) * sgn(k1);
    t55[37] = std::abs(k2) * sgn(k2); t55[38] = (k2 * k2) * sgn(k2);

    for (int i = 0; i < 8; ++i) {
        t55[39 + i] = static_cast<float>((f_turn >> i) & 1);
        t55[47 + i] = static_cast<float>((f_idle >> i) & 1);
    }

    float yaku_feat[65] = {0.0f};
    const uint64_t m[13] = {
        MASKS.crane, MASKS.curtain, MASKS.moon, MASKS.rainman, MASKS.phoenix, MASKS.sake,
        MASKS.boar_deer_butterfly, MASKS.seed, MASKS.red_ribbon, MASKS.blue_ribbon,
        MASKS.red_blue_ribbon, MASKS.ribbon, MASKS.dross
    };

    for (int i = 0; i < 13; ++i) {
        yaku_feat[i]      = static_cast<float>(popcount(m[i] & bb_hand));
        yaku_feat[i + 13] = static_cast<float>(popcount(m[i] & bb_field));
        yaku_feat[i + 26] = static_cast<float>(popcount(m[i] & bb_my_pile));
        yaku_feat[i + 39] = static_cast<float>(popcount(m[i] & bb_op_pile));
        yaku_feat[i + 52] = static_cast<float>(popcount(m[i] & bb_unseen));
    }

    for (int r = 17; r < 72; ++r) {
        float val = t55[r - 17];
        if (val != 0.0f) { for (int c = 0; c < 48; ++c) buf(r, c + offset) = val; }
    }
    for (int r = 72; r < 137; ++r) {
        float val = yaku_feat[r - 72];
        if (val != 0.0f) { for (int c = 0; c < 48; ++c) buf(r, c + offset) = val; }
    }
    for (int r = 0; r < 25; ++r) {
        for (int c = 0; c < 48; ++c) buf(137 + r, c + offset) = cache(r, c);
    }

    auto set_bits = [&](int row, uint64_t bb) {
        while (bb) {
#if defined(__GNUC__) || defined(__clang__)
            int c = __builtin_ctzll(bb);
#elif defined(_MSC_VER)
            unsigned long c; _BitScanForward64(&c, bb);
#else
            int c = 0; uint64_t temp = bb; while((temp & 1) == 0) { temp >>= 1; c++; }
#endif
            buf(row, c + offset) = 1.0f;
            bb &= bb - 1; 
        }
    };

    set_bits(162, bb_hand); set_bits(165, bb_hand);
    set_bits(163, bb_init_board);
    set_bits(164, bb_unseen); set_bits(169, bb_unseen);
    set_bits(166, bb_my_pile);
    set_bits(167, bb_field);
    set_bits(168, bb_op_pile);
    set_bits(170, bb_show);
    set_bits(171, bb_pairing);

    int out_r = 172;
    for (int i = 0; i < 16; ++i) {
        py::ssize_t t = ord(i); 
        for (int sub_r = 0; sub_r < 8; ++sub_r) {
            for (int c = 0; c < 48; ++c) {
                float val = log_buf(t, sub_r, c);
                if (val != 0.0f) buf(out_r, c + offset) = val;
            }
            out_r++;
        }
    }

    if (is_koikoi) {
        for(int r = 0; r < 137; ++r) { 
            buf(r, 0) = buf(r, 2); buf(r, 1) = buf(r, 3); 
        }
        buf(0, 0) = 1.0f; buf(1, 1) = 1.0f;
    }
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
    int peak_size = 0; // 実行中に観測された最大サイズを記録
    std::vector<float> states, rewards;
    std::vector<int> actions;

    KoiKoiTraceBuffer(int cap, int r, int c) : capacity(cap), rows(r), cols(c), current_size(0) {
        states.resize(capacity * rows * cols);
        actions.resize(capacity);
        rewards.resize(capacity);
    }

    void clear() { current_size = 0; }

    // Python API向けのフォールバック（動作変更なし）
    void push(py::array_t<float> state, int action, float reward) {
        if (current_size >= capacity) return;
        auto buf = state.unchecked<2>();
        float* dest = &states[current_size * rows * cols];
        std::memcpy(dest, state.data(), rows * cols * sizeof(float));
        actions[current_size] = action; rewards[current_size] = reward; current_size++;
        if (current_size > peak_size) {
            peak_size = current_size;
        }
    }

    // 遅延再構築用メソッド
    void push_reconstructed(const Snapshot& snap, int action, float reward,
                            const float* log_buf_ptr, const int* order, const float* cache_ptr) {
        if (current_size >= capacity) return;
        float* dest = &states[current_size * rows * cols];
        actions[current_size] = action;
        rewards[current_size] = reward;

        write_feature_core(dest, cols, snap.state_type == 2 ? 2 : 0, snap,
                           log_buf_ptr, order, cache_ptr,
                           true, action); // マスク有効化、アライメント実行
        current_size++;
        if (current_size > peak_size) {
            peak_size = current_size;
        }
    }

    std::pair<torch::Tensor, torch::Tensor> get_tensors() {
        if (current_size == 0) return {torch::empty({0}), torch::empty({0})};
        auto res_states = torch::empty({current_size, rows, cols}, torch::kFloat32);
        auto res_rewards = torch::empty({current_size}, torch::kFloat32);
        // アライメント済みのため、シフトロジックを完全削除して単純コピー
        std::memcpy(res_states.data_ptr<float>(), states.data(), current_size * rows * cols * sizeof(float));
        std::memcpy(res_rewards.data_ptr<float>(), rewards.data(), current_size * sizeof(float));
        return {res_states, res_rewards};
    }

    py::dict finalize() {
        auto res_states = py::array_t<float>({current_size, rows, cols});
        auto res_actions = py::array_t<int>(current_size);
        auto res_rewards = py::array_t<float>(current_size);
        // シフトロジックを完全削除して単純コピー
        std::memcpy(res_states.mutable_data(), states.data(), current_size * rows * cols * sizeof(float));
        std::memcpy(res_actions.mutable_data(), actions.data(), current_size * sizeof(int));
        std::memcpy(res_rewards.mutable_data(), rewards.data(), current_size * sizeof(float));
        py::dict result;
        result["states"] = res_states; result["actions"] = res_actions; result["rewards"] = res_rewards;
        return result;
    }

    size_t print_buffer_memory_report(const std::string& prefix) const {
        size_t states_bytes  = states.capacity() * sizeof(float);
        size_t rewards_bytes = rewards.capacity() * sizeof(float);
        size_t actions_bytes = actions.capacity() * sizeof(int);

        double states_mb  = static_cast<double>(states_bytes)  / (1024.0 * 1024.0);
        double rewards_mb = static_cast<double>(rewards_bytes) / (1024.0 * 1024.0);
        double actions_mb = static_cast<double>(actions_bytes) / (1024.0 * 1024.0);

        std::cout << "  " << prefix << ".states  : " << std::fixed << std::setprecision(2) << states_mb << " MB\n"
                  << "  " << prefix << ".rewards : " << std::fixed << std::setprecision(2) << rewards_mb << " MB\n"
                  << "  " << prefix << ".actions : " << std::fixed << std::setprecision(2) << actions_mb << " MB\n";

        return states_bytes + rewards_bytes + actions_bytes;
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
    std::vector<int> hand[3], field_slot, stock, pile[3], show, collect;
    std::mt19937 rng;

    float card_log_buf[17][8][48];
    std::vector<int> turn_pairCard[17], turn_pairCard2[17];

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
        for(int c : hand[1]) bb_h1 |= (1ULL << c);
        for(int c : hand[2]) bb_h2 |= (1ULL << c);
        for(int c : field_slot) if (c != -1) bb_f |= (1ULL << c);
        for(int c : stock) bb_s |= (1ULL << c);

        state_manager.init_board(bb_h1, bb_h2, bb_f, bb_s);
        state = GameState::DISCARD; wait_action = true;
    }

    int turn_player() const { return ((turn_16 + dealer) % 2 == 0) ? 1 : 2; }
    int idle_player() const { return 3 - turn_player(); }
    int turn_8() const { return (turn_16 + 1) / 2; }
    int koikoi_num(int p) const { int s=0; for(int i=0; i<8; ++i) s+=koikoi[p][i]; return s; }

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

    void set_log_buf(int t16, int idx, const std::vector<int>& cards) {
        if (t16 < 1 || t16 > 16) return;
        for(int c=0; c<48; ++c) card_log_buf[t16][idx][c] = 0.0f;
        for(int c : cards) if(c >= 0 && c < 48) card_log_buf[t16][idx][c] = 1.0f;
    }

    std::vector<int> get_pairing_cards() const {
        std::vector<int> pairs;
        if (show.empty()) return pairs;
        int target = show[0] / 4;
        for (int c : field_slot) if (c != -1 && c / 4 == target) pairs.push_back(c);
        return pairs;
    }

    std::vector<int> field_collect() const {
        std::vector<int> fc = collect;
        auto it = std::find(fc.begin(), fc.end(), show[0]);
        if (it != fc.end()) fc.erase(it);
        return fc;
    }

    std::vector<int> get_uncollect(const std::vector<int>& pairing, const std::vector<int>& fc) const {
        std::vector<int> un;
        for(int c : pairing) if (std::find(fc.begin(), fc.end(), c) == fc.end()) un.push_back(c);
        return un;
    }

    void _collect_card(int card) {
        std::vector<int> pairing = get_pairing_cards();
        if (pairing.empty()) {
            collect.clear();
            auto it = std::find(field_slot.begin(), field_slot.end(), -1);
            if (it != field_slot.end()) *it = show[0];
        } else if (pairing.size() == 1 || pairing.size() == 3) {
            collect = show; collect.insert(collect.end(), pairing.begin(), pairing.end());
            for (int p_card : pairing) {
                auto it = std::find(field_slot.begin(), field_slot.end(), p_card);
                if (it != field_slot.end()) *it = -1;
            }
            pile[turn_player()].insert(pile[turn_player()].end(), collect.begin(), collect.end());
        } else {
            collect = show; collect.push_back(card);
            auto it = std::find(field_slot.begin(), field_slot.end(), card);
            if (it != field_slot.end()) *it = -1;
            pile[turn_player()].insert(pile[turn_player()].end(), collect.begin(), collect.end());
        }
    }

    void step(int action) {
        int p = turn_player();
        if (state == GameState::DISCARD) {
            turn_point = state_manager.get_yaku_point(p, koikoi_num(p));
            auto it = std::find(hand[p].begin(), hand[p].end(), action);
            if (it != hand[p].end()) hand[p].erase(it);
            show = {action};

            std::vector<int> pairing = get_pairing_cards();
            uint64_t pair_bb = 0; for(int c : pairing) pair_bb |= (1ULL<<c);
            state_manager.discard(p, action, pair_bb);

            set_log_buf(turn_16, pairing.empty() ? 1 : 0, show);
            turn_pairCard[turn_16] = pairing;
            state = GameState::DISCARD_PICK; wait_action = (pairing.size() == 2);
        }
        else if (state == GameState::DISCARD_PICK) {
            _collect_card(action);
            uint64_t coll_bb = 0; for(int c : collect) coll_bb |= (1ULL<<c);
            uint64_t field_bb = 0; for(int c : field_slot) if(c != -1) field_bb |= (1ULL<<c);
            state_manager.discard_pick(p, coll_bb, field_bb);

            if (!collect.empty()) {
                auto fc = field_collect();
                set_log_buf(turn_16, 2, fc);
                set_log_buf(turn_16, 3, get_uncollect(turn_pairCard[turn_16], fc));
            }
            state = GameState::DRAW; wait_action = false;
        }
        else if (state == GameState::DRAW) {
            int drawn = stock.back(); stock.pop_back(); show = {drawn};
            std::vector<int> pairing = get_pairing_cards();
            uint64_t pair_bb = 0; for(int c : pairing) pair_bb |= (1ULL<<c);
            state_manager.draw(drawn, pair_bb);

            set_log_buf(turn_16, pairing.empty() ? 5 : 4, show);
            turn_pairCard2[turn_16] = pairing;
            state = GameState::DRAW_PICK; wait_action = (pairing.size() == 2);
        }
        else if (state == GameState::DRAW_PICK) {
            _collect_card(action);
            uint64_t coll_bb = 0; for(int c : collect) coll_bb |= (1ULL<<c);
            uint64_t field_bb = 0; for(int c : field_slot) if(c != -1) field_bb |= (1ULL<<c);
            state_manager.draw_pick(p, coll_bb, field_bb);

            if (!collect.empty()) {
                auto fc = field_collect();
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
        std::vector<int> cards(48);
        while (true) {
            for (int i = 0; i < 48; ++i) cards[i] = i;
            std::shuffle(cards.begin(), cards.end(), rng);
            hand[1] = std::vector<int>(cards.begin(), cards.begin()+8); std::sort(hand[1].begin(), hand[1].end());
            hand[2] = std::vector<int>(cards.begin()+8, cards.begin()+16); std::sort(hand[2].begin(), hand[2].end());
            std::vector<int> init_f(cards.begin()+16, cards.begin()+24); std::sort(init_f.begin(), init_f.end());
            field_slot = init_f; for (int i=0; i<8; ++i) field_slot.push_back(-1);
            stock = std::vector<int>(cards.begin()+24, cards.end());

            bool flag = true;
            for (int s = 0; s < 12; ++s) {
                int c1=0, c2=0, cf=0;
                for(int c : hand[1]) if(c/4 == s) c1++;
                for(int c : hand[2]) if(c/4 == s) c2++;
                for(int c : init_f) if(c/4 == s) cf++;
                if (c1==4 || c2==4 || cf==4) { flag = false; break; }
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
    std::vector<std::vector<TraceSlot>> traces[3];

    KoiKoiTraceBuffer buf_discard, buf_pick, buf_koikoi;
    std::vector<float> win_prob_mat;
    float cache_buf[25][48];
    int order[17][16];

    int finished_games = 0;
    torch::Device device;

    torch::Tensor feat_tensor[3];
    torch::Tensor mask_tensor[3];

    // 【計測用変数】
    double t_trace_alloc = 0;
    double t_trace_memcpy = 0;
    double t_trace_create = 0;
    double t_trace_push = 0;

    double t_ro_discount = 0;
    double t_ro_push_call = 0;

    BatchSimulator(int n_envs, int target, int cap_d, int cap_p, int cap_k, float disc, py::array_t<float> wp_mat, std::string dev_str)
        : num_envs(n_envs), target_games(target), discount(disc),
          buf_discard(cap_d, 300, 48), buf_pick(cap_p, 300, 48), buf_koikoi(cap_k, 300, 50),
          device(dev_str)
    {
        envs.resize(n_envs);
        for(int p=1; p<=2; ++p) traces[p].resize(n_envs);

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

        t_trace_alloc = 0; t_trace_memcpy = 0; t_trace_create = 0; t_trace_push = 0;
        t_ro_discount = 0; t_ro_push_call = 0;
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
        float df = static_cast<float>(env.point[pt] - env.point[pi]) * 0.5f;
        t55[0] = std::sqrt(std::abs(df))*sgn(df); t55[1] = std::abs(df)*sgn(df)*0.5f; t55[2] = std::pow(std::abs(df),1.5f)*sgn(df)*0.1f;
        float yp_t = static_cast<float>(env.state_manager.get_yaku_point(pt, env.koikoi_num(pt)));
        t55[3] = std::sqrt(yp_t)*sgn(yp_t); t55[4] = yp_t*sgn(yp_t)*0.5f; t55[5] = std::pow(yp_t,1.5f)*sgn(yp_t)*0.1f;
        float yp_i = static_cast<float>(env.state_manager.get_yaku_point(pi, env.koikoi_num(pi)));
        t55[6] = std::sqrt(yp_i)*sgn(yp_i); t55[7] = yp_i*sgn(yp_i)*0.5f; t55[8] = std::pow(yp_i,1.5f)*sgn(yp_i)*0.1f;
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
        for(int r=17; r<72; ++r) if(t55[r-17] != 0.0f) for(int c=0; c<48; ++c) set_v(r, c, t55[r-17]);
        for(int r=72; r<137; ++r) if(y_f[r-72] != 0.0f) for(int c=0; c<48; ++c) set_v(r, c, y_f[r-72]);
        for(int r=0; r<25; ++r) for(int c=0; c<48; ++c) set_v(137+r, c, cache_buf[r][c]);

        auto set_b = [&](int row, uint64_t bb) {
            while(bb) {
#if defined(__GNUC__) || defined(__clang__)
                int c = __builtin_ctzll(bb);
#elif defined(_MSC_VER)
                unsigned long c; _BitScanForward64(&c, bb);
#else
                int c=0; uint64_t t=bb; while((t&1)==0){ t>>=1; c++; }
#endif
                set_v(row, c, 1.0f); bb &= bb-1;
            }
        };

        set_b(162, bb_h); set_b(165, bb_h); set_b(163, env.state_manager.bb_initBoard);
        set_b(164, bb_u); set_b(169, bb_u); set_b(166, bb_m); set_b(167, bb_f);
        set_b(168, bb_o); set_b(170, env.state_manager.bb_show); set_b(171, env.state_manager.bb_pairing);

        int out_r = 172, t16 = std::max(1, std::min(16, env.turn_16));
        for(int j=0; j<16; ++j) {
            int t = order[t16][j];
            for(int sr=0; sr<8; ++sr) {
                for(int c=0; c<48; ++c) { float v = env.card_log_buf[t][sr][c]; if(v != 0.0f) set_v(out_r, c, v); }
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
        
        float next_potential = final_potential;
        for (size_t rs = 0; rs < trace.size(); ++rs) {
            auto& slot = trace[trace.size() - 1 - rs];
            float current_potential = slot.snap.potential;
            float pbrs_reward = (discount * next_potential - current_potential) * 10.0f;
            
            int t16_limit = std::max<int>(1, std::min<int>(16, slot.snap.turn_16));
            const float* log_ptr = &env.card_log_buf[0][0][0];
            const float* cache_ptr = &cache_buf[0][0];

            if (slot.state_type == 0) buf_discard.push_reconstructed(slot.snap, slot.action, pbrs_reward, log_ptr, order[t16_limit], cache_ptr);
            else if (slot.state_type == 1) buf_pick.push_reconstructed(slot.snap, slot.action, pbrs_reward, log_ptr, order[t16_limit], cache_ptr);
            else buf_koikoi.push_reconstructed(slot.snap, slot.action, pbrs_reward, log_ptr, order[t16_limit], cache_ptr);
            
            next_potential = current_potential;
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
            for (int i = 0; i < num_envs; ++i) {
                auto& env = envs[i];
                while (!env.needs_action() && env.state != GameState::GAME_OVER) {
                    if (env.state == GameState::ROUND_OVER) {
                        process_round_over(i);
                        if (env.point[1] <= 0 || env.point[2] <= 0 || env.round_num == 8) {
                            finished_games++;
                            if (finished_games < target_games) { env.reset_game(); env.reset_round(); }
                            else break;
                        } else { env.round_num++; env.reset_round(); }
                    } else { env.step(-1); }
                }
            }

            if (finished_games >= target_games) break;

            // P1とP2の独立した推論ループ
            for (int p_id = 1; p_id <= 2; ++p_id) {
                for(int i=0; i<3; ++i) req_env_idx[i].clear();
                
                for (int i = 0; i < num_envs; ++i) {
                    auto& env = envs[i];
                    if (env.needs_action() && env.turn_player() == p_id) {
                        int type = (env.state == GameState::DISCARD) ? 0 : (env.state == GameState::KOIKOI ? 2 : 1);
                        int batch_idx = req_env_idx[type].size();
                        req_env_idx[type].push_back(i);
                        int cols = (type == 2) ? 50 : 48;
                        int m_cols = (type == 2) ? 2 : 48;
                        build_feature_into(i, feat_ptrs[type] + batch_idx * 300 * cols, type);
                        build_mask_into(i, mask_ptrs[type] + batch_idx * m_cols, type);
                    }
                }

                torch::NoGradGuard no_grad; 
                for (int type = 0; type < 3; ++type) {
                    int n = static_cast<int>(req_env_idx[type].size());
                    if (n == 0) continue;

                    torch::Tensor f_gpu = feat_tensor[type].slice(0, 0, n).to(device, true).to(torch::kBFloat16);
                    torch::Tensor m_gpu = mask_tensor[type].slice(0, 0, n).to(device, true);

                    torch::Tensor output;
                    if (type == 0) output = g_models.discard.forward({f_gpu}).toTensor();
                    else if (type == 1) output = g_models.pick.forward({f_gpu}).toTensor();
                    else output = g_models.koikoi.forward({f_gpu}).toTensor();

                    output = output.masked_fill(~m_gpu, -1e9);
                    int64_t* act_ptr = output.argmax(1).cpu().data_ptr<int64_t>();

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

    size_t get_pinned_tensors_bytes() const {
        size_t total = 0;
        for (int i = 0; i < 3; ++i) {
            total += feat_tensor[i].nbytes();
            total += mask_tensor[i].nbytes();
        }
        return total;
    }

    size_t print_simulator_memory_report() const {
        size_t total_buf_bytes = 0;
        total_buf_bytes += buf_discard.print_buffer_memory_report("discard");
        total_buf_bytes += buf_pick.print_buffer_memory_report("pick");
        total_buf_bytes += buf_koikoi.print_buffer_memory_report("koikoi");
        return total_buf_bytes;
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
        states = states.to(device, /*non_blocking=*/true);
        rewards = rewards.to(device, /*non_blocking=*/true);
        
        int64_t num_samples = states.size(0);
        auto indices = torch::randperm(num_samples, torch::TensorOptions().device(device).dtype(torch::kLong));
        
        float total_loss = 0.0f;
        int num_batches = 0;

        py::gil_scoped_release release;

        for (int64_t i = 0; i < num_samples; i += batch_size) {
            int64_t end = std::min(i + batch_size, num_samples);
            auto batch_indices = indices.slice(0, i, end);
            
            auto state_batch = states.index_select(0, batch_indices);
            auto reward_batch = rewards.index_select(0, batch_indices);

            optimizer->zero_grad();

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(state_batch.to(torch::kBFloat16));
            
            torch::Tensor q_values = model.forward(inputs).toTensor().view({-1});
            torch::Tensor reward_flat = reward_batch.view({-1});
            
            auto loss = torch::nn::functional::smooth_l1_loss(
                q_values.to(torch::kFloat32), 
                reward_flat.to(torch::kFloat32), 
                torch::nn::functional::SmoothL1LossFuncOptions().beta(30.0)
            );

            loss.backward();
            optimizer->step();

            total_loss += loss.item<float>();
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
            py::array_t<float> s_arr = data["states"].cast<py::array_t<float>>();
            py::array_t<float> r_arr = data["rewards"].cast<py::array_t<float>>();
            
            if (s_arr.shape(0) > 0) {
                auto s_tensor = torch::from_blob(s_arr.mutable_data(), {s_arr.shape(0), s_arr.shape(1), s_arr.shape(2)}, torch::kFloat32).clone();
                auto r_tensor = torch::from_blob(r_arr.mutable_data(), {r_arr.shape(0)}, torch::kFloat32).clone();
                state_list.push_back(s_tensor);
                reward_list.push_back(r_tensor);
            }
        }
        
        if (state_list.empty()) return 0.0f;
        
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

    void print_memory_report() const {
        if (sims.empty()) return;

        std::cout << "\n[Memory Report]\n";
        
        // 1. 各バッファの内訳 (全スレッドで容量設定は共通なため、sims[0]を代表値として出力)
        size_t single_sim_buf_bytes = sims[0]->print_simulator_memory_report();

        // 2. Pinned Tensor の集計 (全スレッド分)
        size_t total_pinned_bytes = 0;
        for (const auto& sim : sims) {
            total_pinned_bytes += sim->get_pinned_tensors_bytes();
        }

        // 3. 全スレッドの合計
        size_t total_sims_buf_bytes = single_sim_buf_bytes * sims.size();
        size_t manager_total_bytes = total_sims_buf_bytes + total_pinned_bytes;

        double single_sim_mb  = static_cast<double>(single_sim_buf_bytes) / (1024.0 * 1024.0);
        double manager_total_mb = static_cast<double>(manager_total_bytes) / (1024.0 * 1024.0);
        double pinned_total_mb  = static_cast<double>(total_pinned_bytes)  / (1024.0 * 1024.0);

        std::cout << "  BatchSimulator Total    : " << std::fixed << std::setprecision(2) << single_sim_mb << " MB (per thread)\n"
                  << "  Pinned Tensor Total     : " << std::fixed << std::setprecision(2) << pinned_total_mb << " MB (all threads)\n"
                  << "  SimulationManager Total : " << std::fixed << std::setprecision(2) << manager_total_mb << " MB (Buffers + Pinned)\n"
                  << "-------------------------------------\n\n";
    }

    void print_buffer_utilization_report() const {
        if (sims.empty()) return;

        // 全スレッドの現在のサイズとピークサイズを正確に合算するための変数
        long long total_discard_cap = 0, total_discard_size = 0, total_discard_peak = 0;
        long long total_pick_cap = 0, total_pick_size = 0, total_pick_peak = 0;
        long long total_koikoi_cap = 0, total_koikoi_size = 0, total_koikoi_peak = 0;

        for (const auto& sim : sims) {
            total_discard_cap  += sim->buf_discard.capacity;
            total_discard_size += sim->buf_discard.current_size;
            total_discard_peak += sim->buf_discard.peak_size;

            total_pick_cap  += sim->buf_pick.capacity;
            total_pick_size += sim->buf_pick.current_size;
            total_pick_peak += sim->buf_pick.peak_size;

            total_koikoi_cap  += sim->buf_koikoi.capacity;
            total_koikoi_size += sim->buf_koikoi.current_size;
            total_koikoi_peak += sim->buf_koikoi.peak_size;
        }

        auto calc_pct = [](long long part, long long total) -> double {
            return total > 0 ? (static_cast<double>(part) / static_cast<double>(total)) * 100.0 : 0.0;
        };

        std::cout << "\n[Replay Buffer Utilization Report (All Threads Summed)]\n";
        std::cout << "  discard:\n"
                  << "    - Capacity     : " << total_discard_cap << "\n"
                  << "    - Current Size : " << total_discard_size << "\n"
                  << "    - Peak Size    : " << total_discard_peak << "\n"
                  << "    - Usage Rate   : " << std::fixed << std::setprecision(1) << calc_pct(total_discard_peak, total_discard_cap) << " %\n";
                  
        std::cout << "  pick:\n"
                  << "    - Capacity     : " << total_pick_cap << "\n"
                  << "    - Current Size : " << total_pick_size << "\n"
                  << "    - Peak Size    : " << total_pick_peak << "\n"
                  << "    - Usage Rate   : " << std::fixed << std::setprecision(1) << calc_pct(total_pick_peak, total_pick_cap) << " %\n";

        std::cout << "  koikoi:\n"
                  << "    - Capacity     : " << total_koikoi_cap << "\n"
                  << "    - Current Size : " << total_koikoi_size << "\n"
                  << "    - Peak Size    : " << total_koikoi_peak << "\n"
                  << "    - Usage Rate   : " << std::fixed << std::setprecision(1) << calc_pct(total_koikoi_peak, total_koikoi_cap) << " %\n";
        std::cout << "--------------------------------------------------------\n\n";
    }
};

static std::unique_ptr<SimulationManager> g_sim_manager = nullptr;

// ---------------------------------------------------------
// 統合パイプライン
// ---------------------------------------------------------

py::dict run_simulation_and_train(
    int num_threads, int n_envs_per_thread, int target_games_per_thread,
    int cap_d, int cap_p, int cap_k, float disc, py::array_t<float> wp_mat,
    std::string path_discard, std::string path_pick, std::string path_koikoi
    std::string dev_str,
    KoiKoiTrainer& trainer_discard, KoiKoiTrainer& trainer_pick, KoiKoiTrainer& trainer_koikoi,
    int batch_size, bool sync_models) 
{
    torch::Device device(dev_str);

    {
        std::lock_guard<std::mutex> lock(g_models.mtx);
        if (!g_models.loaded) {
            g_models.discard = torch::jit::load(path_discard, device);
            g_models.pick = torch::jit::load(path_pick, device);
            g_models.koikoi = torch::jit::load(path_koikoi, device);
            g_models.discard.eval(); g_models.pick.eval(); g_models.koikoi.eval();
            g_models.loaded = true;
        }
    }

    SimConfig config = {num_threads, n_envs_per_thread, target_games_per_thread, cap_d, cap_p, cap_k, disc, dev_str};
    
    if (g_sim_manager && g_sim_manager->current_config != config) g_sim_manager.reset(); 
    if (!g_sim_manager) g_sim_manager = std::make_unique<SimulationManager>(config, wp_mat);
    else g_sim_manager->reset_all();

    {
        py::gil_scoped_release release; 
        g_sim_manager->run_simulation_parallel();
    }

    std::vector<torch::Tensor> states_d, rewards_d;
    std::vector<torch::Tensor> states_p, rewards_p;
    std::vector<torch::Tensor> states_k, rewards_k;

    for (int i = 0; i < num_threads; ++i) {
        auto d = g_sim_manager->sims[i]->buf_discard.get_tensors();
        if (d.first.size(0) > 0) { states_d.push_back(d.first); rewards_d.push_back(d.second); }
        
        auto p = g_sim_manager->sims[i]->buf_pick.get_tensors();
        if (p.first.size(0) > 0) { states_p.push_back(p.first); rewards_p.push_back(p.second); }
        
        auto k = g_sim_manager->sims[i]->buf_koikoi.get_tensors();
        if (k.first.size(0) > 0) { states_k.push_back(k.first); rewards_k.push_back(k.second); }
    }

    int64_t sample_count_d = 0; for (const auto& t : states_d) sample_count_d += t.size(0);
    int64_t sample_count_p = 0; for (const auto& t : states_p) sample_count_p += t.size(0);
    int64_t sample_count_k = 0; for (const auto& t : states_k) sample_count_k += t.size(0);

    float loss_d = states_d.empty() ? 0.0f : trainer_discard.train_epoch(torch::cat(states_d, 0), torch::cat(rewards_d, 0), batch_size);
    float loss_p = states_p.empty() ? 0.0f : trainer_pick.train_epoch(torch::cat(states_p, 0), torch::cat(rewards_p, 0), batch_size);
    float loss_k = states_k.empty() ? 0.0f : trainer_koikoi.train_epoch(torch::cat(states_k, 0), torch::cat(rewards_k, 0), batch_size);

    if (sync_models) {
        trainer_discard.sync_and_save_action_model(path_discard);
        trainer_pick.sync_and_save_action_model(path_pick);
        trainer_koikoi.sync_and_save_action_model(path_koikoi);

        std::lock_guard<std::mutex> lock(g_models.mtx);
        g_models.discard = torch::jit::load(path_discard, device);
        g_models.pick = torch::jit::load(path_pick, device);
        g_models.koikoi = torch::jit::load(path_koikoi, device);
        g_models.discard.eval(); g_models.pick.eval(); g_models.koikoi.eval();
    }

    py::dict out;
    out["loss_discard"] = loss_d; out["loss_pick"] = loss_p; out["loss_koikoi"] = loss_k;
    out["samples_discard"] = sample_count_d; out["samples_pick"] = sample_count_p; out["samples_koikoi"] = sample_count_k;
    return out;
}

py::list run_parallel_simulations(
    int num_threads, int n_envs_per_thread, int target_games_per_thread,
    int cap_d, int cap_p, int cap_k, float disc, py::array_t<float> wp_mat,
    std::string path_discard, std::string path_pick, std::string path_koikoi, std::string dev_str) 
{
    torch::Device device(dev_str);

    {
        std::lock_guard<std::mutex> lock(g_models.mtx);
        g_models.discard = torch::jit::load(path_discard, device);
        g_models.pick = torch::jit::load(path_pick, device);
        g_models.koikoi = torch::jit::load(path_koikoi, device);
        g_models.discard.eval(); 
        g_models.pick.eval(); 
        g_models.koikoi.eval();
        g_models.loaded = true;
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

    // ==========================================
    // 実行前のメモリレポート
    // ==========================================
    std::cout << "\n>>> [Arena Simulation START] Current Component Memory Status:";
    g_sim_manager->print_memory_report();
    // ==========================================

    {
        py::gil_scoped_release release;
        g_sim_manager->run_simulation_parallel();

        // ==========================================
        std::cout << "\n>>> [Arena Simulation END] Final Component Memory Status:";
        g_sim_manager->print_memory_report(); // --- メモリ使用量測定用
        // g_sim_manager->print_buffer_utilization_report(); // --- メモリ使用量測定用
    }

    py::list results;
    for (int i = 0; i < num_threads; ++i) {
        results.append(g_sim_manager->sims[i]->finalize_buffers());
    }
    return results;
}

PYBIND11_MODULE(koikoicore, m) {
    m.doc() = "C++ Core Engine with Bitboards for KoiKoi AI";
    
    m.def("card_to_multi_hot", &card_to_multi_hot);
    m.def("evaluate_yaku", &evaluate_yaku);
    m.def("get_yaku_point", &get_yaku_point);
    m.def("get_yaku_status_features", &get_yaku_status_features);
    m.def("cards_to_bitboard", &cards_to_bitboard);
    m.def("get_yaku_point_by_bitboard", &get_yaku_point_by_bitboard);
    m.def("evaluate_yaku_by_bitboard", &evaluate_yaku_by_bitboard);
    m.def("get_yaku_status_features_by_bitboard", &get_yaku_status_features_by_bitboard);
    m.def("evaluate_yaku_id_by_bitboard", &evaluate_yaku_id_by_bitboard);
    m.def("cards_to_multi_hot_np", &cards_to_multi_hot_np);
    m.def("get_yaku_status_features_np", &get_yaku_status_features_np);
    m.def("build_feature_inplace", &build_feature_inplace);
    m.def("build_feature_fast", &build_feature_fast);
    m.def("adjust_feature", &adjust_feature);
    m.def("adjust_features_batched", &adjust_features_batched);
    m.def("build_feature_packed", &build_feature_packed);
    m.def("run_parallel_simulations", &run_parallel_simulations);
    m.def("run_simulation_and_train", &run_simulation_and_train);
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
        .def("push", &KoiKoiTraceBuffer::push)
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
        .def("sync_and_save_action_model", &KoiKoiTrainer::sync_and_save_action_model);
}