#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <string>
#include <tuple>
#include <array>

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

// (1) カードリストから一気にNumPy配列を生成して返す
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

// (2) Bitboardから一気に65次元の役ステータスNumPy配列を生成して返す
py::array_t<float> get_yaku_status_features_np(uint64_t mh, uint64_t b, uint64_t mc, uint64_t oc, uint64_t u) {
    auto result = py::array_t<float>(65);
    auto buf = result.mutable_unchecked<1>();

    const uint64_t m[13] = {
        MASKS.crane, MASKS.curtain, MASKS.moon, MASKS.rainman, MASKS.phoenix, MASKS.sake,
        MASKS.boar_deer_butterfly, MASKS.seed, MASKS.red_ribbon, MASKS.blue_ribbon,
        MASKS.red_blue_ribbon, MASKS.ribbon, MASKS.dross
    };

    for (int i = 0; i < 13; ++i) {
        buf(i)      = static_cast<float>(popcount(m[i] & mh));
        buf(i + 13) = static_cast<float>(popcount(m[i] & b));
        buf(i + 26) = static_cast<float>(popcount(m[i] & mc));
        buf(i + 39) = static_cast<float>(popcount(m[i] & oc));
        buf(i + 52) = static_cast<float>(popcount(m[i] & u));
    }
    return result;
}

// 既存互換用の役判定 (デバッグや記録表示用、最適化版)

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

PYBIND11_MODULE(koikoicore, m) {
    m.doc() = "C++ Core Engine with Bitboards for KoiKoi AI";
    
    // 既存ツール向けのバインディング
    m.def("card_to_multi_hot", &card_to_multi_hot);
    m.def("evaluate_yaku", &evaluate_yaku);
    m.def("get_yaku_point", &get_yaku_point);
    m.def("get_yaku_status_features", &get_yaku_status_features);

    // ビットボード向けのバインディング
    m.def("cards_to_bitboard", &cards_to_bitboard);
    m.def("get_yaku_point_by_bitboard", &get_yaku_point_by_bitboard);
    m.def("evaluate_yaku_by_bitboard", &evaluate_yaku_by_bitboard);
    m.def("get_yaku_status_features_by_bitboard", &get_yaku_status_features_by_bitboard);
    m.def("evaluate_yaku_id_by_bitboard", &evaluate_yaku_id_by_bitboard);

    // 今回追加したNumPyネイティブバインディング
    m.def("cards_to_multi_hot_np", &cards_to_multi_hot_np, "Convert card list to 1D numpy array directly");
    m.def("get_yaku_status_features_np", &get_yaku_status_features_np, "Get 65D yaku features as numpy array directly");
}