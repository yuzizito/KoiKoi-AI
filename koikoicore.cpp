#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>
#include <string>
#include <tuple>
#include <bitset>

namespace py = pybind11;

// (suit, rank) から 0〜47 の一意なインデックスを計算
inline int card_index(int suit, int rank) {
    return (suit - 1) * 4 + (rank - 1);
}

// 複数枚のカードリストを 64-bit 整数（Bitboard）に変換
uint64_t cards_to_bitboard(const std::vector<std::vector<int>>& cards) {
    uint64_t bb = 0;
    for (const auto& c : cards) {
        if (c.size() == 2) bb |= (1ULL << card_index(c[0], c[1]));
    }
    return bb;
}

// 役判定用のビットマスク（0/1のフラグ）を事前定義
struct KoiKoiMasks {
    uint64_t light = 0, seed = 0, ribbon = 0, dross = 0;
    uint64_t boar_deer_butterfly = 0, flower_sake = 0, moon_sake = 0;
    uint64_t red_ribbon = 0, blue_ribbon = 0, red_blue_ribbon = 0;
    uint64_t crane = 0, curtain = 0, moon = 0, rainman = 0, phoenix = 0, sake = 0;

    KoiKoiMasks() {
        auto add = [](uint64_t& mask, int s, int r) { mask |= (1ULL << card_index(s, r)); };
        add(crane, 1, 1); add(curtain, 3, 1); add(moon, 8, 1);
        add(rainman, 11, 1); add(phoenix, 12, 1); add(sake, 9, 1);

        light = crane | curtain | moon | rainman | phoenix;

        int seeds[][2] = {{2,1},{4,1},{5,1},{6,1},{7,1},{8,2},{9,1},{10,1},{11,2}};
        for (auto& s : seeds) add(seed, s[0], s[1]);

        int ribbons[][2] = {{1,2},{2,2},{3,2},{4,2},{5,2},{6,2},{7,2},{9,2},{10,2},{11,3}};
        for (auto& r : ribbons) add(ribbon, r[0], r[1]);

        int dross_list[][2] = {
            {1,3},{1,4},{2,3},{2,4},{3,3},{3,4},{4,3},{4,4},{5,3},{5,4},{6,3},{6,4},{7,3},
            {7,4},{8,3},{8,4},{9,3},{9,4},{10,3},{10,4},{11,4},{12,2},{12,3},{12,4},{9,1}
        };
        for (auto& d : dross_list) add(dross, d[0], d[1]);

        add(boar_deer_butterfly, 6, 1); add(boar_deer_butterfly, 7, 1); add(boar_deer_butterfly, 10, 1);
        flower_sake = curtain | sake;
        moon_sake = moon | sake;

        add(red_ribbon, 1, 2); add(red_ribbon, 2, 2); add(red_ribbon, 3, 2);
        add(blue_ribbon, 6, 2); add(blue_ribbon, 9, 2); add(blue_ribbon, 10, 2);
        red_blue_ribbon = red_ribbon | blue_ribbon;
    }
};

static const KoiKoiMasks MASKS;

// 立っているビット（カードの枚数）を高速に数える
inline int popcount(uint64_t bb) {
    return std::bitset<64>(bb).count();
}

// 役判定（C++のビット演算で爆速化）
std::vector<std::tuple<int, std::string, int>> evaluate_yaku(
    const std::vector<std::vector<int>>& pile_cards, int koikoi_num) {

    std::vector<std::tuple<int, std::string, int>> yaku_list;
    uint64_t pile = cards_to_bitboard(pile_cards);

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

// ポイントの直接計算
int get_yaku_point(const std::vector<std::vector<int>>& pile_cards, int koikoi_num) {
    auto yaku_list = evaluate_yaku(pile_cards, koikoi_num);
    int total_point = 0;
    for (const auto& y : yaku_list) {
        if (std::get<1>(y) != "Koi-Koi") total_point += std::get<2>(y);
    }
    if (koikoi_num <= 3) total_point += koikoi_num;
    else total_point *= (koikoi_num - 2);
    return total_point;
}

// 以前のマルチホット変換
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

// 状態テンソル作成用のカウントを一括処理
std::vector<std::vector<int>> get_yaku_status_features(
    const std::vector<std::vector<int>>& my_hand,
    const std::vector<std::vector<int>>& board,
    const std::vector<std::vector<int>>& my_collect,
    const std::vector<std::vector<int>>& op_collect,
    const std::vector<std::vector<int>>& unseen)
{
    uint64_t mh = cards_to_bitboard(my_hand);
    uint64_t b  = cards_to_bitboard(board);
    uint64_t mc = cards_to_bitboard(my_collect);
    uint64_t oc = cards_to_bitboard(op_collect);
    uint64_t u  = cards_to_bitboard(unseen);

    std::vector<uint64_t> sets = {
        MASKS.crane, MASKS.curtain, MASKS.moon, MASKS.rainman, MASKS.phoenix,
        MASKS.sake, MASKS.boar_deer_butterfly, MASKS.seed, MASKS.red_ribbon,
        MASKS.blue_ribbon, MASKS.red_blue_ribbon, MASKS.ribbon, MASKS.dross
    };

    std::vector<int> num_mh, num_b, num_mc, num_oc, num_u;
    for (uint64_t mask : sets) {
        num_mh.push_back(popcount(mask & mh));
        num_b.push_back(popcount(mask & b));
        num_mc.push_back(popcount(mask & mc));
        num_oc.push_back(popcount(mask & oc));
        num_u.push_back(popcount(mask & u));
    }
    return {num_mh, num_b, num_mc, num_oc, num_u};
}

PYBIND11_MODULE(koikoicore, m) {
    m.doc() = "C++ Core Engine with Bitboards for KoiKoi AI";
    m.def("card_to_multi_hot", &card_to_multi_hot);
    m.def("evaluate_yaku", &evaluate_yaku);
    m.def("get_yaku_point", &get_yaku_point);
    m.def("get_yaku_status_features", &get_yaku_status_features);
}