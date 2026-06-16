// 型をBF16に変更する

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>
#include <string>
#include <tuple>
#include <array>
#include <pybind11/numpy.h>

#if defined(_MSVC_LANG) || defined(_MSC_VER)
#include <intrin.h>
#endif

namespace py = pybind11;

// (suit, rank) から 0〜47 の一意なインデックスを計算
inline constexpr int card_index(int suit, int rank) {
    return (suit - 1) * 4 + (rank - 1);
}

// 複数枚のカードリストを 64-bit 整数（Bitboard）に変換
uint64_t cards_to_bitboard(const std::vector<std::vector<int>>& cards) {
    uint64_t bb = 0;
    for (const auto& c : cards) {
        if (c.size() == 2) {
            int idx = card_index(c[0], c[1]);
            if (idx >= 0 && idx < 48) { // ガード追加
                bb |= (1ULL << idx);
            }
        }
    }
    return bb;
}

// 指摘1 & 7: MASKS の constexpr 化のための構造体定義
struct KoiKoiMasks {
    uint64_t light = 0, seed = 0, ribbon = 0, dross = 0;
    uint64_t boar_deer_butterfly = 0, flower_sake = 0, moon_sake = 0;
    uint64_t red_ribbon = 0, blue_ribbon = 0, red_blue_ribbon = 0;
    uint64_t crane = 0, curtain = 0, moon = 0, rainman = 0, phoenix = 0, sake = 0;

    constexpr KoiKoiMasks() {
        // constexpr 内でラムダやループを使用するための初期化
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

// C++11/14互換の実行時 or コンパイル時定数。C++14以降なら完全にconstexprとして処理されます
static const KoiKoiMasks MASKS;

// 指摘1: 立っているビットをCPU命令で高速に数える (std::bitset の排除)
inline int popcount(uint64_t bb) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bb);
#elif defined(_MSC_VER)
    return (int)__popcnt64(bb);
#else
    // フォールバック (C++20なら std::popcount が使用可能)
    bb =           bb - ((bb >> 1) & 0x5555555555555555ULL);
    bb = (bb & 0x3333333333333333ULL) + ((bb >> 2) & 0x3333333333333333ULL);
    return (int)((((bb + (bb >> 4)) & 0xF0F0F0F0F0F0F0FULL) * 0x101010101010101ULL) >> 56);
#endif
}

// 指摘2: Python⇔C++境界の最適化（Bitboardを直接受け取る高速版の追加）
// 指摘3: get_yaku_point は evaluate_yaku を呼ばず、文字列やベクタを一切生成しない
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
    
    // else を削除し、すべて独立して加算されるように修正
    if ((pile & MASKS.red_ribbon) == MASKS.red_ribbon) total_point += 5;
    if ((pile & MASKS.blue_ribbon) == MASKS.blue_ribbon) total_point += 5;
    
    if (num_ribbon >= 5) total_point += (num_ribbon - 4);

    int num_dross = popcount(pile & MASKS.dross);
    if (num_dross >= 10) total_point += (num_dross - 9);

    // こいこい倍率・加算処理
    if (koikoi_num <= 3) total_point += koikoi_num;
    else total_point *= (koikoi_num - 2);

    return total_point;
}

// 既存のPythonコードとの互換性を維持するラッパー
int get_yaku_point(const std::vector<std::vector<int>>& pile_cards, int koikoi_num) {
    uint64_t pile = cards_to_bitboard(pile_cards);
    return get_yaku_point_by_bitboard(pile, koikoi_num);
}

// 既存互換用の役判定 (デバッグや記録表示用、最適化版)
std::vector<std::tuple<int, std::string, int>> evaluate_yaku_by_bitboard(uint64_t pile, int koikoi_num) {
    std::vector<std::tuple<int, std::string, int>> yaku_list;
    // 必要な時にだけ予約確保してアロケーション回数を減らす
    yaku_list.reserve(12); 

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
    uint64_t pile = cards_to_bitboard(pile_cards);
    return evaluate_yaku_by_bitboard(pile, koikoi_num);
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

// 戻り値の型を std::vector から py::array_t<int> に変更
std::vector<std::vector<int>> get_yaku_status_features_by_bitboard(
    uint64_t mh, uint64_t b, uint64_t mc, uint64_t oc, uint64_t u)
{
    std::vector<int> num_mh, num_b, num_mc, num_oc, num_u;
    // あらかじめサイズを確定（アロケーションを1回のみにする）
    num_mh.resize(13); num_b.resize(13); num_mc.resize(13); num_oc.resize(13); num_u.resize(13);

    // 13個のマスク配列を直接展開し、ループのオーバーヘッドをゼロにする
    const uint64_t m0 = MASKS.crane;                 const uint64_t m1 = MASKS.curtain;
    const uint64_t m2 = MASKS.moon;                  const uint64_t m3 = MASKS.rainman;
    const uint64_t m4 = MASKS.phoenix;               const uint64_t m5 = MASKS.sake;
    const uint64_t m6 = MASKS.boar_deer_butterfly;   const uint64_t m7 = MASKS.seed;
    const uint64_t m8 = MASKS.red_ribbon;            const uint64_t m9 = MASKS.blue_ribbon;
    const uint64_t m10 = MASKS.red_blue_ribbon;      const uint64_t m11 = MASKS.ribbon;
    const uint64_t m12 = MASKS.dross;

    // マイハンド
    num_mh[0] = popcount(m0 & mh);   num_mh[1] = popcount(m1 & mh);   num_mh[2] = popcount(m2 & mh);
    num_mh[3] = popcount(m3 & mh);   num_mh[4] = popcount(m4 & mh);   num_mh[5] = popcount(m5 & mh);
    num_mh[6] = popcount(m6 & mh);   num_mh[7] = popcount(m7 & mh);   num_mh[8] = popcount(m8 & mh);
    num_mh[9] = popcount(m9 & mh);   num_mh[10] = popcount(m10 & mh); num_mh[11] = popcount(m11 & mh);
    num_mh[12] = popcount(m12 & mh);

    // ボード
    num_b[0] = popcount(m0 & b);     num_b[1] = popcount(m1 & b);     num_b[2] = popcount(m2 & b);
    num_b[3] = popcount(m3 & b);     num_b[4] = popcount(m4 & b);     num_b[5] = popcount(m5 & b);
    num_b[6] = popcount(m6 & b);     num_b[7] = popcount(m7 & b);     num_b[8] = popcount(m8 & b);
    num_b[9] = popcount(m9 & b);     num_b[10] = popcount(m10 & b);   num_b[11] = popcount(m11 & b);
    num_b[12] = popcount(m12 & b);

    // マイコレクト
    num_mc[0] = popcount(m0 & mc);   num_mc[1] = popcount(m1 & mc);   num_mc[2] = popcount(m2 & mc);
    num_mc[3] = popcount(m3 & mc);   num_mc[4] = popcount(m4 & mc);   num_mc[5] = popcount(m5 & mc);
    num_mc[6] = popcount(m6 & mc);   num_mc[7] = popcount(m7 & mc);   num_mc[8] = popcount(m8 & mc);
    num_mc[9] = popcount(m9 & mc);   num_mc[10] = popcount(m10 & mc); num_mc[11] = popcount(m11 & mc);
    num_mc[12] = popcount(m12 & mc);

    // オポネントコレクト
    num_oc[0] = popcount(m0 & oc);   num_oc[1] = popcount(m1 & oc);   num_oc[2] = popcount(m2 & oc);
    num_oc[3] = popcount(m3 & oc);   num_oc[4] = popcount(m4 & oc);   num_oc[5] = popcount(m5 & oc);
    num_oc[6] = popcount(m6 & oc);   num_oc[7] = popcount(m7 & oc);   num_oc[8] = popcount(m8 & oc);
    num_oc[9] = popcount(m9 & oc);   num_oc[10] = popcount(m10 & oc); num_oc[11] = popcount(m11 & oc);
    num_oc[12] = popcount(m12 & oc);

    // アンシーン
    num_u[0] = popcount(m0 & u);     num_u[1] = popcount(m1 & u);     num_u[2] = popcount(m2 & u);
    num_u[3] = popcount(m3 & u);     num_u[4] = popcount(m4 & u);     num_u[5] = popcount(m5 & u);
    num_u[6] = popcount(m6 & u);     num_u[7] = popcount(m7 & u);     num_u[8] = popcount(m8 & u);
    num_u[9] = popcount(m9 & u);     num_u[10] = popcount(m10 & u);   num_u[11] = popcount(m11 & u);
    num_u[12] = popcount(m12 & u);

    return {num_mh, num_b, num_mc, num_oc, num_u};
}

// 既存互換用の特徴量抽出関数
std::vector<std::vector<int>> get_yaku_status_features(
    const std::vector<std::vector<int>>& my_hand,
    const std::vector<std::vector<int>>& board,
    const std::vector<std::vector<int>>& my_collect,
    const std::vector<std::vector<int>>& op_collect,
    const std::vector<std::vector<int>>& unseen)
{
    return get_yaku_status_features_by_bitboard(
        cards_to_bitboard(my_hand),
        cards_to_bitboard(board),
        cards_to_bitboard(my_collect),
        cards_to_bitboard(op_collect),
        cards_to_bitboard(unseen)
    );
}

// 新設：文字列生成を完全に排除し、(役ID, 点数) のペアのベクタを返す超軽量版
std::vector<std::pair<int, int>> evaluate_yaku_id_by_bitboard(uint64_t pile, int koikoi_num) {
    std::vector<std::pair<int, int>> yaku_list;
    yaku_list.reserve(12); // 十分な領域をあらかじめ確保

    int num_light = popcount(pile & MASKS.light);
    if (num_light == 5) yaku_list.emplace_back(1, 10);
    else if (num_light == 4 && (pile & MASKS.rainman) == 0) yaku_list.emplace_back(2, 8);
    else if (num_light == 4) yaku_list.emplace_back(3, 7);
    else if (num_light == 3 && (pile & MASKS.rainman) == 0) yaku_list.emplace_back(4, 5);

    int num_seed = popcount(pile & MASKS.seed);
    if ((pile & MASKS.boar_deer_butterfly) == MASKS.boar_deer_butterfly) yaku_list.emplace_back(5, 5);
    if ((pile & MASKS.flower_sake) == MASKS.flower_sake) yaku_list.emplace_back(koikoi_num == 0 ? 6 : 7, koikoi_num == 0 ? 1 : 3);
    if ((pile & MASKS.moon_sake) == MASKS.moon_sake) yaku_list.emplace_back(koikoi_num == 0 ? 8 : 9, koikoi_num == 0 ? 1 : 3);
    if (num_seed >= 5) yaku_list.emplace_back(10, num_seed - 4);

    int num_ribbon = popcount(pile & MASKS.ribbon);
    if ((pile & MASKS.red_blue_ribbon) == MASKS.red_blue_ribbon) yaku_list.emplace_back(11, 10);
    if ((pile & MASKS.red_ribbon) == MASKS.red_ribbon) yaku_list.emplace_back(12, 5);
    if ((pile & MASKS.blue_ribbon) == MASKS.blue_ribbon) yaku_list.emplace_back(13, 5);
    if (num_ribbon >= 5) yaku_list.emplace_back(14, num_ribbon - 4);

    int num_dross = popcount(pile & MASKS.dross);
    if (num_dross >= 10) yaku_list.emplace_back(15, num_dross - 9);
    if (koikoi_num > 0) yaku_list.emplace_back(16, koikoi_num);

    return yaku_list;
}

PYBIND11_MODULE(koikoicore, m) {
    m.doc() = "C++ Core Engine with Bitboards for KoiKoi AI";
    m.def("card_to_multi_hot", &card_to_multi_hot);
    m.def("evaluate_yaku", &evaluate_yaku);
    m.def("get_yaku_point", &get_yaku_point);
    m.def("get_yaku_status_features", &get_yaku_status_features);

    // 新設：Python側を徐々にBitboardベースに移行するための高速API
    m.def("cards_to_bitboard", &cards_to_bitboard, "Convert card list to 64bit integer");
    m.def("get_yaku_point_by_bitboard", &get_yaku_point_by_bitboard);
    m.def("evaluate_yaku_by_bitboard", &evaluate_yaku_by_bitboard);
    m.def("get_yaku_status_features_by_bitboard", &get_yaku_status_features_by_bitboard);
    m.def("evaluate_yaku_id_by_bitboard", &evaluate_yaku_id_by_bitboard);
}