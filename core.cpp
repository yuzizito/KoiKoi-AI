#include <torch/extension.h>
#include <torch/script.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <array>
#include <cmath>
#include <mutex>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#if defined(_MSVC_LANG) || defined(_MSC_VER)
#include <intrin.h>
#endif

namespace py = pybind11;
using namespace pybind11::literals;

constexpr int C_CH = 43; // カード特徴チャネル数
constexpr int G_CH = 7;  // グローバル特徴要素数

struct ModelBox {
    torch::jit::script::Module model;
    std::mutex mtx;
    bool loaded = false;
};
static ModelBox g_box;

constexpr uint64_t mk_mask(std::initializer_list<int> ids) {
    uint64_t m = 0;
    for (int id : ids) m |= (1ULL << id);
    return m;
}

struct Masks {
    uint64_t crane = 1ULL << 0;
    uint64_t curt  = 1ULL << 8;
    uint64_t moon  = 1ULL << 28;
    uint64_t sake  = 1ULL << 32;
    uint64_t rain  = 1ULL << 40;
    uint64_t phon  = 1ULL << 44;

    uint64_t lt = crane | curt | moon | rain | phon;
    uint64_t sd = mk_mask({4, 12, 16, 20, 24, 29, 32, 36, 41});
    uint64_t rb = mk_mask({1, 5, 9, 13, 17, 21, 25, 33, 37, 42});
    uint64_t ks = mk_mask({
        2,3, 6,7, 10,11, 14,15, 18,19, 22,23, 26,27, 30,31,
        32, 34,35, 38,39, 43, 45,46,47
    });

    uint64_t bdb    = mk_mask({20, 24, 36});
    uint64_t f_sake = curt | sake;
    uint64_t m_sake = moon | sake;

    uint64_t r_rib  = mk_mask({1, 5, 9});
    uint64_t b_rib  = mk_mask({21, 33, 37});
    uint64_t rb_rib = r_rib | b_rib;
};
static const Masks M;

struct Rules {
    int lt5 = 10, lt4 = 8, rainy4 = 7, lt3 = 5, bdb = 5;
    bool ena_f_sake = true, ena_m_sake = true;
    int fm_base = 1, fm_koi = 3;
    int rb_rib = 10, r_rib = 5, b_rib = 5;
};
static Rules R;

void set_rules(py::dict d) {
    if (d.contains("lt5")) R.lt5 = d["lt5"].cast<int>();
    if (d.contains("lt4")) R.lt4 = d["lt4"].cast<int>();
    if (d.contains("rainy4")) R.rainy4 = d["rainy4"].cast<int>();
    if (d.contains("lt3")) R.lt3 = d["lt3"].cast<int>();
    if (d.contains("bdb")) R.bdb = d["bdb"].cast<int>();
    if (d.contains("ena_f_sake")) R.ena_f_sake = d["ena_f_sake"].cast<bool>();
    if (d.contains("ena_m_sake")) R.ena_m_sake = d["ena_m_sake"].cast<bool>();
    if (d.contains("fm_base")) R.fm_base = d["fm_base"].cast<int>();
    if (d.contains("fm_koi")) R.fm_koi = d["fm_koi"].cast<int>();
    if (d.contains("rb_rib")) R.rb_rib = d["rb_rib"].cast<int>();
    if (d.contains("r_rib")) R.r_rib = d["r_rib"].cast<int>();
    if (d.contains("b_rib")) R.b_rib = d["b_rib"].cast<int>();
}

template<typename T, int Cap>
struct SVec {
    std::array<T, Cap> d_{};
    int sz_ = 0;

    void clear() { sz_ = 0; }
    void push(const T& v) { if (sz_ < Cap) [[likely]] d_[sz_++] = v; }
    void push_back(const T& v) { push(v); } // エリアス復元
    void pop() { if (sz_ > 0) sz_--; }
    T& back() { return d_[sz_ - 1]; }
    const T& back() const { return d_[sz_ - 1]; }
    int size() const { return sz_; }
    bool empty() const { return sz_ == 0; }
    T& operator[](int i) { return d_[i]; }
    const T& operator[](int i) const { return d_[i]; }

    T* begin() { return d_.data(); }
    T* end() { return d_.data() + sz_; }
    const T* begin() const { return d_.data(); }
    const T* end() const { return d_.data() + sz_; }

    void erase(T* it) {
        int ix = static_cast<int>(it - d_.data());
        for (int i = ix; i < sz_ - 1; ++i) d_[i] = d_[i + 1];
        sz_--;
    }

    void insert_end(const T* f, const T* l) {
        int n = static_cast<int>(l - f);
        if (sz_ + n > Cap) throw std::out_of_range("SVec overflow");
        for (int i = 0; i < n; ++i) d_[sz_++] = f[i];
    }
};

inline int popcnt(uint64_t b) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(b);
#elif defined(_MSC_VER)
    return static_cast<int>(__popcnt64(b));
#else
    b = b - ((b >> 1) & 0x5555555555555555ULL);
    b = (b & 0x3333333333333333ULL) + ((b >> 2) & 0x3333333333333333ULL);
    return static_cast<int>((((b + (b >> 4)) & 0xF0F0F0F0F0F0F0FULL) * 0x101010101010101ULL) >> 56);
#endif
}

inline int ctz(uint64_t b) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(b);
#elif defined(_MSC_VER)
    unsigned long c; _BitScanForward64(&c, b); return static_cast<int>(c);
#else
    int c = 0; while ((b & 1) == 0) { b >>= 1; c++; } return c;
#endif
}

int calc_yaku_pt(uint64_t pile, int koi_num) {
    int pt = 0;
    int n_lt = popcnt(pile & M.lt);
    if (n_lt == 5) pt += R.lt5;
    else if (n_lt == 4 && !(pile & M.rain)) pt += R.lt4;
    else if (n_lt == 4) pt += R.rainy4;
    else if (n_lt == 3 && !(pile & M.rain)) pt += R.lt3;

    int n_sd = popcnt(pile & M.sd);
    if ((pile & M.bdb) == M.bdb) pt += R.bdb;
    
    int s_pt = (koi_num == 0 ? R.fm_base : R.fm_koi);
    if (R.ena_f_sake && (pile & M.f_sake) == M.f_sake) pt += s_pt;
    if (R.ena_m_sake && (pile & M.m_sake) == M.m_sake) pt += s_pt;
    if (n_sd >= 5) pt += (n_sd - 4);

    int n_rb = popcnt(pile & M.rb);
    if ((pile & M.rb_rib) == M.rb_rib) pt += R.rb_rib;
    if ((pile & M.r_rib) == M.r_rib) pt += R.r_rib;
    if ((pile & M.b_rib) == M.b_rib) pt += R.b_rib;
    if (n_rb >= 5) pt += (n_rb - 4);

    int n_ks = popcnt(pile & M.ks);
    if (n_ks >= 10) pt += (n_ks - 9);

    return pt + (koi_num <= 3 ? koi_num : pt * (koi_num - 2));
}

struct Snap {
    uint64_t hand, unseen, my_pile, field, op_pile, my_dis, show;
    uint16_t op_play, op_ign;
    int pt_t, pt_i, rnd, t16, koi_t, koi_i;
    bool is_dl;
    uint8_t st_type;
};

struct Slot {
    int st_type, act;
    float old_val, old_pi, old_mu; // 探索前の純粋な方策(pi)と実際のサンプリング確率(mu)
    float old_log[48];
    bool mask[48];
    Snap snap;
};

struct ExpPars { float temp, eps, noise; };

// write_feat関数の完全書き換え
inline void write_feat(float* c_dst, float* g_dst, const Snap& sn, int max_rnd) {
    std::memset(c_dst, 0, C_CH * 48 * sizeof(float));

    auto set_bb = [c_dst](int row, uint64_t bb, float val = 1.0f) {
        float* p = c_dst + row * 48;
        while (bb) { p[ctz(bb)] = val; bb &= (bb - 1); }
    };

    set_bb(0, sn.hand); set_bb(1, sn.field); set_bb(2, sn.my_pile); set_bb(3, sn.op_pile); set_bb(4, sn.unseen);

    for (int m = 0; m < 12; ++m) {
        uint64_t m_bb = 0xFULL << (m * 4);
        float val = popcnt(sn.unseen & m_bb) * 0.25f;
        std::fill_n(c_dst + 5 * 48 + m * 4, 4, val);
    }

    // C6: 合法マッチフラグ (DISCARDフェーズは場札と一致する手札、PICKフェーズは取得契機札と一致する場札)
    if (sn.st_type == 0) {
        uint64_t h = sn.hand;
        while (h) {
            int c = ctz(h);
            uint64_t m_bb = 0xFULL << ((c / 4) * 4);
            if (sn.field & m_bb) c_dst[6 * 48 + c] = 1.0f;
            h &= (h - 1);
        }
    } else if (sn.st_type == 1 && sn.show != 0) {
        int s_c = ctz(sn.show);
        uint64_t m_bb = 0xFULL << ((s_c / 4) * 4);
        set_bb(6, sn.field & m_bb, 1.0f);
    }

    // --- ポテンシャル (取りに行く価値) ---
    auto set_pot = [&](int r, uint64_t mk, int req, int lim, uint64_t sf, uint64_t op, bool is_fixed) {
        int op_cnt = popcnt(op & mk);
        int my_cnt = popcnt(sf & mk);

        if (is_fixed && op_cnt > 0) return;
        if (!is_fixed && op_cnt >= lim) return;
        if (is_fixed && my_cnt >= req) return; // 固定役完成後はポテンシャル0

        int n;
        if (my_cnt >= req) {
            int total_cards = popcnt(mk);
            if (my_cnt + op_cnt >= total_cards) return; // 札枯渇により発展不可能な場合は0
            n = 1; // 発展可能であれば、常に次は「あと1枚」で更新される
        } else {
            n = req - my_cnt;
        }
        
        float val = std::max(0.0f, 0.1f * (11 - n));
        set_bb(r, mk, val);
    };

    // --- 光役専用ポテンシャル ---
    auto set_lt_pot = [&](int r, uint64_t sf, uint64_t op) {
        int my_reg = popcnt(sf & (M.lt & ~M.rain));
        int op_reg = popcnt(op & (M.lt & ~M.rain));
        bool my_rain = (sf & M.rain) != 0;
        bool op_rain = (op & M.rain) != 0;

        // 柳以外の光札 (次に達成可能な役までの最短距離を算出)
        int min_n_reg = 999;
        if (my_reg < 3 && op_reg <= 1) min_n_reg = std::min(min_n_reg, 3 - my_reg);
        if ((my_reg < 3 || my_rain == 0) && op_reg <= 1 && op_rain == 0) min_n_reg = std::min(min_n_reg, 3 - my_reg + (1 - my_rain));
        if (my_reg < 4 && op_reg == 0) min_n_reg = std::min(min_n_reg, 4 - my_reg);
        if ((my_reg < 4 || my_rain == 0) && op_reg == 0 && op_rain == 0) min_n_reg = std::min(min_n_reg, 4 - my_reg + (1 - my_rain));

        if (min_n_reg != 999) set_bb(r, M.lt & ~M.rain, std::min(1.0f, 0.1f * (11 - min_n_reg)));

        // 柳の札 (三光には貢献しない)
        int min_n_rain = 999;
        if ((my_reg < 3 || my_rain == 0) && op_reg <= 1 && op_rain == 0) min_n_rain = std::min(min_n_rain, 3 - my_reg + (1 - my_rain));
        if ((my_reg < 4 || my_rain == 0) && op_reg == 0 && op_rain == 0) min_n_rain = std::min(min_n_rain, 4 - my_reg + (1 - my_rain));
        
        if (min_n_rain != 999) set_bb(r, M.rain, std::min(1.0f, 0.1f * (11 - min_n_rain)));
    };

    // --- 確定ステータス ---
    auto set_stat = [&](int r, uint64_t mk, int req, uint64_t sf) {
        if (popcnt(sf & mk) >= req) set_bb(r, mk, 1.0f);
    };

    // --- 光役専用確定ステータス ---
    auto set_lt_stat = [&](int r, uint64_t sf) {
        int my_reg = popcnt(sf & (M.lt & ~M.rain));
        bool my_rain = (sf & M.rain) != 0;
        
        if (my_reg >= 3) set_bb(r, M.lt & ~M.rain, 1.0f); // 三光以上なら柳以外はすべて役の一部
        if ((my_reg == 3 || my_reg == 4) && my_rain) set_bb(r, M.rain, 1.0f); // 雨四光・五光なら柳も役の一部
    };

    // C7-C15: 自分 役ポテンシャル
    set_lt_pot(7, sn.my_pile, sn.op_pile);
    set_pot(8,  M.bdb,    3,  1,  sn.my_pile, sn.op_pile, true);
    set_pot(9,  M.r_rib,  3,  1,  sn.my_pile, sn.op_pile, true);
    set_pot(10, M.b_rib,  3,  1,  sn.my_pile, sn.op_pile, true);
    set_pot(11, M.f_sake, 2,  1,  sn.my_pile, sn.op_pile, true);
    set_pot(12, M.m_sake, 2,  1,  sn.my_pile, sn.op_pile, true);
    set_pot(13, M.sd,     5,  5,  sn.my_pile, sn.op_pile, false);
    set_pot(14, M.rb,     5,  6,  sn.my_pile, sn.op_pile, false);
    set_pot(15, M.ks,    10, 15,  sn.my_pile, sn.op_pile, false);

    // C16-C24: 自分 確定ステータス
    set_lt_stat(16, sn.my_pile);
    set_stat(17, M.bdb,    3, sn.my_pile);
    set_stat(18, M.r_rib,  3, sn.my_pile);
    set_stat(19, M.b_rib,  3, sn.my_pile);
    set_stat(20, M.f_sake, 2, sn.my_pile);
    set_stat(21, M.m_sake, 2, sn.my_pile);
    set_stat(22, M.sd,     5, sn.my_pile);
    set_stat(23, M.rb,     5, sn.my_pile);
    set_stat(24, M.ks,    10, sn.my_pile);

    // C25-C33: 相手 役ポテンシャル
    set_lt_pot(25, sn.op_pile, sn.my_pile);
    set_pot(26, M.bdb,    3,  1,  sn.op_pile, sn.my_pile, true);
    set_pot(27, M.r_rib,  3,  1,  sn.op_pile, sn.my_pile, true);
    set_pot(28, M.b_rib,  3,  1,  sn.op_pile, sn.my_pile, true);
    set_pot(29, M.f_sake, 2,  1,  sn.op_pile, sn.my_pile, true);
    set_pot(30, M.m_sake, 2,  1,  sn.op_pile, sn.my_pile, true);
    set_pot(31, M.sd,     5,  5,  sn.op_pile, sn.my_pile, false);
    set_pot(32, M.rb,     5,  6,  sn.op_pile, sn.my_pile, false);
    set_pot(33, M.ks,    10, 15,  sn.op_pile, sn.my_pile, false);

    // C34-C42: 相手 確定ステータス
    set_lt_stat(34, sn.op_pile);
    set_stat(35, M.bdb,    3, sn.op_pile);
    set_stat(36, M.r_rib,  3, sn.op_pile);
    set_stat(37, M.b_rib,  3, sn.op_pile);
    set_stat(38, M.f_sake, 2, sn.op_pile);
    set_stat(39, M.m_sake, 2, sn.op_pile);
    set_stat(40, M.sd,     5, sn.op_pile);
    set_stat(41, M.rb,     5, sn.op_pile);
    set_stat(42, M.ks,    10, sn.op_pile);

    g_dst[0] = (sn.pt_t - sn.pt_i) / 30.0f;
    g_dst[1] = calc_yaku_pt(sn.my_pile, sn.koi_t) / 10.0f;
    g_dst[2] = calc_yaku_pt(sn.op_pile, sn.koi_i) / 10.0f;
    g_dst[3] = sn.koi_t > 0 ? 1.0f : (sn.koi_i > 0 ? -1.0f : 0.0f);
    g_dst[4] = sn.t16 / 16.0f;
    g_dst[5] = sn.is_dl ? 1.0f : -1.0f;
    g_dst[6] = static_cast<float>(sn.rnd) / max_rnd;
}

class StateMgr {
public:
    uint64_t hand[3]{}, pile[3]{}, field = 0, stock = 0, show = 0, pair = 0, dis_hist[3]{};
    uint16_t play[3]{}, ign[3]{};

    void init(uint64_t h1, uint64_t h2, uint64_t f, uint64_t s) {
        hand[1] = h1; hand[2] = h2; field = f; stock = s;
        pile[1] = 0; pile[2] = 0; show = 0; pair = 0; dis_hist[1] = 0; dis_hist[2] = 0;
        play[1] = 0; play[2] = 0; ign[1] = 0; ign[2] = 0;
    }

    void discard(int p, int c, uint64_t pr) {
        hand[p] &= ~(1ULL << c); show = (1ULL << c); pair = pr; dis_hist[p] |= (1ULL << c);
        int st = c / 4; play[p] |= (1 << st);
        uint16_t fs = 0; uint64_t f = field;
        while (f) { fs |= (1 << (ctz(f) / 4)); f &= (f - 1); }
        ign[p] |= (fs & ~(1 << st));
    }

    void pick(int p, uint64_t cl, uint64_t fr) { pile[p] |= cl; field = fr; show = 0; pair = 0; }
    void draw(int c, uint64_t pr) { stock &= ~(1ULL << c); show = (1ULL << c); pair = pr; }

    py::tuple feat(bool is_koi, int pt_t, int pt_i, int rnd, int t16, bool is_dl, int koi_t, int koi_i, int tp, int ip, int mx_rnd) {
        auto c_arr = py::array_t<float>({C_CH, 48});
        auto g_arr = py::array_t<float>({7});
        Snap sn{hand[tp], stock | hand[ip], pile[tp], field, pile[ip], dis_hist[tp], show, play[ip], ign[ip], pt_t, pt_i, rnd, t16, koi_t, koi_i, is_dl, static_cast<uint8_t>(is_koi ? 2 : 0)};
        write_feat(c_arr.mutable_data(), g_arr.mutable_data(), sn, mx_rnd);
        return py::make_tuple(c_arr, g_arr);
    }

    py::array_t<bool> mask(int t, int p) {
        int sz = (t == 2) ? 2 : 48;
        auto arr = py::array_t<bool>(sz);
        bool* ptr = arr.mutable_data();
        std::memset(ptr, 0, sz * sizeof(bool));
        if (t == 2) { ptr[0] = true; ptr[1] = true; }
        else {
            uint64_t b = (t == 0) ? hand[p] : pair;
            while (b) { ptr[ctz(b)] = true; b &= (b - 1); }
        }
        return arr;
    }
};

class TraceBuf {
public:
    int cap, sz = 0;
    torch::Tensor c_st, g_st, act, rew, old_log, old_val, mask, old_pi, old_mu;
    float *cp, *gp, *rp, *lp, *vp, *pi_p, *mu_p; int64_t* ap; bool* mp;

    explicit TraceBuf(int c) : cap(c) {
        auto opt_f = torch::TensorOptions().dtype(torch::kFloat32);
        c_st = torch::empty({cap, C_CH, 48}, opt_f); g_st = torch::empty({cap, G_CH}, opt_f);
        act = torch::empty({cap}, torch::kInt64); rew = torch::empty({cap}, opt_f);
        old_log = torch::empty({cap, 48}, opt_f); old_val = torch::empty({cap}, opt_f);
        mask = torch::empty({cap, 48}, torch::kBool); 
        old_pi = torch::empty({cap}, opt_f); old_mu = torch::empty({cap}, opt_f);

        cp = c_st.data_ptr<float>(); gp = g_st.data_ptr<float>(); ap = act.data_ptr<int64_t>();
        rp = rew.data_ptr<float>(); lp = old_log.data_ptr<float>(); vp = old_val.data_ptr<float>();
        mp = mask.data_ptr<bool>(); 
        pi_p = old_pi.data_ptr<float>(); mu_p = old_mu.data_ptr<float>();
    }

    void clear() { sz = 0; }
    void push(const Slot& sl, float ret, int mx_rnd) {
        int ix;
        #pragma omp atomic capture
        ix = sz++;

        if (ix >= cap) {
            #pragma omp atomic write
            sz = cap; return;
        }
        ap[ix] = sl.act; rp[ix] = ret; vp[ix] = sl.old_val; 
        pi_p[ix] = sl.old_pi; mu_p[ix] = sl.old_mu;
        std::memcpy(lp + ix * 48, sl.old_log, 48 * sizeof(float));
        std::memcpy(mp + ix * 48, sl.mask, 48 * sizeof(bool));
        write_feat(cp + ix * C_CH * 48, gp + ix * G_CH, sl.snap, mx_rnd);
    }

    py::dict fin() {
        int s = std::min(cap, sz);
        return py::dict("card_states"_a=c_st.slice(0,0,s).clone(), "global_states"_a=g_st.slice(0,0,s).clone(),
                        "actions"_a=act.slice(0,0,s).clone(), "rewards"_a=rew.slice(0,0,s).clone(),
                        "old_logits"_a=old_log.slice(0,0,s).clone(), "old_values"_a=old_val.slice(0,0,s).clone(),
                        "legal_masks"_a=mask.slice(0,0,s).clone(), 
                        "old_pi"_a=old_pi.slice(0,0,s).clone(), "old_mu"_a=old_mu.slice(0,0,s).clone());
    }
};

enum class GState : uint8_t { DISCARD, DISCARD_PICK, DRAW, DRAW_PICK, KOIKOI, ROUND_OVER, GAME_OVER };

class Env {
public:
    int rnd = 1, dlr = 1, win = 1, t16 = 1, t_pt = 0, pt[3]{}, koi[3][8]{};
    bool ex = false, wait = false;
    float phi_begin[3]{0.0f, 0.0f, 0.0f};
    GState st = GState::ROUND_OVER;
    StateMgr mgr; std::mt19937 rng;
    SVec<int, 48> hand[3], field, stock, pile[3];
    SVec<int, 4> show; SVec<int, 8> coll;

    explicit Env(int seed = 42) : rng(seed) { reset_game(); }
    void set_seed(unsigned int s) { rng.seed(s); }
    void reset_game() { rnd = 1; pt[1] = 30; pt[2] = 30; dlr = std::uniform_int_distribution<>(1, 2)(rng); win = dlr; st = GState::ROUND_OVER; }
    void reset_round() {
        if (win != 0 && win != -1) dlr = win;
        t16 = 1; pile[1].clear(); pile[2].clear(); std::memset(koi, 0, sizeof(koi)); show.clear(); coll.clear();
        win = 0; ex = false; t_pt = 0; deal();

        uint64_t h1=0, h2=0, f=0, s=0;
        for (int c : hand[1]) h1 |= (1ULL << c); for (int c : hand[2]) h2 |= (1ULL << c);
        for (int c : field) if (c != -1) f |= (1ULL << c); for (int c : stock) s |= (1ULL << c);
        mgr.init(h1, h2, f, s); 
        st = GState::DISCARD; 
        wait = (hand[turn_p()].size() > 1);
    }

    int turn_p() const { return ((t16 + dlr) % 2 == 0) ? 1 : 2; }
    int idle_p() const { return 3 - turn_p(); }
    int turn8() const { return (t16 + 1) / 2; }
    int koi_num(int p) const { int s=0; for(int v:koi[p]) s+=v; return s; }

    int rnd_pt(int p) {
        if (win == 0) return 0;
        return ex ? (dlr == p ? 1 : -1) : (win == p ? calc_yaku_pt(mgr.pile[win], koi_num(win)) : -calc_yaku_pt(mgr.pile[win], koi_num(win)));
    }
    bool needs_act() const { return wait && st != GState::ROUND_OVER && st != GState::GAME_OVER && st != GState::DRAW; }

    SVec<int, 4> get_pairs() const {
        SVec<int, 4> pr; if (show.empty()) return pr;
        for (int c : field) if (c != -1 && c / 4 == show[0] / 4) pr.push(c);
        return pr;
    }
    void rm_field(int c) { auto it = std::find(field.begin(), field.end(), c); if (it != field.end()) *it = -1; }

    void collect(int c) {
        SVec<int, 4> pr = get_pairs(); coll.clear();
        if (pr.empty()) *std::find(field.begin(), field.end(), -1) = show[0];
        else {
            coll.insert_end(show.begin(), show.end());
            if (pr.size() == 1 || pr.size() == 3) { coll.insert_end(pr.begin(), pr.end()); for(int pc:pr) rm_field(pc); }
            else { coll.push(c); rm_field(c); }
            pile[turn_p()].insert_end(coll.begin(), coll.end());
        }
    }

    void step(int act) {
        int p = turn_p();
        switch (st) {
            case GState::DISCARD:
                if (act == -1) {
                    if (hand[p].empty()) throw std::logic_error("Discard act=-1 but hand is empty");
                    act = hand[p][0];
                }
                t_pt = calc_yaku_pt(mgr.pile[p], koi_num(p));
                hand[p].erase(std::find(hand[p].begin(), hand[p].end(), act));
                show.clear(); show.push(act);
                { SVec<int, 4> pr = get_pairs(); uint64_t b=0; for(int c:pr) b|=(1ULL<<c);
                  mgr.discard(p, act, b); st = GState::DISCARD_PICK; wait = (pr.size() == 2); }
                break;
            case GState::DISCARD_PICK:
                collect(act);
                { uint64_t cb=0, fb=0; for(int c:coll) cb|=(1ULL<<c); for(int c:field) if(c!=-1) fb|=(1ULL<<c);
                  mgr.pick(p, cb, fb); st = GState::DRAW; wait = false; }
                break;
            case GState::DRAW:
                { int d = stock.back(); stock.pop(); show.clear(); show.push(d);
                  SVec<int, 4> pr = get_pairs(); uint64_t b=0; for(int c:pr) b|=(1ULL<<c);
                  mgr.draw(d, b); st = GState::DRAW_PICK; wait = (pr.size() == 2); }
                break;
            case GState::DRAW_PICK:
                collect(act);
                { uint64_t cb=0, fb=0; for(int c:coll) cb|=(1ULL<<c); for(int c:field) if(c!=-1) fb|=(1ULL<<c);
                  mgr.pick(p, cb, fb); st = GState::KOIKOI; wait = (calc_yaku_pt(mgr.pile[p], koi_num(p)) > t_pt) && (turn8() < 8); }
                break;
            case GState::KOIKOI:
                { bool kk = (act != 0) && !(turn8() == 8 && calc_yaku_pt(mgr.pile[p], koi_num(p)) > t_pt);
                  if (wait) koi[p][turn8() - 1] = kk ? 1 : 0;
                  if (!kk) { st = GState::ROUND_OVER; wait = false; win = p; }
                  else if (t16 == 16) { st = GState::ROUND_OVER; wait = false; ex = true; win = dlr; }
                  else { t16++; st = GState::DISCARD; wait = (hand[turn_p()].size() > 1); } }
                break;
            default: break;
        }
    }

private:
    void deal() {
        std::array<int, 48> cd{}; for(int i=0;i<48;++i) cd[i]=i;
        while (true) {
            std::shuffle(cd.begin(), cd.end(), rng);
            int d = dlr, nd = 3 - dlr; hand[d].clear(); hand[nd].clear(); field.clear(); stock.clear();
            for (int i=0;i<8;++i) hand[d].push(cd[i]); std::sort(hand[d].begin(), hand[d].end());
            for (int i=8;i<16;++i) hand[nd].push(cd[i]); std::sort(hand[nd].begin(), hand[nd].end());
            for (int i=16;i<24;++i) { field.push(cd[i]); field.push(-1); }
            for (int i=24;i<48;++i) stock.push(cd[i]);

            uint64_t h1=0, h2=0, fb=0;
            for (int c:hand[1]) h1|=(1ULL<<c); for (int c:hand[2]) h2|=(1ULL<<c); for (int c:field) if(c!=-1) fb|=(1ULL<<c);
            bool ok = true;
            for (int s=0;s<12;++s) { uint64_t m = 0xFULL << (s * 4); if (popcnt(h1&m)==4 || popcnt(h2&m)==4 || popcnt(fb&m)==4) { ok=false; break; } }
            if (ok) break;
        }
    }
};

class BatchSim {
public:
    int n_env, tg_games, fin = 0;
    bool force_stop_koikoi;
    int mx_rnd;
    std::vector<float> wpm; // Design Intent: Pythonから渡された勝率マトリクス[2, 9, 61]のフラット配列
    std::vector<Env> envs; std::vector<SVec<Slot, 512>> tr[3];
    TraceBuf b_dis, b_pic, b_koi; torch::Device dev;
    torch::Tensor ct[3], gt[3], mk[3]; ExpPars ep;
private:
    std::vector<int64_t> ab_; std::vector<float> pi_b_, mu_b_, vb_;

public:
    BatchSim(int ne, int tg, int cd, int cp, int ck, const std::string& dv, ExpPars e, bool fs, int mx_r, const std::vector<float>& wp)
        : n_env(ne), tg_games(tg), force_stop_koikoi(fs), mx_rnd(mx_r), wpm(wp), b_dis(cd), b_pic(cp), b_koi(ck), dev(dv), ep(e), ab_(ne), pi_b_(ne), mu_b_(ne), vb_(ne) {
        envs.reserve(ne); std::random_device rd;
        for (int i=0;i<ne;++i) envs.emplace_back(rd());
        for (int p=1;p<=2;++p) tr[p].resize(ne);
        auto of = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        auto ob = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);
        for (int i=0;i<3;++i) { ct[i] = torch::empty({ne, C_CH, 48}, of); gt[i] = torch::empty({ne, G_CH}, of); }
        mk[0] = torch::empty({ne, 48}, ob); mk[1] = torch::empty({ne, 48}, ob); mk[2] = torch::empty({ne, 2}, ob);
        reset();
    }

    // Design Intent: 現在手番プレイヤー視点での勝率を win_prob_mat[2, 9, 61] から高速に取得する
    float get_phi(int pt, int op_pt, int dlr, int rnd) const {
        int safe_rnd = std::max(1, std::min(rnd, 8));
        int diff = pt - op_pt;
        diff = std::max(-30, std::min(diff, 30));
        int idx_diff = diff + 30;
        
        int idx = (dlr * 9 * 61) + (safe_rnd * 61) + idx_diff;
        return wpm[idx];
    }

    void reset() {
        fin = 0; 
        for (auto& e : envs) { 
            e.reset_game(); 
            e.reset_round(); 
            e.phi_begin[1] = get_phi(e.pt[1], e.pt[2], e.dlr == 1 ? 1 : 0, e.rnd);
            e.phi_begin[2] = get_phi(e.pt[2], e.pt[1], e.dlr == 2 ? 1 : 0, e.rnd);
        }
        for (int p=1;p<=2;++p) for (auto& v : tr[p]) v.clear();
        b_dis.clear(); b_pic.clear(); b_koi.clear();
    }

    py::dict fin_bufs() { return py::dict("discard"_a=b_dis.fin(), "pick"_a=b_pic.fin(), "koikoi"_a=b_koi.fin()); }

    Snap get_snap(const Env& e, int pt, int pi, int type) {
        uint64_t my_p = e.mgr.pile[pt];
        uint64_t sh = 0;
        
        if (type == 1 && !e.show.empty()) {
            sh = (1ULL << e.show[0]);
            my_p |= sh;
        }

        return {
            e.mgr.hand[pt],
            e.mgr.stock | e.mgr.hand[pi],
            my_p,
            e.mgr.field,
            e.mgr.pile[pi],
            e.mgr.dis_hist[pt], 
            sh, // <-- 追加部分
            e.mgr.play[pi], 
            e.mgr.ign[pi], 
            e.pt[pt], e.pt[pi], e.rnd, e.t16, 
            e.koi_num(pt), e.koi_num(pi), 
            e.dlr == pt, 
            static_cast<uint8_t>(type)
        };
    }

    void set_mask(int i, bool* m, int t) {
        int sz = (t == 2) ? 2 : 48; std::memset(m, 0, sz * sizeof(bool));
        if (t == 2) { 
            m[0] = true; 
            m[1] = !force_stop_koikoi; // Design Intent: force_stop_koikoiがtrueの場合、こいこい継続(1)を合法手から除外し必ずストップ(0)に誘導する
        }
        else { uint64_t b = (t == 0) ? envs[i].mgr.hand[envs[i].turn_p()] : envs[i].mgr.pair; while (b) { m[ctz(b)] = true; b &= (b - 1); } }
    }

    void step_idle() {
        int f = 0;
        #pragma omp parallel for reduction(+:f)
        for (int i = 0; i < n_env; ++i) {
            auto& e = envs[i];
            while (!e.needs_act() && e.st != GState::GAME_OVER) {
                if (e.st == GState::ROUND_OVER) {
                    e.pt[1] += e.rnd_pt(1); e.pt[2] += e.rnd_pt(2);
                    float phi_end[3];
                    bool game_end = (e.pt[1] <= 0 || e.pt[2] <= 0 || e.rnd == mx_rnd);

                    // Design Intent: ラウンド終了状態に応じた phi_end の取得（終局時は勝敗固定値）
                    if (game_end) {
                        if (e.pt[1] > e.pt[2]) { phi_end[1] = 1.0f; phi_end[2] = 0.0f; }
                        else if (e.pt[2] > e.pt[1]) { phi_end[1] = 0.0f; phi_end[2] = 1.0f; }
                        else { phi_end[1] = 0.5f; phi_end[2] = 0.5f; }
                    } else {
                        e.rnd++; 
                        e.reset_round(); 
                        phi_end[1] = get_phi(e.pt[1], e.pt[2], e.dlr == 1 ? 1 : 0, e.rnd);
                        phi_end[2] = get_phi(e.pt[2], e.pt[1], e.dlr == 2 ? 1 : 0, e.rnd);
                    }

                    // Design Intent: 時間割引を廃止し、ラウンド内の全行動に PBRS 報酬 (phi_end - phi_begin) を一括付与
                    for (int p = 1; p <= 2; ++p) {
                        auto& t = tr[p][i];
                        if (!t.empty()) {
                            float r = phi_end[p] - e.phi_begin[p];
                            int T = t.size();
                            for (int k = 0; k < T; ++k) {
                                (t[k].st_type == 0 ? b_dis : (t[k].st_type == 1 ? b_pic : b_koi)).push(t[k], r, mx_rnd);
                            }
                            t.clear();
                        }
                    }

                    if (game_end) {
                        f++; 
                        e.reset_game(); 
                        e.reset_round();
                        e.phi_begin[1] = get_phi(e.pt[1], e.pt[2], e.dlr == 1 ? 1 : 0, e.rnd);
                        e.phi_begin[2] = get_phi(e.pt[2], e.pt[1], e.dlr == 2 ? 1 : 0, e.rnd);
                    } else {
                        e.phi_begin[1] = phi_end[1];
                        e.phi_begin[2] = phi_end[2];
                    }
                } else {
                    e.step(-1);
                }
            }
        }
        fin += f;
    }

    void sample_neurd(int t, int n, const int* ix_list, const float* l_ptr, const float* q_ptr, const bool* m_ptr, int64_t* a_ptr, float* pi_ptr, float* mu_ptr, float* v_ptr) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f); 
        int cols = (t == 2) ? 2 : 48;
        
        for (int i = 0; i < n; ++i) {
            SVec<int, 48> leg; 
            for (int c = 0; c < cols; ++c) {
                if (m_ptr[i * cols + c]) leg.push(c);
            }
            if (leg.empty()) { a_ptr[i] = 0; pi_ptr[i] = 1.0f; mu_ptr[i] = 1.0f; v_ptr[i] = 0.0f; continue; }

            std::array<float, 48> base_pi{};
            float mx = -1e9f, sum_e = 0.0f;
            for (int a : leg) if (l_ptr[i * cols + a] > mx) mx = l_ptr[i * cols + a];

            if (ep.temp <= 0.0f) {
                int bc = 0; for (int a : leg) if (l_ptr[i * cols + a] == mx) bc++;
                for (int a : leg) base_pi[a] = (l_ptr[i * cols + a] == mx) ? 1.0f / bc : 0.0f;
            } else {
                for (int a : leg) { 
                    base_pi[a] = std::exp((l_ptr[i * cols + a] - mx) / ep.temp); 
                    sum_e += base_pi[a]; 
                }
                for (int a : leg) base_pi[a] /= sum_e;
            }

            std::array<float, 48> pure_pi{};
            for (int a : leg)
                pure_pi[a] = base_pi[a];

            if (ep.noise > 0.0f && leg.size() > 1) {
                float alpha = 0.3f; 
                std::gamma_distribution<float> gamma(alpha, 1.0f);
                std::array<float, 48> dir{};
                float sum_dir = 0.0f;
                
                for (int a : leg) {
                    dir[a] = gamma(envs[ix_list[i]].rng);
                    sum_dir += dir[a];
                }
                
                if (sum_dir > 0.0f) {
                    for (int a : leg) {
                        base_pi[a] = (1.0f - ep.noise) * base_pi[a] + ep.noise * (dir[a] / sum_dir);
                    }
                }
            }

            std::array<float, 48> mu{};
            float uni = ep.eps / leg.size();
            float w_pi = 1.0f - ep.eps;
            for (int a : leg) {
                mu[a] = uni + w_pi * base_pi[a];
            }

            float r = dist(envs[ix_list[i]].rng), acc = 0.0f; 
            int ch = leg.back();
            for (int a : leg) { 
                acc += mu[a]; 
                if (r <= acc) { ch = a; break; } 
            }
            
            a_ptr[i] = ch; 
            // それぞれ純粋方策πと挙動方策μを独立して保存する
            pi_ptr[i] = pure_pi[ch];
            mu_ptr[i] = mu[ch];

            float vs = 0.0f; 
            for (int a : leg) vs += pure_pi[a] * q_ptr[i * cols + a];
            v_ptr[i] = vs;
        }
    }

    void play() {
        std::array<std::vector<int>, 3> req{};
        float *cp[3] = {ct[0].data_ptr<float>(), ct[1].data_ptr<float>(), ct[2].data_ptr<float>()};
        float *gp[3] = {gt[0].data_ptr<float>(), gt[1].data_ptr<float>(), gt[2].data_ptr<float>()};
        bool  *mp[3] = {mk[0].data_ptr<bool>(),  mk[1].data_ptr<bool>(),  mk[2].data_ptr<bool>()};

        while (fin < tg_games) {
            step_idle(); if (fin >= tg_games) break;
            for (int p = 1; p <= 2; ++p) {
                for (int t=0;t<3;++t) req[t].clear();
                for (int i=0;i<n_env;++i) if (envs[i].needs_act() && envs[i].turn_p() == p) req[(envs[i].st == GState::DISCARD) ? 0 : (envs[i].st == GState::KOIKOI ? 2 : 1)].push_back(i);

                torch::NoGradGuard ng;
                for (int t = 0; t < 3; ++t) {
                    int n = req[t].size(); if (n == 0) continue;
                    #pragma omp parallel for
                    for (int i=0;i<n;++i) { int ei = req[t][i]; write_feat(cp[t] + i * C_CH * 48, gp[t] + i * G_CH, get_snap(envs[ei], p, 3 - p, t), mx_rnd); set_mask(ei, mp[t] + i * ((t == 2) ? 2 : 48), t); }

                    auto out = g_box.model.forward({ct[t].slice(0,0,n).to(dev,true), gt[t].slice(0,0,n).to(dev,true)}).toTuple();
                    torch::Tensor l_cpu = out->elements()[(t == 2) ? 2 : 0].toTensor().cpu(), q_cpu = out->elements()[(t == 2) ? 3 : 1].toTensor().cpu();

                    sample_neurd(t, n, req[t].data(), l_cpu.data_ptr<float>(), q_cpu.data_ptr<float>(), mp[t], ab_.data(), pi_b_.data(), mu_b_.data(), vb_.data());

                    #pragma omp parallel for
                    for (int i=0;i<n;++i) {
                        int ei = req[t][i]; Slot sl{t, static_cast<int>(ab_[i]), vb_[i], pi_b_[i], mu_b_[i]};
                        int cols = (t == 2) ? 2 : 48; std::memset(sl.old_log, 0, 48 * sizeof(float)); std::memset(sl.mask, 0, 48 * sizeof(bool));
                        for (int c=0;c<cols;++c) { sl.old_log[c] = l_cpu.data_ptr<float>()[i * cols + c]; sl.mask[c] = mp[t][i * cols + c]; }
                        sl.snap = get_snap(envs[ei], p, 3 - p, t); tr[p][ei].push_back(sl); envs[ei].step(sl.act);
                    }
                }
            }
        }
    }
};

py::list run_sim(int nth, int ne, int tg, int cd, int cp, int ck, const std::string& mod_bytes, const std::string& dv, int mx_rnd, float tmp, float eps, float nz, bool force_stop_koikoi, const std::vector<float>& wpm_vec) {
    { 
        std::lock_guard<std::mutex> lk(g_box.mtx); 
        if (!g_box.loaded) { 
            std::istringstream stream(mod_bytes);
            g_box.model = torch::jit::load(stream);
            g_box.model.eval(); 
            g_box.loaded = true; 
        } 
    }
    ExpPars ep{tmp, eps, nz}; std::vector<std::unique_ptr<BatchSim>> s(nth);
    for(int i=0;i<nth;++i) s[i] = std::make_unique<BatchSim>(ne, tg, cd, cp, ck, dv, ep, force_stop_koikoi, mx_rnd, wpm_vec);
    { py::gil_scoped_release rel;
        #pragma omp parallel for schedule(dynamic)
        for (int i=0;i<nth;++i) s[i]->play(); }
    py::list res; for (int i=0;i<nth;++i) res.append(s[i]->fin_bufs()); return res;
}

// グローバルに相手モデルをキャッシュするための構造体を追加
struct ArenaBox {
    torch::jit::script::Module opp_mod;
    std::mutex mtx;
    bool loaded = false;
};
static ArenaBox a_box;

class ArenaSim {
public:
    int n_env, tg, task_idx = 0;
    int w1 = 0, w2 = 0, draws = 0, fin = 0;
    int start_rnd, start_p1, start_p2, start_dlr;
    std::vector<Env> envs;
    std::vector<bool> is_mir;
    std::vector<unsigned int> seeds;
    torch::jit::Module m1, m2;
    torch::Device dev;
    torch::Tensor ct[3], gt[3], mk[3];
    std::vector<int64_t> ab_;

    ArenaSim(int ne, int t, torch::jit::Module p1, torch::jit::Module p2, const std::string& dv, int s_r = 0, int s_p1 = 30, int s_p2 = 30, int s_d = 0)
        : n_env(ne), tg(t), m1(p1), m2(p2), dev(dv), ab_(ne), start_rnd(s_r), start_p1(s_p1), start_p2(s_p2), start_dlr(s_d) {
        
        envs.reserve(ne);
        is_mir.resize(ne, false);
        
        int num_seeds = (start_rnd > 0) ? tg : (tg / 2);
        seeds.resize(num_seeds);
        std::random_device rd;
        for (int i = 0; i < num_seeds; ++i) seeds[i] = rd();

        auto of = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        auto ob = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);
        for (int i = 0; i < 3; ++i) { 
            ct[i] = torch::empty({ne, C_CH, 48}, of); 
            gt[i] = torch::empty({ne, G_CH}, of); 
        }
        mk[0] = torch::empty({ne, 48}, ob); mk[1] = torch::empty({ne, 48}, ob); mk[2] = torch::empty({ne, 2}, ob);

        for (int i = 0; i < ne; ++i) { envs.emplace_back(42); assign_task(i); }
    }

    bool assign_task(int i) {
        if (task_idx >= tg) { envs[i].st = GState::GAME_OVER; return false; }
        
        bool mir = false;
        int s_idx = 0;

        if (start_rnd > 0) {
            s_idx = task_idx;
            mir = false;
            envs[i].rnd = start_rnd;
            envs[i].pt[1] = start_p1;
            envs[i].pt[2] = start_p2;
            envs[i].dlr = (start_dlr == 0) ? std::uniform_int_distribution<>(1, 2)(envs[i].rng) : start_dlr;
            envs[i].win = envs[i].dlr;
            envs[i].st = GState::ROUND_OVER;
        } else {
            s_idx = task_idx / 2;
            mir = (task_idx % 2 == 1);
            envs[i].reset_game();
        }
        
        envs[i].set_seed(seeds[s_idx]);
        task_idx++;
        envs[i].reset_round();
        
        is_mir[i] = mir; return true;
    }

    Snap get_snap(const Env& e, int pt, int pi, int type) {
        uint64_t my_p = e.mgr.pile[pt];
        uint64_t sh = 0;
        if (type == 1 && !e.show.empty()) {
            sh = (1ULL << e.show[0]);
            my_p |= sh;
        }
        return { e.mgr.hand[pt], e.mgr.stock | e.mgr.hand[pi], my_p, e.mgr.field, e.mgr.pile[pi], e.mgr.dis_hist[pt], sh, e.mgr.play[pi], e.mgr.ign[pi], e.pt[pt], e.pt[pi], e.rnd, e.t16, e.koi_num(pt), e.koi_num(pi), e.dlr == pt, static_cast<uint8_t>(type) };
    }

    void set_mask(int i, bool* m, int t) {
        int sz = (t == 2) ? 2 : 48; std::memset(m, 0, sz * sizeof(bool));
        if (t == 2) { m[0] = true; m[1] = true; }
        else { uint64_t b = (t == 0) ? envs[i].mgr.hand[envs[i].turn_p()] : envs[i].mgr.pair; while (b) { m[ctz(b)] = true; b &= (b - 1); } }
    }

    void step_idle(int mx_rnd) {
        int f = 0, win1 = 0, win2 = 0, drw = 0;
        // Design Intent: GPU推論前の環境ステップ(CPU処理)を並列化しボトルネックを解消
        #pragma omp parallel for reduction(+:f, win1, win2, drw)
        for (int i = 0; i < n_env; ++i) {
            auto& e = envs[i];
            while (!e.needs_act() && e.st != GState::GAME_OVER) {
                if (e.st == GState::ROUND_OVER) {
                    e.pt[1] += e.rnd_pt(1); e.pt[2] += e.rnd_pt(2);
                    if (e.pt[1] <= 0 || e.pt[2] <= 0 || e.rnd == mx_rnd) {
                        bool p1_won = (e.pt[1] > e.pt[2]), p2_won = (e.pt[2] > e.pt[1]);
                        if (is_mir[i]) { if (p1_won) win2++; else if (p2_won) win1++; else drw++; } 
                        else { if (p1_won) win1++; else if (p2_won) win2++; else drw++; }
                        f++; 
                        bool assigned = false;
                        #pragma omp critical
                        { assigned = assign_task(i); }
                        if (!assigned) break; 
                    } else { e.rnd++; e.reset_round(); }
                } else e.step(-1);
            }
        }
        fin += f; w1 += win1; w2 += win2; draws += drw;
    }

    void sample_greedy(int t, int n, const float* l_ptr, const bool* m_ptr, int64_t* a_ptr) {
        int cols = (t == 2) ? 2 : 48;
        for (int i = 0; i < n; ++i) {
            int best = -1; float mx = -1e9f;
            for (int c = 0; c < cols; ++c) {
                if (m_ptr[i * cols + c] && l_ptr[i * cols + c] > mx) { mx = l_ptr[i * cols + c]; best = c; }
            }
            a_ptr[i] = (best != -1) ? best : 0; 
        }
    }

    void play(int mx_rnd) {
        std::array<std::vector<int>, 3> req1{}, req2{};
        float *cp[3] = {ct[0].data_ptr<float>(), ct[1].data_ptr<float>(), ct[2].data_ptr<float>()};
        float *gp[3] = {gt[0].data_ptr<float>(), gt[1].data_ptr<float>(), gt[2].data_ptr<float>()};
        bool  *mp[3] = {mk[0].data_ptr<bool>(),  mk[1].data_ptr<bool>(),  mk[2].data_ptr<bool>()};

        while (fin < tg) {
            step_idle(mx_rnd); 
            if (fin >= tg) break;
            for (int t = 0; t < 3; ++t) { req1[t].clear(); req2[t].clear(); }
            
            for (int i = 0; i < n_env; ++i) {
                if (envs[i].needs_act()) {
                    int p = envs[i].turn_p();
                    int t = (envs[i].st == GState::DISCARD) ? 0 : ((envs[i].st == GState::KOIKOI) ? 2 : 1);
                    bool is_m1 = (!is_mir[i] && p == 1) || (is_mir[i] && p == 2);
                    if (is_m1) req1[t].push_back(i); else req2[t].push_back(i);
                }
            }

            auto proc_mod = [&](torch::jit::Module& model, std::array<std::vector<int>, 3>& req) {
                torch::NoGradGuard ng;
                for (int t = 0; t < 3; ++t) {
                    int n = req[t].size(); if (n == 0) continue;
                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) { 
                        int ei = req[t][i];
                        write_feat(cp[t] + i * C_CH * 48, gp[t] + i * G_CH, get_snap(envs[ei], envs[ei].turn_p(), 3 - envs[ei].turn_p(), t), mx_rnd); 
                        set_mask(ei, mp[t] + i * ((t == 2) ? 2 : 48), t); 
                    }
                    auto out = model.forward({ct[t].slice(0, 0, n).to(dev, true), gt[t].slice(0, 0, n).to(dev, true)}).toTuple();
                    sample_greedy(t, n, out->elements()[(t == 2) ? 2 : 0].toTensor().cpu().data_ptr<float>(), mp[t], ab_.data());
                    for (int i = 0; i < n; ++i) envs[req[t][i]].step(ab_[i]);
                }
            };
            proc_mod(m1, req1); proc_mod(m2, req2);
        }
    }
};

py::dict run_arena(int tot, const std::string& p1_b, const std::string& p2_b, const std::string& dv, int mx, int s_rnd, int s_p1, int s_p2, int s_dlr) {
    torch::jit::Module p1, p2;
    torch::Device dev(dv);
    try {
        if (p1_b.empty()) {
            std::lock_guard<std::mutex> lk(g_box.mtx);
            if (!g_box.loaded) throw std::runtime_error("P1 Cache not loaded. Call reload_model first.");
            p1 = g_box.model;
        } else {
            p1 = torch::jit::load(std::istringstream(p1_b));
            p1.to(dev); p1.eval();
        }

        if (p2_b.empty()) {
            std::lock_guard<std::mutex> lk(a_box.mtx);
            if (!a_box.loaded) {
                std::lock_guard<std::mutex> lk_g(g_box.mtx);
                p2 = g_box.model;
            } else {
                p2 = a_box.opp_mod;
            }
        } else {
            std::lock_guard<std::mutex> lk(a_box.mtx);
            a_box.opp_mod = torch::jit::load(std::istringstream(p2_b));
            a_box.opp_mod.to(dev);
            a_box.opp_mod.eval();
            a_box.loaded = true;
            p2 = a_box.opp_mod;
        }
    } catch (const c10::Error& e) { throw std::runtime_error("JIT Load Error: " + std::string(e.what())); }
    
    int tg = (tot % 2 == 0) ? tot : tot + 1;
    ArenaSim sim(tg, tg, p1, p2, dv, s_rnd, s_p1, s_p2, s_dlr);
    { py::gil_scoped_release rel; sim.play(mx); }
    return py::dict("win"_a=sim.w1, "lose"_a=sim.w2, "draw"_a=sim.draws, "score"_a=0.0);
}

PYBIND11_MODULE(core, m) {
    m.def("calc_yaku_pt", &calc_yaku_pt);
    m.def("run_sim", [](int nth, int ne, int tg, int cd, int cp, int ck,
                        const py::bytes& mod_bytes_py,
                        const std::string& dv, int mx_rnd, float tmp, float eps, float nz, bool force_stop_koikoi, const py::array_t<float>& wpm_arr) {
        std::string mod_bytes = mod_bytes_py;
        
        auto buf = wpm_arr.request();
        if (buf.size != 2 * 9 * 61) throw std::runtime_error("wpm shape error: Expected [2, 9, 61]");
        std::vector<float> wpm_vec(2 * 9 * 61);
        std::memcpy(wpm_vec.data(), buf.ptr, 2 * 9 * 61 * sizeof(float));

        return run_sim(nth, ne, tg, cd, cp, ck, mod_bytes, dv, mx_rnd, tmp, eps, nz, force_stop_koikoi, wpm_vec);
    }, "nth"_a, "ne"_a, "tg"_a, "cd"_a, "cp"_a, "ck"_a, "mod_bytes_py"_a, "dv"_a, "mx_rnd"_a, "tmp"_a, "eps"_a, "nz"_a, "force_stop_koikoi"_a = false, "wpm_arr"_a);

   m.def("run_arena", [](int tot, const py::bytes& p1_bytes, const py::bytes& p2_bytes, const std::string& dv, int mx, int s_rnd, int s_p1, int s_p2, int s_dlr) {
        return run_arena(tot, std::string(p1_bytes), std::string(p2_bytes), dv, mx, s_rnd, s_p1, s_p2, s_dlr);
    }, "tot"_a, "p1_bytes"_a, "p2_bytes"_a, "dv"_a, "mx"_a, "s_rnd"_a=0, "s_p1"_a=30, "s_p2"_a=30, "s_dlr"_a=0);
    m.def("set_rules", &set_rules);
    m.def("close_sim", []() {});
    m.def("reload_model", [](const py::bytes& mod_bytes_py) {
        std::string mod_bytes = mod_bytes_py;
        std::istringstream stream(mod_bytes);
        std::lock_guard<std::mutex> lk(g_box.mtx);
        g_box.model = torch::jit::load(stream);
        g_box.model.eval();
        g_box.loaded = true;
    });

    py::class_<StateMgr>(m, "StateMgr")
        .def(py::init<>())
        .def("init", &StateMgr::init)
        .def("discard", &StateMgr::discard)
        .def("pick", &StateMgr::pick)
        .def("draw", &StateMgr::draw)
        .def("feat", &StateMgr::feat)
        .def("mask", &StateMgr::mask);

    py::enum_<GState>(m, "GState")
        .value("DISCARD", GState::DISCARD)
        .value("DISCARD_PICK", GState::DISCARD_PICK)
        .value("DRAW", GState::DRAW)
        .value("DRAW_PICK", GState::DRAW_PICK)
        .value("KOIKOI", GState::KOIKOI)
        .value("ROUND_OVER", GState::ROUND_OVER)
        .value("GAME_OVER", GState::GAME_OVER)
        .export_values();

    py::class_<Env>(m, "Env")
        .def(py::init<int>(), "seed"_a = 42)
        .def("reset_game", &Env::reset_game)
        .def("reset_round", &Env::reset_round)
        .def("step", &Env::step)
        .def("needs_act", &Env::needs_act)
        .def("turn_p", &Env::turn_p)
        .def("idle_p", &Env::idle_p)
        .def("turn8", &Env::turn8)
        .def("koi_num", &Env::koi_num)
        .def("rnd_pt", &Env::rnd_pt)
        .def_readonly("st", &Env::st)
        .def_readonly("wait", &Env::wait)
        .def_readwrite("win", &Env::win)
        .def_readwrite("dlr", &Env::dlr)
        .def_readonly("t16", &Env::t16)
        .def_readonly("ex", &Env::ex)
        .def_readonly("mgr", &Env::mgr)
        .def("get_hand", [](const Env& e, int p) { return std::vector<int>(e.hand[p].begin(), e.hand[p].end()); })
        .def("get_pile", [](const Env& e, int p) { return std::vector<int>(e.pile[p].begin(), e.pile[p].end()); })
        .def("get_field", [](const Env& e) { return std::vector<int>(e.field.begin(), e.field.end()); })
        .def("get_show", [](const Env& e) { return std::vector<int>(e.show.begin(), e.show.end()); });

    m.attr("MAX_RND") = 6;
}