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

constexpr int C_CH = 26; // カード特徴チャネル数
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
    uint64_t hand, unseen, my_pile, field, op_pile, my_dis;
    uint16_t op_play, op_ign;
    int pt_t, pt_i, rnd, t16, koi_t, koi_i;
    bool is_dl;
    uint8_t st_type;
};

struct Slot {
    int st_type, act;
    float old_val, old_prb;
    float old_log[48];
    bool mask[48];
    Snap snap;
};

struct ExpPars { float temp, eps, noise; };

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
        if (sn.field & m_bb) std::fill_n(c_dst + 6 * 48 + m * 4, 4, 1.0f);
    }

    auto set_pot = [&](int r, uint64_t mk, int req, int lim, uint64_t sf, uint64_t op, bool is_fixed = false) {
    if (popcnt(op & mk) < lim) {
        int my_cnt = popcnt(sf & mk);
        // 【追加】上限固定役で、すでに必要枚数に達していたら「0」のまま抜ける
        if (is_fixed && my_cnt >= req) return; 
        
        set_bb(r, mk, std::min(1.0f, static_cast<float>(my_cnt + 1) / req));
    }
};

    set_pot(7,  M.lt,     5,  3,  sn.my_pile, sn.op_pile, false); // 光(五光が天井だが発展型)
    set_pot(8,  M.bdb,    3,  1,  sn.my_pile, sn.op_pile, true);  // 猪鹿蝶【固定】
    set_pot(9,  M.r_rib,  3,  1,  sn.my_pile, sn.op_pile, true);  // 赤短【固定】
    set_pot(10, M.b_rib,  3,  1,  sn.my_pile, sn.op_pile, true);  // 青短【固定】
    set_pot(11, M.f_sake, 2,  1,  sn.my_pile, sn.op_pile, true);  // 花見【固定】
    set_pot(12, M.m_sake, 2,  1,  sn.my_pile, sn.op_pile, true);  // 月見【固定】
    set_pot(13, M.sd,     5,  5,  sn.my_pile, sn.op_pile, false); // タネ
    set_pot(14, M.rb,     5,  6,  sn.my_pile, sn.op_pile, false); // タン
    set_pot(15, M.ks,     10, 16, sn.my_pile, sn.op_pile, false); // カス

    for (int m = 0; m < 12; ++m) {
        bool played  = (sn.op_play >> m) & 1;
        bool ignored = (sn.op_ign  >> m) & 1;

        // 「1枚も出しておらず、かつ場札をスルーした」場合のみ、確実に持っていない(-1.0)
        if (ignored && !played) {
            std::fill_n(c_dst + 16 * 48 + m * 4, 4, -1.0f);
        }
        // それ以外（0,0 / 1,0 / 1,1）はすべて 0.0f（初期値のまま）
    }

    set_pot(17, M.lt,     5,  3, sn.op_pile, sn.my_pile, false);
    set_pot(18, M.bdb,    3,  1, sn.op_pile, sn.my_pile, true);
    set_pot(19, M.r_rib,  3,  1, sn.op_pile, sn.my_pile, true);
    set_pot(20, M.b_rib,  3,  1, sn.op_pile, sn.my_pile, true);
    set_pot(21, M.f_sake, 2,  1, sn.op_pile, sn.my_pile, true);
    set_pot(22, M.m_sake, 2,  1, sn.op_pile, sn.my_pile, true);
    set_pot(23, M.sd,     5,  5, sn.op_pile, sn.my_pile, false);
    set_pot(24, M.rb,     5,  6, sn.op_pile, sn.my_pile, false);
    set_pot(25, M.ks,     10, 16, sn.op_pile, sn.my_pile, false);

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
        auto c_arr = py::array_t<float>({26, 48});
        auto g_arr = py::array_t<float>({7});
        Snap sn{hand[tp], stock | hand[ip], pile[tp], field, pile[ip], dis_hist[tp], play[ip], ign[ip], pt_t, pt_i, rnd, t16, koi_t, koi_i, is_dl, static_cast<uint8_t>(is_koi ? 2 : 0)};
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
    torch::Tensor c_st, g_st, act, rew, old_log, old_val, mask, old_prb;
    float *cp, *gp, *rp, *lp, *vp, *pp; int64_t* ap; bool* mp;

    explicit TraceBuf(int c) : cap(c) {
        auto opt_f = torch::TensorOptions().dtype(torch::kFloat32);
        c_st = torch::empty({cap, C_CH, 48}, opt_f); g_st = torch::empty({cap, G_CH}, opt_f);
        act = torch::empty({cap}, torch::kInt64); rew = torch::empty({cap}, opt_f);
        old_log = torch::empty({cap, 48}, opt_f); old_val = torch::empty({cap}, opt_f);
        mask = torch::empty({cap, 48}, torch::kBool); old_prb = torch::empty({cap}, opt_f);

        cp = c_st.data_ptr<float>(); gp = g_st.data_ptr<float>(); ap = act.data_ptr<int64_t>();
        rp = rew.data_ptr<float>(); lp = old_log.data_ptr<float>(); vp = old_val.data_ptr<float>();
        mp = mask.data_ptr<bool>(); pp = old_prb.data_ptr<float>();
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
        ap[ix] = sl.act; rp[ix] = ret; vp[ix] = sl.old_val; pp[ix] = sl.old_prb;
        std::memcpy(lp + ix * 48, sl.old_log, 48 * sizeof(float));
        std::memcpy(mp + ix * 48, sl.mask, 48 * sizeof(bool));
        write_feat(cp + ix * C_CH * 48, gp + ix * G_CH, sl.snap, mx_rnd);
    }

    py::dict fin() {
        int s = std::min(cap, sz);
        return py::dict("card_states"_a=c_st.slice(0,0,s).clone(), "global_states"_a=g_st.slice(0,0,s).clone(),
                        "actions"_a=act.slice(0,0,s).clone(), "rewards"_a=rew.slice(0,0,s).clone(),
                        "old_logits"_a=old_log.slice(0,0,s).clone(), "old_values"_a=old_val.slice(0,0,s).clone(),
                        "legal_masks"_a=mask.slice(0,0,s).clone(), "old_action_probs"_a=old_prb.slice(0,0,s).clone());
    }
};

enum class GState : uint8_t { DISCARD, DISCARD_PICK, DRAW, DRAW_PICK, KOIKOI, ROUND_OVER, GAME_OVER };

class Env {
public:
    int rnd = 1, dlr = 1, win = 1, t16 = 1, t_pt = 0, pt[3]{}, koi[3][8]{};
    bool ex = false, wait = false;
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
    int n_env, tg_games, fin = 0; float disc, lam;
    std::vector<Env> envs; std::vector<SVec<Slot, 512>> tr[3];
    TraceBuf b_dis, b_pic, b_koi; torch::Device dev;
    torch::Tensor ct[3], gt[3], mk[3]; ExpPars ep;
private:
    std::vector<int64_t> ab_; std::vector<float> pb_, vb_;

public:
    BatchSim(int ne, int tg, int cd, int cp, int ck, float dc, float lm, const std::string& dv, ExpPars e)
        : n_env(ne), tg_games(tg), disc(dc), lam(lm), b_dis(cd), b_pic(cp), b_koi(ck), dev(dv), ep(e), ab_(ne), pb_(ne), vb_(ne) {
        envs.reserve(ne); std::random_device rd;
        for (int i=0;i<ne;++i) envs.emplace_back(rd());
        for (int p=1;p<=2;++p) tr[p].resize(ne);
        auto of = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        auto ob = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);
        for (int i=0;i<3;++i) { ct[i] = torch::empty({ne, C_CH, 48}, of); gt[i] = torch::empty({ne, G_CH}, of); }
        mk[0] = torch::empty({ne, 48}, ob); mk[1] = torch::empty({ne, 48}, ob); mk[2] = torch::empty({ne, 2}, ob);
    }

    void reset() {
        fin = 0; for (auto& e : envs) { e.reset_game(); e.reset_round(); }
        for (int p=1;p<=2;++p) for (auto& v : tr[p]) v.clear();
        b_dis.clear(); b_pic.clear(); b_koi.clear();
    }

    py::dict fin_bufs() { return py::dict("discard"_a=b_dis.fin(), "pick"_a=b_pic.fin(), "koikoi"_a=b_koi.fin()); }

    Snap get_snap(const Env& e, int pt, int pi, int type) {
        uint64_t my_p = e.mgr.pile[pt];

        // 拾い選択フェーズ(type == 1)のとき、
        // 「獲得することが100%確定しているトリガー札(show)」を自身の獲得札に先行合流させる
        if (type == 1 && !e.show.empty()) {
            my_p |= (1ULL << e.show[0]);
        }

        return {
            e.mgr.hand[pt],
            e.mgr.stock | e.mgr.hand[pi],
            my_p,                          // <-- 自身の取札に先行合流
            e.mgr.field,                   // 場札は純粋に「選択肢となる2枚」だけが残る
            e.mgr.pile[pi],
            e.mgr.dis_hist[pt], 
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
        if (t == 2) { m[0] = true; m[1] = true; }
        else { uint64_t b = (t == 0) ? envs[i].mgr.hand[envs[i].turn_p()] : envs[i].mgr.pair; while (b) { m[ctz(b)] = true; b &= (b - 1); } }
    }

    void end_game(int i, int mx_rnd) {
        auto& e = envs[i]; int w = (e.pt[1] > e.pt[2]) ? 1 : ((e.pt[2] > e.pt[1]) ? 2 : 0);
        for (int p=1;p<=2;++p) {
            auto& t = tr[p][i]; if (t.empty()) continue;
            float r = (w == p) ? 1.0f : (w != 0 ? -1.0f : 0.0f);
            int T = t.size(); std::array<float, 512> ret{}; float ga = 0.0f;
            for (int k = T - 1; k >= 0; --k) {
                float v = t[k].old_val, vn = (k == T - 1) ? 0.0f : t[k + 1].old_val;
                ga = ((k == T - 1 ? r : 0.0f) + disc * vn - v) + (disc * lam) * ga; ret[k] = ga + v;
            }
            for (int k=0;k<T;++k) (t[k].st_type == 0 ? b_dis : (t[k].st_type == 1 ? b_pic : b_koi)).push(t[k], ret[k], mx_rnd);
            t.clear();
        }
    }

    void step_idle(int mx_rnd) {
        int f = 0;
        #pragma omp parallel for reduction(+:f)
        for (int i = 0; i < n_env; ++i) {
            auto& e = envs[i];
            while (!e.needs_act() && e.st != GState::GAME_OVER) {
                if (e.st == GState::ROUND_OVER) {
                    e.pt[1] += e.rnd_pt(1); e.pt[2] += e.rnd_pt(2);
                    if (e.pt[1] <= 0 || e.pt[2] <= 0 || e.rnd == mx_rnd) { end_game(i, mx_rnd); f++; e.reset_game(); e.reset_round(); }
                    else { e.rnd++; e.reset_round(); }
                } else e.step(-1);
            }
        }
        fin += f;
    }

    void sample_neurd(int t, int n, const int* ix_list, const float* l_ptr, const float* q_ptr, const bool* m_ptr, int64_t* a_ptr, float* p_ptr, float* v_ptr) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f); 
        int cols = (t == 2) ? 2 : 48;
        
        for (int i = 0; i < n; ++i) {
            SVec<int, 48> leg; 
            for (int c = 0; c < cols; ++c) {
                if (m_ptr[i * cols + c]) leg.push(c);
            }
            if (leg.empty()) { a_ptr[i] = 0; p_ptr[i] = 1.0f; v_ptr[i] = 0.0f; continue; }

            // 1. ベースとなるネットワークの方策確率 (Greedy または Softmax) を計算
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

            // 2. Dirichletノイズの適用 (合法手が複数ある場合のみ)
            if (ep.noise > 0.0f && leg.size() > 1) {
                // Design Intent: alpha=0.3はチェスや将棋などのAlphaZero系でよく使われる標準的な分散パラメータ
                float alpha = 0.3f; 
                std::gamma_distribution<float> gamma(alpha, 1.0f);
                std::array<float, 48> dir{};
                float sum_dir = 0.0f;
                
                // ガンマ分布からサンプリングして合計で割ることでディリクレ分布を生成
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

            // 3. ε-greedy (一様ランダム探索) の適用
            std::array<float, 48> mu{};
            float uni = ep.eps / leg.size();
            float w_pi = 1.0f - ep.eps;
            for (int a : leg) {
                mu[a] = uni + w_pi * base_pi[a];
            }

            // 4. 最終的な確率分布 `mu` に従ってサンプリング
            float r = dist(envs[ix_list[i]].rng), acc = 0.0f; 
            int ch = leg.back(); // フォールバック
            for (int a : leg) { 
                acc += mu[a]; 
                if (r <= acc) { ch = a; break; } 
            }
            
            a_ptr[i] = ch; 
            p_ptr[i] = mu[ch]; // Design Intent: NeuRDのアドバンテージ補正に使うため、実際にサンプリングされた確率を保存

            // 5. 価値(Q)の期待値を計算
            float vs = 0.0f; 
            for (int a : leg) vs += mu[a] * q_ptr[i * cols + a];
            v_ptr[i] = vs;
        }
    }

    void play(int mx_rnd) {
        std::array<std::vector<int>, 3> req{};
        float *cp[3] = {ct[0].data_ptr<float>(), ct[1].data_ptr<float>(), ct[2].data_ptr<float>()};
        float *gp[3] = {gt[0].data_ptr<float>(), gt[1].data_ptr<float>(), gt[2].data_ptr<float>()};
        bool  *mp[3] = {mk[0].data_ptr<bool>(),  mk[1].data_ptr<bool>(),  mk[2].data_ptr<bool>()};

        while (fin < tg_games) {
            step_idle(mx_rnd); if (fin >= tg_games) break;
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

                    sample_neurd(t, n, req[t].data(), l_cpu.data_ptr<float>(), q_cpu.data_ptr<float>(), mp[t], ab_.data(), pb_.data(), vb_.data());

                    #pragma omp parallel for
                    for (int i=0;i<n;++i) {
                        int ei = req[t][i]; Slot sl{t, static_cast<int>(ab_[i]), vb_[i], pb_[i]};
                        int cols = (t == 2) ? 2 : 48; std::memset(sl.old_log, 0, 48 * sizeof(float)); std::memset(sl.mask, 0, 48 * sizeof(bool));
                        for (int c=0;c<cols;++c) { sl.old_log[c] = l_cpu.data_ptr<float>()[i * cols + c]; sl.mask[c] = mp[t][i * cols + c]; }
                        sl.snap = get_snap(envs[ei], p, 3 - p, t); tr[p][ei].push_back(sl); envs[ei].step(sl.act);
                    }
                }
            }
        }
    }
};

py::list run_sim(int nth, int ne, int tg, int cd, int cp, int ck, float dc, const std::string& mod_bytes, const std::string& dv, int mx_rnd, float tmp, float eps, float nz, float lam) {
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
    for(int i=0;i<nth;++i) s[i] = std::make_unique<BatchSim>(ne, tg, cd, cp, ck, dc, lam, dv, ep);
    { py::gil_scoped_release rel;
        #pragma omp parallel for schedule(dynamic)
        for (int i=0;i<nth;++i) s[i]->play(mx_rnd); }
    py::list res; for (int i=0;i<nth;++i) res.append(s[i]->fin_bufs()); return res;
}

class ArenaSim {
public:
    int n_env, tg_games, task_idx = 0;
    int p1_wins = 0, p2_wins = 0, draws = 0, fin = 0;
    std::vector<Env> envs;
    std::vector<bool> is_mirror_env;
    std::vector<unsigned int> seeds;
    torch::jit::Module m1, m2;
    torch::Device dev;
    torch::Tensor ct[3], gt[3], mk[3];
    std::vector<int64_t> ab_;

    ArenaSim(int ne, int tg, torch::jit::Module p1, torch::jit::Module p2, const std::string& dv)
        : n_env(ne), tg_games(tg), m1(p1), m2(p2), dev(dv), ab_(ne) {
        
        envs.reserve(ne);
        is_mirror_env.resize(ne, false);
        
        // Design Intent: 指定ゲーム数をペア(2試合1組)で分割し、マスターシードを生成
        int num_pairs = tg_games / 2;
        seeds.resize(num_pairs);
        std::random_device rd;
        for (int i = 0; i < num_pairs; ++i) {
            seeds[i] = rd();
        }

        auto of = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
        auto ob = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);
        for (int i = 0; i < 3; ++i) { 
            ct[i] = torch::empty({ne, C_CH, 48}, of); 
            gt[i] = torch::empty({ne, G_CH}, of); 
        }
        mk[0] = torch::empty({ne, 48}, ob); 
        mk[1] = torch::empty({ne, 48}, ob); 
        mk[2] = torch::empty({ne, 2}, ob);

        for (int i = 0; i < ne; ++i) {
            envs.emplace_back(42); 
            assign_task(i);
        }
    }

    // 環境が空いた際に、共有タスクキューから次のシードと担当陣営(ノーマルorミラー)を割り当てる
    bool assign_task(int env_idx) {
        if (task_idx >= tg_games) {
            envs[env_idx].st = GState::GAME_OVER; // 全タスク完了で待機状態へ
            return false;
        }
        int pair_idx = task_idx / 2;
        bool is_mirror = (task_idx % 2 == 1); // 奇数番目のタスクはミラーマッチ(モデル反転)
        task_idx++;
        
        envs[env_idx].set_seed(seeds[pair_idx]); // ペアで全く同じシードを注入
        envs[env_idx].reset_game();
        envs[env_idx].reset_round();
        is_mirror_env[env_idx] = is_mirror;
        return true;
    }

    Snap get_snap(const Env& e, int pt, int pi, int type) {
        uint64_t my_p = e.mgr.pile[pt];
        if (type == 1 && !e.show.empty()) my_p |= (1ULL << e.show[0]);
        return {
            e.mgr.hand[pt], e.mgr.stock | e.mgr.hand[pi], my_p, e.mgr.field, e.mgr.pile[pi],
            e.mgr.dis_hist[pt], e.mgr.play[pi], e.mgr.ign[pi], e.pt[pt], e.pt[pi], e.rnd, e.t16, 
            e.koi_num(pt), e.koi_num(pi), e.dlr == pt, static_cast<uint8_t>(type)
        };
    }

    void set_mask(int i, bool* m, int t) {
        int sz = (t == 2) ? 2 : 48; 
        std::memset(m, 0, sz * sizeof(bool));
        if (t == 2) { m[0] = true; m[1] = true; }
        else { 
            uint64_t b = (t == 0) ? envs[i].mgr.hand[envs[i].turn_p()] : envs[i].mgr.pair; 
            while (b) { m[ctz(b)] = true; b &= (b - 1); } 
        }
    }

    void step_idle(int mx_rnd) {
        for (int i = 0; i < n_env; ++i) {
            auto& e = envs[i];
            while (!e.needs_act() && e.st != GState::GAME_OVER) {
                if (e.st == GState::ROUND_OVER) {
                    e.pt[1] += e.rnd_pt(1); 
                    e.pt[2] += e.rnd_pt(2);
                    if (e.pt[1] <= 0 || e.pt[2] <= 0 || e.rnd == mx_rnd) {
                        bool p1_won = (e.pt[1] > e.pt[2]);
                        bool p2_won = (e.pt[2] > e.pt[1]);
                        
                        // Design Intent: ミラー対戦時はゲーム内のP1が外部のModel 2に相当するため勝敗を反転して集計
                        if (is_mirror_env[i]) {
                            if (p1_won) p2_wins++; 
                            else if (p2_won) p1_wins++;
                            else draws++;
                        } else {
                            if (p1_won) p1_wins++;
                            else if (p2_won) p2_wins++;
                            else draws++;
                        }
                        
                        fin++; 
                        assign_task(i); 
                    } else { 
                        e.rnd++; e.reset_round(); 
                    }
                } else {
                    e.step(-1);
                }
            }
        }
    }

    void sample_greedy(int t, int n, const float* l_ptr, const bool* m_ptr, int64_t* a_ptr) {
        int cols = (t == 2) ? 2 : 48;
        for (int i = 0; i < n; ++i) {
            int best_a = -1; float mx = -1e9f;
            for (int c = 0; c < cols; ++c) {
                if (m_ptr[i * cols + c] && l_ptr[i * cols + c] > mx) {
                    mx = l_ptr[i * cols + c]; 
                    best_a = c;
                }
            }
            a_ptr[i] = (best_a != -1) ? best_a : 0; 
        }
    }

    void play(int mx_rnd) {
        // Design Intent: ターンプレイヤーとミラー状態を掛け合わせ、モデル毎の推論バッチを分割する
        std::array<std::vector<int>, 3> req_m1{};
        std::array<std::vector<int>, 3> req_m2{};
        float *cp[3] = {ct[0].data_ptr<float>(), ct[1].data_ptr<float>(), ct[2].data_ptr<float>()};
        float *gp[3] = {gt[0].data_ptr<float>(), gt[1].data_ptr<float>(), gt[2].data_ptr<float>()};
        bool  *mp[3] = {mk[0].data_ptr<bool>(),  mk[1].data_ptr<bool>(),  mk[2].data_ptr<bool>()};

        while (fin < tg_games) {
            step_idle(mx_rnd); 
            if (fin >= tg_games) break;
            
            for (int t = 0; t < 3; ++t) {
                req_m1[t].clear();
                req_m2[t].clear();
            }
            
            for (int i = 0; i < n_env; ++i) {
                if (envs[i].needs_act()) {
                    int p = envs[i].turn_p();
                    int t = (envs[i].st == GState::DISCARD) ? 0 : ((envs[i].st == GState::KOIKOI) ? 2 : 1);
                    
                    // ノーマル環境ならP1がm1、ミラー環境ならP2がm1
                    bool is_m1 = (!is_mirror_env[i] && p == 1) || (is_mirror_env[i] && p == 2);
                    if (is_m1) req_m1[t].push_back(i);
                    else       req_m2[t].push_back(i);
                }
            }

            auto process_model = [&](torch::jit::Module& model, std::array<std::vector<int>, 3>& req) {
                torch::NoGradGuard ng;
                for (int t = 0; t < 3; ++t) {
                    int n = req[t].size(); 
                    if (n == 0) continue;
                    
                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) { 
                        int ei = req[t][i];
                        int p = envs[ei].turn_p();
                        write_feat(cp[t] + i * C_CH * 48, gp[t] + i * G_CH, get_snap(envs[ei], p, 3 - p, t), mx_rnd); 
                        set_mask(ei, mp[t] + i * ((t == 2) ? 2 : 48), t); 
                    }

                    auto out = model.forward({ct[t].slice(0, 0, n).to(dev, true), gt[t].slice(0, 0, n).to(dev, true)}).toTuple();
                    torch::Tensor l_cpu = out->elements()[(t == 2) ? 2 : 0].toTensor().cpu(); 
                    sample_greedy(t, n, l_cpu.data_ptr<float>(), mp[t], ab_.data());

                    for (int i = 0; i < n; ++i) envs[req[t][i]].step(ab_[i]);
                }
            };

            process_model(m1, req_m1);
            process_model(m2, req_m2);
        }
    }
};

py::dict run_arena(int tot, const std::string& p1_bytes, const std::string& p2_bytes, const std::string& dv, int mx) {
    torch::jit::Module p1, p2;
    try {
        std::istringstream p1_stream(p1_bytes);
        p1 = torch::jit::load(p1_stream);
        p1.eval();

        std::istringstream p2_stream(p2_bytes);
        p2 = torch::jit::load(p2_stream);
        p2.eval();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading JIT module from bytes: " + std::string(e.what()));
    }
    
    // Design Intent: デュプリケート方式を成立させるため、指定ゲーム数を必ず偶数に丸める
    int tg_games = (tot % 2 == 0) ? tot : tot + 1;
    
    // 同時実行環境数はテスト総数または上限(例:32)の小さい方を採用
    int n_env = std::min(32, tg_games); 
    ArenaSim sim(n_env, tg_games, p1, p2, dv);
    
    { 
        py::gil_scoped_release rel; 
        sim.play(mx); 
    }
    
    return py::dict("win"_a=sim.p1_wins, "lose"_a=sim.p2_wins, "draw"_a=sim.draws, "score"_a=0.0);
}

PYBIND11_MODULE(core, m) {
    m.def("calc_yaku_pt", &calc_yaku_pt);
    m.def("run_sim", [](int nth, int ne, int tg, int cd, int cp, int ck, float dc,
                        const py::bytes& mod_bytes_py,
                        const std::string& dv, int mx_rnd, float tmp, float eps, float nz, float lam) {
        std::string mod_bytes = mod_bytes_py; // py::bytes -> std::stringへ自動キャスト
        return run_sim(nth, ne, tg, cd, cp, ck, dc, mod_bytes, dv, mx_rnd, tmp, eps, nz, lam);
    });

    m.def("run_arena", [](int tot, const py::bytes& p1_bytes, const py::bytes& p2_bytes, const std::string& dv, int mx) {
        return run_arena(tot, std::string(p1_bytes), std::string(p2_bytes), dv, mx);
    });
    m.def("set_rules", &set_rules);
    m.def("close_sim", []() {});
    m.def("reload_model", [](const py::bytes& mod_bytes_py) {
        std::string mod_bytes = mod_bytes_py;
        std::istringstream stream(mod_bytes);
        std::lock_guard<std::mutex> lk(g_box.mtx);
        g_box.model = torch::jit::load(stream);
        g_box.model.eval();
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