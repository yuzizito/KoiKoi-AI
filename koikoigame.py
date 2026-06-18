#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import json
import time
import numpy as np
import torch
import koikoicore

class DefaultVar():
    DEFAULT_ROUND_TOTAL = 8
    DEFAULT_INIT_POINT = 30

class KoiKoiRoundStateBase():
    def __init__(self, dealer=None):
        assert dealer in [1,2,None]
        self.hand = {1:[], 2:[]}
        self.pile = {1:[], 2:[]}
        self.field_slot = []
        self.stock = []
        self.show = []
        self.collect = []
        self.turn_16 = 1
        self.dealer = random.randint(1,2) if dealer==None else dealer
        self.koikoi = {1:[0,0,0,0,0,0,0,0], 2:[0,0,0,0,0,0,0,0]}
        self.winner = None
        self.exhausted = False
        self.log = {}
        self.silence = True
        self.turn_point = 0
        self.state = 'init'
        self.wait_action = False        
        self.__deal_card()
    
    def new_round(self):
        self.__init__(dealer=self.winner)
        return
    
    @property
    def turn_player(self):
        return 1 if (self.turn_16+self.dealer)%2==0 else 2
    
    @property
    def idle_player(self):
        return 3-self.turn_player
    
    @property
    def turn_8(self):
        return (self.turn_16+1)//2
    
    @property
    def field(self):
        return sorted([slot for slot in self.field_slot if slot != -1])
    
    @property
    def unseen_card(self):
        return {1:(self.stock+self.hand[2]), 2:(self.stock+self.hand[1])}
    
    @property
    def pairing_card(self):
        if not self.show:
            return []
        target_suit = self.show[0] // 4
        return [card for card in self.field if card // 4 == target_suit]
    
    @property
    def field_collect(self):
        collect_card = self.collect.copy()
        if self.show[0] in collect_card:
            collect_card.remove(self.show[0])
        return collect_card
    
    @property
    def koikoi_num(self):
        return {1:sum(self.koikoi[1]), 2:sum(self.koikoi[2])}
    
    @property
    def round_over(self):
        return self.state == 'round-over'
    
    @property
    def round_point(self):
        if self.winner == None:
            return {1:None, 2:None}
        elif self.exhausted:
            return {1:1, 2:-1} if self.dealer==1 else {1:-1, 2:1}
        elif self.winner == 1:
            return {1:self.yaku_point(1), 2:-self.yaku_point(1)}
        else:
            return {1:-self.yaku_point(2), 2:self.yaku_point(2)}
    
    def __deal_card(self):
        while True:
            # 0〜47のIDをシャッフル
            cards = list(range(48))
            random.shuffle(cards)
            
            self.hand[1] = sorted(cards[0:8])
            self.hand[2] = sorted(cards[8:16])
            
            # field_slot は [0,0] の代わりに -1 を使用
            self.field_slot = sorted(cards[16:24]) + [-1 for _ in range(10)]
            self.stock = cards[24:]
            
            flag = True
            # 同月4枚のチェック (card_id // 4 が同じものが4枚あるか)
            for suit_idx in range(12):
                counts = [
                    sum(1 for c in self.hand[1] if c // 4 == suit_idx),
                    sum(1 for c in self.hand[2] if c // 4 == suit_idx),
                    sum(1 for c in self.field if c // 4 == suit_idx)
                ]
                if 4 in counts:
                    flag = False        
            if flag:
                break
        self.__write_log()   
        self.state = 'discard'
        self.wait_action = True
        return
    
    def __collect_card(self,card):
        if len(self.pairing_card) == 0:
            self.collect = []
            self.field_slot[self.field_slot.index(-1)] = self.show[0]
        elif len(self.pairing_card) in [1,3]:
            self.collect = self.show + self.pairing_card
            for paired_card in self.pairing_card:
                self.field_slot[self.field_slot.index(paired_card)] = -1
            self.pile[self.turn_player].extend(self.collect)
        else:
            self.collect = self.show + [card]
            self.field_slot[self.field_slot.index(card)] = -1
            self.pile[self.turn_player].extend(self.collect)
        return
    
    def discard(self, card=None):        
        self.turn_point = self.yaku_point(self.turn_player)
        ind = self.hand[self.turn_player].index(card)
        self.show = [self.hand[self.turn_player].pop(ind)]
        self.__write_log()
        self.state = 'discard-pick'
        self.wait_action = len(self.pairing_card) == 2
        return self.state
        
    def discard_pick(self, card=None):
        self.__collect_card(card)
        self.__write_log()
        self.state = 'draw'
        self.wait_action = False
        return self.state
        
    def draw(self, card=None):
        self.show = [self.stock.pop()]
        self.__write_log()
        self.state = 'draw-pick'
        self.wait_action = len(self.pairing_card) == 2
        return self.state
        
    def draw_pick(self, card=None):
        self.__collect_card(card)
        self.__write_log()
        self.state = 'koikoi'
        self.wait_action = (self.yaku_point(self.turn_player) > self.turn_point) and (self.turn_8 < 8)
        return self.state
    
    def claim_koikoi(self, is_koikoi=None):
        if (self.yaku_point(self.turn_player) > self.turn_point) and (self.turn_8 == 8):
            is_koikoi = False
        self.koikoi[self.turn_player][self.turn_8-1] = int(is_koikoi==True)
        self.__write_log(is_koikoi)
        
        if is_koikoi == False:
            self.state = 'round-over'
            self.wait_action = False
            self.winner = self.turn_player
            self.__write_log()
        elif self.turn_16 == 16:
            self.state = 'round-over'
            self.wait_action = False
            self.exhausted = True
            self.winner = self.dealer
            self.__write_log()
        else:
            self.turn_16 += 1
            self.state = 'discard'
            self.wait_action = True
        return self.state

    def yaku(self, player):
        # C++側の高速なビットボード役判定ロジックを呼び出して、GUI用のリストを返す
        bb = 0
        for c in self.pile[player]:
            bb |= (1 << c)
        return koikoicore.evaluate_yaku_by_bitboard(bb, self.koikoi_num[player])
    
    def yaku_point(self, player):
        # Python側で整数のリストから64bit整数のビットボードを生成
        bb = 0
        for c in self.pile[player]:
            bb |= (1 << c)
        return koikoicore.get_yaku_point_by_bitboard(bb, self.koikoi_num[player])
    
    def __write_log(self, content=None):
        turn = str(self.turn_16)
        if self.state == 'init':
            self.log['basic'] = {'initBoard': self.field.copy()}
        elif self.state == 'discard':
            self.log['turn'+turn] = {'pairCard': self.pairing_card.copy()}
        elif self.state == 'draw':
            self.log['turn'+turn]['pairCard2'] = self.pairing_card.copy()
        return
        
class KoiKoiGameStateBase():
    def __init__(self, round_num=1, round_total=DefaultVar.DEFAULT_ROUND_TOTAL,
                 init_point=[DefaultVar.DEFAULT_INIT_POINT,DefaultVar.DEFAULT_INIT_POINT],
                 init_dealer=None, player_name=['Player1','Player2'], 
                 record_path='', save_record=False):
        
        self.round_total = round_total
        self.init_point = init_point
        self.init_dealer = init_dealer        
        self.player_name = {1:player_name[0], 2:player_name[1]}
        self.record_path = record_path
        self.save_record = save_record
        
        self.round_state = KoiKoiRoundState(dealer=self.init_dealer)
        self.round = round_num
        self.point = {1:init_point[0],2:init_point[1]}
        self.game_over = False
        self.winner = None
        self.log = {}
        self.__init_record()
        
    def new_game(self):
        self.__init__(round_num=1, round_total=self.round_total, 
                      init_point=self.init_point, init_dealer=self.init_dealer,
                      player_name=[self.player_name[1],self.player_name[2]], 
                      record_path=self.record_path, save_record=self.save_record)
        return
        
    def new_round(self):
        assert self.round_state.state == 'round-over'
        self.point[1] = self.point[1]+self.round_state.round_point[1]
        self.point[2] = self.point[2]+self.round_state.round_point[2]
        self.__round_result_record()
        if self.point[1]<=0 or self.point[2]<=0 or self.round==self.round_total:
            self.game_over = True
            self.winner = 1 if self.point[1]>self.point[2] else (2 if self.point[1]<self.point[2] else 0)
            self.__game_result_record()
        else:
            self.round_state.new_round()
            self.round += 1
        return
    
    def __init_record(self):
        self.log['info'] = {'startTime':time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime()), 
                            'endTime':None,
                            'player1Name':self.player_name[1],
                            'player2Name':self.player_name[2],
                            'player1InitPts':self.point[1], 
                            'player2InitPts':self.point[2], 
                            'numRound':self.round_total}
        self.log['result'] = {'isOver':False, 'gameWinner':None, 
                              'player1EndPts':None, "player2EndPts":None}
        self.log['save'] = {}
        self.log['record'] = {}
        return
    
    def __round_result_record(self):
        # GUI (koikoigui.py) が要求する 'basic' と 各種得点キーを明示的に補完して保存する
        round_log = self.round_state.log.copy()
        
        # 'basic' ディクショナリがない、または必要な得点キーがない場合に安全に作成
        if 'basic' not in round_log:
            round_log['basic'] = {}
        else:
            round_log['basic'] = round_log['basic'].copy()
            
        # C++側から計算された最新のラウンド得点をGUI用にログへ格納
        round_log['basic']['player1RoundPts'] = self.round_state.round_point[1]
        round_log['basic']['player2RoundPts'] = self.round_state.round_point[2]
        
        # 元のログ互換用のキー（Dealerなど）も未定義なら補完
        if 'Dealer' not in round_log['basic']:
            round_log['basic']['Dealer'] = self.round_state.dealer
            
        self.log['record']['round'+str(self.round)] = round_log
        return
    
    def __game_result_record(self):
        self.log['info']['endTime'] = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
        self.log['result'] = {'isOver':True, 'gameWinner':self.winner,
                              'player1EndPts':self.point[1], "player2EndPts":self.point[2]}
        if self.save_record == True:
            self.__save_record()
        return
    
    def __save_record(self):
        filename = self.record_path + self.log['info']['startTime'] + ' ' \
            + self.log['info']['player1Name'] + ' vs ' + self.log['info']['player2Name'] +'.json'
        with open(filename, 'w') as f:
            json.dump(self.log, f)
        return
       
class KoiKoiRoundState(KoiKoiRoundStateBase):
    _cached_card_key = None
    _card_suit_array = None
    
    @classmethod
    def _init_caches(cls):
        if cls._cached_card_key is None:
            card_dict = {
                'Crane':{(1,1)}, 'Curtain':{(3,1)}, 'Moon':{(8,1)},
                'Rainman':{(11,1)}, 'Phoenix':{(12,1)}, 'Sake':{(9,1)},
                'BoarDeerButterfly':{(6,1),(7,1),(10,1)}, 'Seed':{(2,1),(4,1),(5,1),(6,1),(7,1),(8,2),(9,1),(10,1),(11,2)},
                'RedRibbon':{(1,2),(2,2),(3,2)}, 'BlueRibbon':{(6,2),(9,2),(10,2)},
                'RedAndBlue':{(1,2),(2,2),(3,2),(6,2),(9,2),(10,2)}, 'Ribbon':{(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),(9,2),(10,2),(11,3)}, 
                'Dross':{(1,3),(1,4),(2,3),(2,4),(3,3),(3,4),(4,3),(4,4),(5,3),(5,4),(6,3),(6,4),(7,3),(7,4),(8,3),(8,4),(9,3),(9,4),(10,3),(10,4),(11,4),(12,2),(12,3),(12,4),(9,1)}
            }
            # Cast to float32 to prevent Pybind11 implicit copy overhead during C++ function calls
            cls._cached_card_key = np.array([
                koikoicore.cards_to_multi_hot_np([list(c) for c in card_set]) for _, card_set in card_dict.items()
            ], dtype=np.float32)
            
        if cls._card_suit_array is None:
            # Cast to float32 to prevent Pybind11 implicit copy overhead
            arr = np.zeros([12, 48], dtype=np.float32)
            for ii in range(12):
                arr[ii, 4*ii:4*ii+4] = 1.0
            cls._card_suit_array = arr

    def __init__(self, dealer=None):
        super().__init__(dealer)
        self._card_log_buf = np.zeros((17, 8, 48), dtype=np.float32)
        
        # ★C++のステートマネージャーをインスタンス化
        self.cpp_state = koikoicore.KoiKoiStateManager()
        self._init_bb()
        
    def _init_bb(self):
        # 初期状態のビットボードを計算し、C++側に渡す（ラウンド開始時のみ実行）
        h1 = sum(1 << c for c in self.hand[1])
        h2 = sum(1 << c for c in self.hand[2])
        f = sum(1 << c for c in self.field_slot if c != -1)
        s = sum(1 << c for c in self.stock)
        self.cpp_state.init_board(h1, h2, f, s)
        
    def yaku_point(self, player):
        # ★C++側で管理されている自身の山札ビットボードから直接計算
        return self.cpp_state.get_yaku_point(player, self.koikoi_num[player])
    
    def _to_multi_hot(self, cards):
        # Cast to float32 to prevent Pybind11 implicit copy overhead
        arr = np.zeros(48, dtype=np.float32)
        if cards:
            arr[cards] = 1.0
        return arr
    
    def discard(self, card=None): 
        out = super().discard(card)
        p = self.turn_player
        
        # C++へ状態更新を通知
        pair_bb = sum(1 << c for c in self.pairing_card)
        self.cpp_state.discard(p, card, pair_bb)

        idx = 0 if self.pairing_card else 1
        self._card_log_buf[self.turn_16, idx, :] = self._to_multi_hot(self.show)
        return out
        
    def discard_pick(self, card=None):
        out = super().discard_pick(card)
        p = self.turn_player
        
        # C++へ状態更新を通知
        coll_bb = sum(1 << c for c in self.collect)
        field_bb = sum(1 << c for c in self.field_slot if c != -1)
        self.cpp_state.discard_pick(p, coll_bb, field_bb)

        if self.collect:
            p_collect = self._to_multi_hot(self.field_collect)
            p_discard = self._to_multi_hot(self.log[f'turn{self.turn_16}']['pairCard'])
            self._card_log_buf[self.turn_16, 2, :] = p_collect
            self._card_log_buf[self.turn_16, 3, :] = p_discard - p_collect
        return out
        
    def draw(self, card=None):
        out = super().draw(card)
        drawn_card = self.show[0]
        
        # C++へ状態更新を通知
        pair_bb = sum(1 << c for c in self.pairing_card)
        self.cpp_state.draw(drawn_card, pair_bb)

        idx = 4 if self.pairing_card else 5
        self._card_log_buf[self.turn_16, idx, :] = self._to_multi_hot(self.show)
        return out
    
    def draw_pick(self, card=None):
        out = super().draw_pick(card)
        p = self.turn_player
        
        # C++へ状態更新を通知
        coll_bb = sum(1 << c for c in self.collect)
        field_bb = sum(1 << c for c in self.field_slot if c != -1)
        self.cpp_state.draw_pick(p, coll_bb, field_bb)

        if self.collect:
            p_collect = self._to_multi_hot(self.field_collect)
            p_discard = self._to_multi_hot(self.log[f'turn{self.turn_16}']['pairCard2'])
            self._card_log_buf[self.turn_16, 6, :] = p_collect
            self._card_log_buf[self.turn_16, 7, :] = p_discard - p_collect
        return out

    # step メソッドで koikoi 時にも同期
    def step(self, action):
        assert self.state in ['discard','discard-pick','draw','draw-pick','koikoi']
        if self.state == 'discard':
            self.discard(action)
        elif self.state == 'discard-pick':
            self.discard_pick(action)            
        elif self.state == 'draw':
            self.draw(action)            
        elif self.state == 'draw-pick':
            self.draw_pick(action)            
        elif self.state == 'koikoi':
            self.claim_koikoi(action)
        return

    @property
    def action_mask(self):
        # state_type: 0=discard, 1=pick, 2=koikoi にマッピング
        if self.state == 'discard':
            state_type = 0
        elif self.state in ['discard-pick', 'draw-pick']:
            state_type = 1
        else: # koikoi
            state_type = 2
            
        return self.cpp_state.get_action_mask(state_type, self.turn_player)
    
    # ─── 【重要】アロケーションゼロのためのバッファ埋め込みメソッド群 ───
    def _fill_yaku_status(self, buf_1d):
        my_hand_bb    = koikoicore.cards_to_bitboard(self.hand[self.turn_player])
        board_bb      = koikoicore.cards_to_bitboard(self.field)
        my_collect_bb = koikoicore.cards_to_bitboard(self.pile[self.turn_player])
        op_collect_bb = koikoicore.cards_to_bitboard(self.pile[self.idle_player])
        unseen_bb     = koikoicore.cards_to_bitboard(self.hand[self.idle_player] + self.stock)

        features = koikoicore.get_yaku_status_features_by_bitboard(
            my_hand_bb, board_bb, my_collect_bb, op_collect_bb, unseen_bb
        )
        idx = 0
        for f_list in features:
            for val in f_list:
                buf_1d[idx] = val
                idx += 1

    def _fill_card_init_position(self, view_2d):
        view_2d[0, :] = koikoicore.card_to_multi_hot(self.hand[self.turn_player])
        view_2d[1, :] = koikoicore.card_to_multi_hot(self.log['basic']['initBoard'])
        view_2d[2, :] = koikoicore.card_to_multi_hot(self.unseen_card[self.turn_player])

    def _fill_card_current_position(self, view_2d):
        view_2d[0, :] = koikoicore.card_to_multi_hot(self.hand[self.turn_player])
        view_2d[1, :] = koikoicore.card_to_multi_hot(self.pile[self.turn_player])
        view_2d[2, :] = koikoicore.card_to_multi_hot(self.field)
        view_2d[3, :] = koikoicore.card_to_multi_hot(self.pile[self.idle_player])
        view_2d[4, :] = koikoicore.card_to_multi_hot(self.unseen_card[self.turn_player])

    def _fill_card_pairing_state(self, view_2d):
        if self.state in ['discard-pick', 'draw-pick']:
            view_2d[0, :] = koikoicore.card_to_multi_hot(self.show)
            view_2d[1, :] = koikoicore.card_to_multi_hot(self.pairing_card)
        else:
            view_2d[0, :] = 0
            view_2d[1, :] = 0

    def _fill_card_log(self, view_2d):
        turn_list = [x for x in range(self.turn_16, 0, -1)] + [x for x in range(self.turn_16 + 1, 17)]
        idx = 0
        for turn in turn_list:
            d = self.card_log_dict[turn]
            view_2d[idx, :]   = d['CardDiscardedAndPaired']
            view_2d[idx+1, :] = d['CardDiscardedAndUnpaired']
            view_2d[idx+2, :] = d['CardPairedByDiscardCollect']
            view_2d[idx+3, :] = d['CardPairedByDiscardUncollect']
            view_2d[idx+4, :] = d['CardDrawnAndPaired']
            view_2d[idx+5, :] = d['CardDrawnAndUnpaired']
            view_2d[idx+6, :] = d['CardPairedByDrawnCollect']
            view_2d[idx+7, :] = d['CardPairedByDrawnUncollect']
            idx += 8
    
class KoiKoiGameState(KoiKoiGameStateBase):
    _order_cache = {
        i: np.array([x for x in range(i, 0, -1)] + [x for x in range(i + 1, 17)], dtype=np.intp)
        for i in range(1, 17)
    }
    _cache_buf = None  # C++へ渡すための共有キャッシュ

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if KoiKoiGameState._cache_buf is None:
            KoiKoiRoundState._init_caches()
            cb = np.zeros((25, 48), dtype=np.float32)
            cb[0:13, :] = KoiKoiRoundState._cached_card_key
            cb[13:25, :] = KoiKoiRoundState._card_suit_array
            KoiKoiGameState._cache_buf = cb

    @property
    def feature_np(self):
        rs = self.round_state
        is_koikoi = (rs.state == 'koikoi')
        turn_p = rs.turn_player
        idle_p = rs.idle_player
        
        f_turn = sum(v << i for i, v in enumerate(rs.koikoi[turn_p]))
        f_idle = sum(v << i for i, v in enumerate(rs.koikoi[idle_p]))

        # ★C++側で内部の永続ビットボードを使って一気に特徴量を組み上げる
        return rs.cpp_state.get_feature(
            is_koikoi, 
            self.point[turn_p], self.point[idle_p],
            self.round, rs.turn_16, rs.dealer,
            rs.koikoi_num[turn_p], rs.koikoi_num[idle_p],
            f_turn, f_idle,
            turn_p, idle_p,
            rs._card_log_buf,
            KoiKoiGameState._order_cache[rs.turn_16],
            KoiKoiGameState._cache_buf
        )

    def _fill_game_status(self, buf):
        turn_player = self.round_state.turn_player
        idle_player = self.round_state.idle_player
        
        point_diff = self.point[turn_player] - self.point[idle_player]
        
        def fill_tuple(idx, x, power=[0.5, 1, 1.5], weight=[1, 0.5, 0.1]):
            x = float(x)
            sgn = 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
            abs_x = abs(x)
            buf[idx]   = (abs_x ** power[0]) * sgn * weight[0]
            buf[idx+1] = (abs_x ** power[1]) * sgn * weight[1]
            buf[idx+2] = (abs_x ** power[2]) * sgn * weight[2]

        fill_tuple(0, float(point_diff)/2.0, [0.5,1,1.5], [1,0.5,0.1])
        fill_tuple(3, self.round_state.yaku_point(turn_player), [0.5,1,1.5], [1,0.5,0.1])
        fill_tuple(6, self.round_state.yaku_point(idle_player), [0.5,1,1.5], [1,0.5,0.1])
        
        buf[9:17] = 0
        buf[9 + self.round - 1] = 1
        
        buf[17:33] = 0
        buf[17 + self.round_state.turn_16 - 1] = 1
        
        buf[33:35] = 0
        buf[33 + self.round_state.dealer - 1] = 1
        
        def fill_tuple_2(idx, x, power=[1,2], weight=[1,1]):
            x = float(x)
            sgn = 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
            abs_x = abs(x)
            buf[idx]   = (abs_x ** power[0]) * sgn * weight[0]
            buf[idx+1] = (abs_x ** power[1]) * sgn * weight[1]

        fill_tuple_2(35, self.round_state.koikoi_num[turn_player])
        fill_tuple_2(37, self.round_state.koikoi_num[idle_player])
        
        buf[39:47] = self.round_state.koikoi[turn_player]
        buf[47:55] = self.round_state.koikoi[idle_player]
    
    @property
    def feature_tensor(self):
        return torch.from_numpy(self.feature_np)