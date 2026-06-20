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

class KoiKoiRoundState():
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
            cls._cached_card_key = np.array([
                koikoicore.cards_to_multi_hot_np([list(c) for c in card_set]) for _, card_set in card_dict.items()
            ], dtype=np.float32)
            
        if cls._card_suit_array is None:
            arr = np.zeros([12, 48], dtype=np.float32)
            for ii in range(12):
                arr[ii, 4*ii:4*ii+4] = 1.0
            cls._card_suit_array = arr

    def __init__(self, dealer=None):
        self.hand = {1:[], 2:[]}
        self.pile = {1:[], 2:[]}
        self.field_slot = []
        self.stock = []
        self.show = []
        self.collect = []
        self.turn_16 = 1
        self.dealer = random.randint(1,2) if dealer is None else dealer
        self.koikoi = {1:[0]*8, 2:[0]*8}
        self.winner = None
        self.exhausted = False
        self.log = {}
        self.turn_point = 0
        self.state = 'init'
        self.wait_action = False        
        
        # 型チェッカーおよびランタイム用のログバッファ初期化
        self._card_log_buf = np.zeros((17, 8, 48), dtype=np.float32)
        
        self.cpp_state = koikoicore.KoiKoiStateManager()
        self.__deal_card()
    
    def new_round(self):
        self.__init__(dealer=self.winner)
    
    @property
    def turn_player(self):
        return 1 if (self.turn_16 + self.dealer) % 2 == 0 else 2
    
    @property
    def idle_player(self):
        return 3 - self.turn_player
    
    @property
    def turn_8(self):
        return (self.turn_16 + 1) // 2
    
    @property
    def field(self):
        return sorted([slot for slot in self.field_slot if slot != -1])
    
    @property
    def pairing_card(self):
        if not self.show:
            return []
        target_suit = self.show[0] // 4
        return [card for card in self.field if card // 4 == target_suit]
    
    @property
    def koikoi_num(self):
        return {1:sum(self.koikoi[1]), 2:sum(self.koikoi[2])}
    
    @property
    def round_over(self):
        return self.state == 'round-over'
    
    @property
    def round_point(self):
        if self.winner is None:
            return {1:None, 2:None}
        elif self.exhausted:
            return {1:1, 2:-1} if self.dealer == 1 else {1:-1, 2:1}
        elif self.winner == 1:
            return {1:self.yaku_point(1), 2:-self.yaku_point(1)}
        else:
            return {1:-self.yaku_point(2), 2:self.yaku_point(2)}
    
    def __deal_card(self):
        while True:
            cards = list(range(48))
            random.shuffle(cards)
            
            self.hand[1] = sorted(cards[0:8])
            self.hand[2] = sorted(cards[8:16])
            self.field_slot = sorted(cards[16:24]) + [-1] * 10
            self.stock = cards[24:]
            
            flag = True
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
        
        self.log['basic'] = {'initBoard': self.field.copy()}
        
        # C++のステートマネージャー初期化
        h1 = sum(1 << c for c in self.hand[1])
        h2 = sum(1 << c for c in self.hand[2])
        f = sum(1 << c for c in self.field_slot if c != -1)
        s = sum(1 << c for c in self.stock)
        self.cpp_state.init_board(h1, h2, f, s)
        
        self.state = 'discard'
        self.wait_action = True
    
    def __collect_card(self, card):
        if len(self.pairing_card) == 0:
            self.collect = []
            self.field_slot[self.field_slot.index(-1)] = self.show[0]
        elif len(self.pairing_card) in [1, 3]:
            self.collect = self.show + self.pairing_card
            for paired_card in self.pairing_card:
                self.field_slot[self.field_slot.index(paired_card)] = -1
            self.pile[self.turn_player].extend(self.collect)
        else:
            self.collect = self.show + [card]
            self.field_slot[self.field_slot.index(card)] = -1
            self.pile[self.turn_player].extend(self.collect)
    
    def yaku_point(self, player):
        return self.cpp_state.get_yaku_point(player, self.koikoi_num[player])
    
    def _to_multi_hot(self, cards):
        arr = np.zeros(48, dtype=np.float32)
        if cards:
            arr[cards] = 1.0
        return arr
    
    def discard(self, card=None):        
        self.turn_point = self.yaku_point(self.turn_player)
        ind = self.hand[self.turn_player].index(card)
        self.show = [self.hand[self.turn_player].pop(ind)]
        
        self.log[f'turn{self.turn_16}'] = {'pairCard': self.pairing_card.copy()}
        
        pair_bb = sum(1 << c for c in self.pairing_card)
        self.cpp_state.discard(self.turn_player, card, pair_bb)

        idx = 0 if self.pairing_card else 1
        self._card_log_buf[self.turn_16, idx, :] = self._to_multi_hot(self.show)
        self.state = 'discard-pick'
        self.wait_action = len(self.pairing_card) == 2
        return self.state
        
    def discard_pick(self, card=None):
        self.__collect_card(card)
        
        coll_bb = sum(1 << c for c in self.collect)
        field_bb = sum(1 << c for c in self.field_slot if c != -1)
        self.cpp_state.discard_pick(self.turn_player, coll_bb, field_bb)

        if self.collect:
            fc = [c for c in self.collect if c != self.show[0]]
            p_discard = self._to_multi_hot(self.log[f'turn{self.turn_16}']['pairCard'])
            self._card_log_buf[self.turn_16, 2, :] = self._to_multi_hot(fc)
            self._card_log_buf[self.turn_16, 3, :] = p_discard - self._to_multi_hot(fc)
            
        self.state = 'draw'
        self.wait_action = False
        return self.state
        
    def draw(self, card=None):
        self.show = [self.stock.pop()]
        self.log[f'turn{self.turn_16}']['pairCard2'] = self.pairing_card.copy()
        
        pair_bb = sum(1 << c for c in self.pairing_card)
        self.cpp_state.draw(self.show[0], pair_bb)

        idx = 4 if self.pairing_card else 5
        self._card_log_buf[self.turn_16, idx, :] = self._to_multi_hot(self.show)
        self.state = 'draw-pick'
        self.wait_action = len(self.pairing_card) == 2
        return self.state
        
    def draw_pick(self, card=None):
        self.__collect_card(card)
        
        coll_bb = sum(1 << c for c in self.collect)
        field_bb = sum(1 << c for c in self.field_slot if c != -1)
        self.cpp_state.draw_pick(self.turn_player, coll_bb, field_bb)

        if self.collect:
            fc = [c for c in self.collect if c != self.show[0]]
            p_discard = self._to_multi_hot(self.log[f'turn{self.turn_16}']['pairCard2'])
            self._card_log_buf[self.turn_16, 6, :] = self._to_multi_hot(fc)
            self._card_log_buf[self.turn_16, 7, :] = p_discard - self._to_multi_hot(fc)
            
        self.state = 'koikoi'
        self.wait_action = (self.yaku_point(self.turn_player) > self.turn_point) and (self.turn_8 < 8)
        return self.state
    
    def claim_koikoi(self, is_koikoi=None):
        if (self.yaku_point(self.turn_player) > self.turn_point) and (self.turn_8 == 8):
            is_koikoi = False
        self.koikoi[self.turn_player][self.turn_8-1] = int(is_koikoi==True)
        
        if is_koikoi == False:
            self.state = 'round-over'
            self.wait_action = False
            self.winner = self.turn_player
        elif self.turn_16 == 16:
            self.state = 'round-over'
            self.wait_action = False
            self.exhausted = True
            self.winner = self.dealer
        else:
            self.turn_16 += 1
            self.state = 'discard'
            self.wait_action = True
        return self.state

    def step(self, action):
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

    @property
    def action_mask(self):
        if self.state == 'discard':
            state_type = 0
        elif self.state in ['discard-pick', 'draw-pick']:
            state_type = 1
        else:
            state_type = 2
        return self.cpp_state.get_action_mask(state_type, self.turn_player)
    
    def yaku(self, player):
        bb = 0
        for c in self.pile[player]:
            bb |= (1 << c)
        return koikoicore.evaluate_yaku_by_bitboard(bb, self.koikoi_num[player])

class KoiKoiGameState():
    _order_cache = {
        i: np.array([x for x in range(i, 0, -1)] + [x for x in range(i + 1, 17)], dtype=np.intp)
        for i in range(1, 17)
    }
    _cache_buf = None

    def __init__(self, round_num=1, round_total=DefaultVar.DEFAULT_ROUND_TOTAL,
                 init_point=[DefaultVar.DEFAULT_INIT_POINT, DefaultVar.DEFAULT_INIT_POINT],
                 init_dealer=None, player_name=['Player1', 'Player2'], 
                 record_path='', save_record=False):
        
        self.round_total = round_total
        self.init_point = init_point
        self.init_dealer = init_dealer        
        self.player_name = {1:player_name[0], 2:player_name[1]}
        self.record_path = record_path
        self.save_record = save_record
        
        self.round_state = KoiKoiRoundState(dealer=self.init_dealer)
        self.round = round_num
        self.point = {1:init_point[0], 2:init_point[1]}
        self.game_over = False
        self.winner = None
        self.log = {}
        
        if KoiKoiGameState._cache_buf is None:
            KoiKoiRoundState._init_caches()
            cb = np.zeros((25, 48), dtype=np.float32)
            cb[0:13, :] = KoiKoiRoundState._cached_card_key
            cb[13:25, :] = KoiKoiRoundState._card_suit_array
            KoiKoiGameState._cache_buf = cb
            
        self.__init_record()
        
    def new_game(self):
        self.__init__(round_num=1, round_total=self.round_total, 
                      init_point=self.init_point, init_dealer=self.init_dealer,
                      player_name=[self.player_name[1], self.player_name[2]], 
                      record_path=self.record_path, save_record=self.save_record)
        
    def new_round(self):
        assert self.round_state.state == 'round-over'
        self.point[1] += self.round_state.round_point[1]
        self.point[2] += self.round_state.round_point[2]
        self.__round_result_record()
        
        if self.point[1] <= 0 or self.point[2] <= 0 or self.round == self.round_total:
            self.game_over = True
            self.winner = 1 if self.point[1] > self.point[2] else (2 if self.point[1] < self.point[2] else 0)
            self.__game_result_record()
        else:
            self.round_state.new_round()
            self.round += 1
    
    def __init_record(self):
        self.log['info'] = {'startTime':time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                            'endTime':None,
                            'player1Name':self.player_name[1],
                            'player2Name':self.player_name[2],
                            'player1InitPts':self.point[1], 
                            'player2InitPts':self.point[2], 
                            'numRound':self.round_total}
        self.log['result'] = {'isOver':False, 'gameWinner':None, 
                              'player1EndPts':None, "player2EndPts":None}
        self.log['record'] = {}
    
    def __round_result_record(self):
        round_log = self.round_state.log.copy()
        if 'basic' not in round_log:
            round_log['basic'] = {}
        else:
            round_log['basic'] = round_log['basic'].copy()
            
        round_log['basic']['player1RoundPts'] = self.round_state.round_point[1]
        round_log['basic']['player2RoundPts'] = self.round_state.round_point[2]
        
        if 'Dealer' not in round_log['basic']:
            round_log['basic']['Dealer'] = self.round_state.dealer
            
        self.log['record']['round'+str(self.round)] = round_log
    
    def __game_result_record(self):
        self.log['info']['endTime'] = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log['result'] = {'isOver':True, 'gameWinner':self.winner,
                              'player1EndPts':self.point[1], "player2EndPts":self.point[2]}
        if self.save_record:
            filename = self.record_path + self.log['info']['startTime'] + ' ' \
                + self.log['info']['player1Name'] + ' vs ' + self.log['info']['player2Name'] + '.json'
            with open(filename, 'w') as f:
                json.dump(self.log, f)
       
    @property
    def feature_np(self):
        rs = self.round_state
        is_koikoi = (rs.state == 'koikoi')
        turn_p = rs.turn_player
        idle_p = rs.idle_player
        
        f_turn = sum(v << i for i, v in enumerate(rs.koikoi[turn_p]))
        f_idle = sum(v << i for i, v in enumerate(rs.koikoi[idle_p]))

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
    
    @property
    def feature_tensor(self):
        return torch.from_numpy(self.feature_np)