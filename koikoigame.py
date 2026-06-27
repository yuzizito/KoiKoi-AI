# author: shguan3

import json
import time
import numpy as np
import torch
import random
import configparser
import os
import koikoicore

MAX_ROUND = 6  # 1〜8の整数で設定可能

def apply_koikoi_config(cfg_path='koikoi.cfg'):
    config = configparser.ConfigParser()
    default_rules = {
        'five_lights': 10, 'four_lights': 8, 'rainy_four_lights': 7,
        'three_lights': 5, 'bdb': 5,
        'enable_flower_sake': True, 'enable_moon_sake': True,
        'flower_moon_base_pt': 1, 'flower_moon_koikoi_pt': 3,
        'red_blue_ribbon': 10, 'red_ribbon': 5, 'blue_ribbon': 5
    }
    
    if os.path.exists(cfg_path):
        config.read(cfg_path, encoding='utf-8')
        if 'Rules' in config:
            cfg_rules = config['Rules']
            apply_dict = {}
            for key, default_val in default_rules.items():
                if key in cfg_rules:
                    if isinstance(default_val, bool):
                        apply_dict[key] = cfg_rules.getboolean(key)
                    else:
                        apply_dict[key] = cfg_rules.getint(key)
            koikoicore.set_rules(apply_dict)
            return
            
    koikoicore.set_rules(default_rules)
    
apply_koikoi_config()

class KoiKoiRoundState:
    def __init__(self, dealer=None):
        self.hand = {1: [], 2: []}
        self.pile = {1: [], 2: []}
        self.field_slot = []
        self.stock = []
        self.show = []
        self.collect = []
        self.turn_16 = 1
        self.dealer = random.randint(1, 2) if dealer is None else dealer
        self.koikoi = {1: [0]*8, 2: [0]*8}
        self.winner = None
        self.exhausted = False
        self.log = {}
        self.turn_point = 0
        self.state = 'init'
        self.wait_action = False        
        
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
        return sorted(slot for slot in self.field_slot if slot != -1)
    
    @property
    def pairing_card(self):
        if not self.show:
            return []
        target_suit = self.show[0] // 4 # type: ignore
        return [card for card in self.field if card // 4 == target_suit]
    
    @property
    def koikoi_num(self):
        return {1: sum(self.koikoi[1]), 2: sum(self.koikoi[2])}
    
    @property
    def round_over(self):
        return self.state == 'round-over'
    
    @property
    def round_point(self):
        if self.winner is None:
            return {1: None, 2: None}
        if self.exhausted:
            return {1: 1, 2: -1} if self.dealer == 1 else {1: -1, 2: 1}
        
        pts = self.yaku_point(self.winner)
        return {1: pts, 2: -pts} if self.winner == 1 else {1: -pts, 2: pts}
    
    def __deal_card(self):
        while True:
            cards = list(range(48))
            random.shuffle(cards)
            
            self.hand[1] = sorted(cards[:8])
            self.hand[2] = sorted(cards[8:16])
            self.field_slot = sorted(cards[16:24]) + [-1] * 10
            self.stock = cards[24:]
            
            flag = True
            for suit in range(12):
                c1 = sum(1 for c in self.hand[1] if c // 4 == suit)
                c2 = sum(1 for c in self.hand[2] if c // 4 == suit)
                cf = sum(1 for c in self.field if c // 4 == suit)
                if c1 == 4 or c2 == 4 or cf == 4:
                    flag = False
                    break
            if flag:
                break
        
        self.log['basic'] = {'initBoard': self.field.copy()}
        
        h1 = sum(1 << c for c in self.hand[1])
        h2 = sum(1 << c for c in self.hand[2])
        f = sum(1 << c for c in self.field_slot if c != -1)
        s = sum(1 << c for c in self.stock)
        self.cpp_state.init_board(h1, h2, f, s)
        
        self.state = 'discard'
        self.wait_action = True
    
    def __collect_card(self, card):
        pairs = self.pairing_card
        if not pairs:
            self.collect = []
            self.field_slot[self.field_slot.index(-1)] = self.show[0] # type: ignore
        elif len(pairs) in (1, 3):
            self.collect = self.show + pairs
            for pc in pairs:
                self.field_slot[self.field_slot.index(pc)] = -1
            self.pile[self.turn_player].extend(self.collect)
        else:
            self.collect = self.show + [card]
            self.field_slot[self.field_slot.index(card)] = -1
            self.pile[self.turn_player].extend(self.collect)
    
    def yaku_point(self, player):
        return self.cpp_state.get_yaku_point(player, sum(self.koikoi[player]))
    
    def discard(self, card=None):        
        self.turn_point = self.yaku_point(self.turn_player)
        self.hand[self.turn_player].remove(card)
        self.show = [card]
        
        pairs = self.pairing_card
        self.log[f'turn{self.turn_16}'] = {'pairCard': pairs.copy()}
        
        pair_bb = sum(1 << c for c in pairs)
        self.cpp_state.discard(self.turn_player, card, pair_bb, self.turn_16)

        self.state = 'discard-pick'
        self.wait_action = (len(pairs) == 2)
        
    def discard_pick(self, card=None):
        self.__collect_card(card)
        
        coll_bb = sum(1 << c for c in self.collect) # type: ignore
        field_bb = sum(1 << c for c in self.field_slot if c != -1)
        self.cpp_state.discard_pick(self.turn_player, coll_bb, field_bb, self.turn_16)

        self.state = 'draw'
        self.wait_action = False
        
    def draw(self, card=None):
        self.show = [self.stock.pop()]
        pairs = self.pairing_card
        self.log[f'turn{self.turn_16}']['pairCard2'] = pairs.copy()
        
        pair_bb = sum(1 << c for c in pairs)
        self.cpp_state.draw(self.show[0], pair_bb, self.turn_16)

        self.state = 'draw-pick'
        self.wait_action = (len(pairs) == 2)
        
    def draw_pick(self, card=None):
        self.__collect_card(card)
        
        coll_bb = sum(1 << c for c in self.collect) # type: ignore
        field_bb = sum(1 << c for c in self.field_slot if c != -1)
        self.cpp_state.draw_pick(self.turn_player, coll_bb, field_bb, self.turn_16)

        self.state = 'koikoi'
        self.wait_action = (self.yaku_point(self.turn_player) > self.turn_point) and (self.turn_8 < 8)
    
    def claim_koikoi(self, is_koikoi=None):
        if (self.yaku_point(self.turn_player) > self.turn_point) and (self.turn_8 == 8):
            is_koikoi = False
            
        self.koikoi[self.turn_player][self.turn_8-1] = int(is_koikoi is True)
        
        if is_koikoi is False:
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
        elif self.state in ('discard-pick', 'draw-pick'):
            state_type = 1
        else:
            state_type = 2
        return self.cpp_state.get_action_mask(state_type, self.turn_player)
    
    def yaku(self, player):
        return self.cpp_state.get_yaku_list(player, sum(self.koikoi[player]))

class KoiKoiGameState:
    def __init__(self, round_num=1, round_total=MAX_ROUND,
                 init_point=(30, 30), init_dealer=None,
                 player_name=('Player1', 'Player2'), 
                 record_path='', save_record=False):
        
        self.round_total = round_total
        self.init_point = list(init_point)
        self.init_dealer = init_dealer        
        self.player_name = {1: player_name[0], 2: player_name[1]}
        self.record_path = record_path
        self.save_record = save_record
        
        self.round_state = KoiKoiRoundState(dealer=self.init_dealer)
        self.round = round_num
        self.point = {1: self.init_point[0], 2: self.init_point[1]}
        self.game_over = False
        self.winner = None
        self.log = {}
            
        self.__init_record()
        
    def new_game(self):
        self.__init__(round_num=1, round_total=self.round_total, 
                      init_point=self.init_point, init_dealer=self.init_dealer,
                      player_name=[self.player_name[1], self.player_name[2]], 
                      record_path=self.record_path, save_record=self.save_record)
        
    def new_round(self):
        assert self.round_state.state == 'round-over'
        pts = self.round_state.round_point
        self.point[1] += pts[1]
        self.point[2] += pts[2]
        self.__round_result_record()
        
        if self.point[1] <= 0 or self.point[2] <= 0 or self.round == self.round_total:
            self.game_over = True
            if self.point[1] > self.point[2]:
                self.winner = 1
            elif self.point[1] < self.point[2]:
                self.winner = 2
            else:
                self.winner = 0
            self.__game_result_record()
        else:
            self.round_state.new_round()
            self.round += 1
    
    def __init_record(self):
        self.log['info'] = {
            'startTime': time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
            'endTime': None,
            'player1Name': self.player_name[1],
            'player2Name': self.player_name[2],
            'player1InitPts': self.point[1], 
            'player2InitPts': self.point[2], 
            'numRound': self.round_total
        }
        self.log['result'] = {
            'isOver': False, 'gameWinner': None, 
            'player1EndPts': None, 'player2EndPts': None
        }
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
            
        self.log['record'][f'round{self.round}'] = round_log
    
    def __game_result_record(self):
        self.log['info']['endTime'] = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log['result'] = {
            'isOver': True, 'gameWinner': self.winner,
            'player1EndPts': self.point[1], 'player2EndPts': self.point[2]
        }
        if self.save_record:
            filename = f"{self.record_path}{self.log['info']['startTime']} {self.player_name[1]} vs {self.player_name[2]}.json"
            with open(filename, 'w') as f:
                json.dump(self.log, f)
       
    @property
    def feature_np(self):
        rs = self.round_state
        is_koikoi = (rs.state == 'koikoi')
        turn_p = rs.turn_player
        idle_p = rs.idle_player

        return rs.cpp_state.get_feature(
            is_koikoi, 
            self.point[turn_p], self.point[idle_p],
            self.round, rs.turn_16, rs.dealer,
            sum(rs.koikoi[turn_p]), sum(rs.koikoi[idle_p]),
            turn_p, idle_p,
            MAX_ROUND
        )
    
    @property
    def feature_tensor(self):
        return torch.from_numpy(self.feature_np)