#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 22:30:19 2021

@author: guansanghai
"""

import random
import json
import time

import numpy as np
import torch
import koikoicore

class DefaultVar():
    DEFAULT_ROUND_TOTAL = 8
    DEFAULT_INIT_POINT = 30

card_to_multi_hot = koikoicore.card_to_multi_hot

class KoiKoiCard():
    crane = {(1,1)}
    curtain = {(3,1)}
    moon = {(8,1)}
    rainman = {(11,1)}
    phoenix = {(12,1)}
    sake = {(9,1)}
    
    light  = {(1,1),(3,1),(8,1),(11,1),(12,1)}
    seed   = {(2,1),(4,1),(5,1),(6,1),(7,1),(8,2),(9,1),(10,1),(11,2)}
    ribbon = {(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),(9,2),(10,2),(11,3)}
    dross  = {(1,3),(1,4),(2,3),(2,4),(3,3),(3,4),(4,3),(4,4),(5,3),(5,4),(6,3),(6,4),(7,3),
              (7,4),(8,3),(8,4),(9,3),(9,4),(10,3),(10,4),(11,4),(12,2),(12,3),(12,4),(9,1)}
            
    boar_deer_butterfly = {(6,1),(7,1),(10,1)}
    flower_sake = {(3,1),(9,1)}
    moon_sake = {(8,1),(9,1)}
    red_ribbon = {(1,2),(2,2),(3,2)}
    blue_ribbon = {(6,2),(9,2),(10,2)}
    red_blue_ribbon = {(1,2),(2,2),(3,2),(6,2),(9,2),(10,2)}

YAKU_ID_TO_NAME = {
    1: 'Five Lights', 2: 'Four Lights', 3: 'Rainy Four Lights', 4: 'Three Lights',
    5: 'Boar-Deer-Butterfly', 6: 'Flower Viewing Sake', 7: 'Flower Viewing Sake',
    8: 'Moon Viewing Sake', 9: 'Moon Viewing Sake', 10: 'Tane',
    11: 'Red & Blue Ribbons', 12: 'Red Ribbons', 13: 'Blue Ribbons', 14: 'Tan',
    15: 'Kasu', 16: 'Koi-Koi'
}

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
        
        # action
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
        return sorted([slot for slot in self.field_slot if slot!=[0,0]])
    
    @property
    def unseen_card(self):
        return {1:(self.stock+self.hand[2]), 2:(self.stock+self.hand[1])}
    
    @property
    def pairing_card(self):
        return [card for card in self.field if card[0]==self.show[0][0]]
    
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
            card = [[ii+1,jj+1] for ii in range(12) for jj in range (4)]
            random.shuffle(card)
            self.hand[1] = sorted(card[0:8])
            self.hand[2] = sorted(card[8:16])
            self.field_slot = sorted(card[16:24])+[[0,0] for _ in range(10)]
            self.stock = card[24:]
            flag = True
            for suit in range(1,13):
                if 4 in [[card[0] for card in self.hand[1]].count(suit),
                         [card[0] for card in self.hand[2]].count(suit),
                         [card[0] for card in self.field].count(suit)]:
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
            self.field_slot[self.field_slot.index([0,0])] = self.show[0]
        elif len(self.pairing_card) in [1,3]:
            self.collect = self.show + self.pairing_card
            for paired_card in self.pairing_card:
                self.field_slot[self.field_slot.index(paired_card)] = [0,0]
            self.pile[self.turn_player].extend(self.collect)
        else:
            self.collect = self.show + [card]
            self.field_slot[self.field_slot.index(card)] = [0,0]
            self.pile[self.turn_player].extend(self.collect)
        return
    
    def discard(self, card=None):        
        assert self.state == 'discard'
        assert card in self.hand[self.turn_player]
        self.turn_point = self.yaku_point(self.turn_player)
        ind = self.hand[self.turn_player].index(card)
        self.show = [self.hand[self.turn_player].pop(ind)]
        self.__write_log()
        self.state = 'discard-pick'
        self.wait_action = len(self.pairing_card) == 2
        return self.state if self.silence else self.__call__()
        
    def discard_pick(self, card=None):
        assert self.state == 'discard-pick'
        assert (card in self.pairing_card) if self.wait_action else (card == None)
        self.__collect_card(card)
        self.__write_log()
        self.state = 'draw'
        self.wait_action = False
        return self.state if self.silence else self.__call__()
        
    def draw(self, card=None):
        assert self.state == 'draw'
        self.show = [self.stock.pop()]
        self.__write_log()
        self.state = 'draw-pick'
        self.wait_action = len(self.pairing_card) == 2
        return self.state if self.silence else self.__call__()
        
    def draw_pick(self, card=None):
        assert self.state == 'draw-pick'
        assert (card in self.pairing_card) if self.wait_action else (card == None)
        self.__collect_card(card)
        self.__write_log()
        self.state = 'koikoi'
        self.wait_action = (self.yaku_point(self.turn_player) > self.turn_point) and (self.turn_8 < 8)
        return self.state if self.silence else self.__call__()
    
    def claim_koikoi(self, is_koikoi=None):
        assert self.state == 'koikoi'
        assert (type(is_koikoi) == bool) if self.wait_action else (is_koikoi == None)
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
            
        return self.state if self.silence else self.__call__()

    def yaku(self, player):
        pile_bb = koikoicore.cards_to_bitboard(self.pile[player])
        yaku_ids = koikoicore.evaluate_yaku_id_by_bitboard(pile_bb, self.koikoi_num[player])
        return [(y_id, YAKU_ID_TO_NAME.get(y_id, 'Unknown'), pt) for y_id, pt in yaku_ids]
    
    def yaku_point(self, player):
        return koikoicore.get_yaku_point(self.pile[player], self.koikoi_num[player])
    
    def __write_log(self, content=None):
        turn = str(self.turn_16)
        if self.state == 'init':
            self.log['basic'] = {}
            self.log['basic']['initHand1'] = self.hand[1].copy()
            self.log['basic']['initHand2'] = self.hand[2].copy()
            self.log['basic']['initBoard'] = self.field.copy()
            self.log['basic']['initPile'] = self.stock.copy()
            self.log['basic']['Dealer'] = self.dealer
        elif self.state == 'discard':
            self.log['turn'+turn] = {}
            self.log['turn'+turn]['playerInTurn'] = self.turn_player
            self.log['turn'+turn]['discardCard'] = self.show[0].copy()
            self.log['turn'+turn]['pairCard'] = self.pairing_card.copy()
        elif self.state == 'discard-pick':
            self.log['turn'+turn]['collectCard'] = self.collect.copy()
        elif self.state == 'draw':
            self.log['turn'+turn]['drawCard'] = self.show[0].copy()
            self.log['turn'+turn]['pairCard2'] = self.pairing_card.copy()
        elif self.state == 'draw-pick':
            self.log['turn'+turn]['collectCard2'] = self.collect.copy()
        elif self.state == 'koikoi':
            self.log['turn'+turn]['isKoiKoi'] = content
        elif self.state == 'round-over':
            self.log['basic']['roundWinner'] = self.winner
            self.log['basic']['player1RoundPts'] = self.round_point[1]
            self.log['basic']['player2RoundPts'] = self.round_point[2]
        return
    
    def __call__(self, view=None):
        view = self.turn_player if view == None else view
        op_view = 3-view
        pile = set([tuple(card) for card in self.pile[view]])
        op_pile = set([tuple(card) for card in self.pile[op_view]])
        
        print('Turn: '+str(self.turn_8)+',  State: '+self.state)
        print('-----------------------------------------------')
        print('Opponent\'s Yaku:')
        print([[yaku[1],yaku[2]] for yaku in self.yaku(op_view)])
        print('Total Point: '+str(self.yaku_point(op_view)))
        print('-----------------------------------------------')
        print('Opponent\'s Pile:')
        print('Light: '+ str(list(op_pile & KoiKoiCard.light)))
        print('Seed: '+ str(list(op_pile & KoiKoiCard.seed)))
        print('Ribbon: '+ str(list(op_pile & KoiKoiCard.ribbon)))
        print('Dross: '+ str(list(op_pile & KoiKoiCard.dross)))
        print('-----------------------------------------------')
        print('Opponent\'s Hand:')
        print([[0,0] for card in self.hand[op_view]])
        print('-----------------------------------------------')
        print('Field:')
        print(self.field)
        print('-----------------------------------------------')
        print('Your Hand:')
        print(self.hand[view])
        print('-----------------------------------------------')
        print('Your Pile:')
        print('Light: '+ str(list(pile & KoiKoiCard.light)))
        print('Seed: '+ str(list(pile & KoiKoiCard.seed)))
        print('Ribbon: '+ str(list(pile & KoiKoiCard.ribbon)))
        print('Dross: '+ str(list(pile & KoiKoiCard.dross)))
        print('-----------------------------------------------')
        print('Your Yaku:')
        print([[yaku[1],yaku[2]] for yaku in self.yaku(view)])
        print('Total Point: '+str(self.yaku_point(view)))
        print('-----------------------------------------------')
        
        if view != self.turn_player:
            print('Opponent\'s turn, waiting action...')
            return
            
        if self.state == 'discard':
            print('Use discard(card) to discard from hand.')
        elif self.state == 'discard-pick':
            print('Discard: '+str(self.show[0]))
            print('Pairing: '+str(self.pairing_card))
            if self.wait_action:
                print('Use discard_pick(card) to pick a pairing field card.')
            else:
                print('Use discard_pick() to continue.')
        elif self.state == 'draw':
            print('Use draw() to draw from stock.')
        elif self.state == 'draw-pick':
            print('Draw: '+str(self.show[0]))
            print('Pairing: '+str(self.pairing_card))
            if self.wait_action:
                print('Use draw_pick(card) to pick a pairing field card.')
            else:
                print('Use draw_pick() to continue.')
        elif self.state == 'koikoi':
            if self.wait_action:
                print('Use claim_koikoi(bool) to koikoi or stop.')
            else:
                print('Use claim_koikoi() to continue.')        
        elif self.state == 'round-over':
            print('Round Over')
            print('Round Point: You '+str(self.round_point[view])+\
                  ', Opponent '+str(self.round_point[op_view]))
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
        self.log['record']['round'+str(self.round)] = self.round_state.log
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

    def __call__(self):
        print('-----------------------------------------------')
        print('Round: '+str(self.round)+' / '+str(self.round_total))
        print(self.log['info']['player1Name']+': '+str(self.point[1])+', '+\
              self.log['info']['player2Name']+': '+str(self.point[2]))
        if self.game_over:
            print('Game Over')
        print('-----------------------------------------------')
        return
       
class KoiKoiRoundState(KoiKoiRoundStateBase):
    
    # 【追加】クラス変数としてキャッシュを定義
    _cached_card_key = None
    _card_suit_array = None
    
    @classmethod
    def _init_caches(cls):
        if cls._cached_card_key is None:
            card_dict = {
                'Crane':KoiKoiCard.crane, 'Curtain':KoiKoiCard.curtain, 'Moon':KoiKoiCard.moon,
                'Rainman':KoiKoiCard.rainman, 'Phoenix':KoiKoiCard.phoenix, 'Sake':KoiKoiCard.sake,
                'BoarDeerButterfly':KoiKoiCard.boar_deer_butterfly, 'Seed':KoiKoiCard.seed,
                'RedRibbon':KoiKoiCard.red_ribbon, 'BlueRibbon':KoiKoiCard.blue_ribbon,
                'RedAndBlue':KoiKoiCard.red_blue_ribbon, 'Ribbon':KoiKoiCard.ribbon, 'Dross':KoiKoiCard.dross
            }
            cls._cached_card_key = np.array([
                koikoicore.card_to_multi_hot([list(c) for c in card_set]) for _, card_set in card_dict.items()
            ], dtype=np.float32)
            
        if cls._card_suit_array is None:
            arr = np.zeros([12, 48], dtype=np.float32)
            for ii in range(12):
                arr[ii, 4*ii:4*ii+4] = 1.0
            cls._card_suit_array = arr
    
    def __init__(self, dealer=None):
        super().__init__(dealer)
        self.card_log_dict = {}
        self.__write_card_log_array('init')
    
    def discard(self, card=None): 
        output = super().discard(card)
        self.__write_card_log_array('discard')
        return output
        
    def discard_pick(self, card=None):
        output = super().discard_pick(card)
        self.__write_card_log_array('discard-pick')
        return output
        
    def draw(self, card=None):
        output = super().draw(card)
        self.__write_card_log_array('draw')
        return output
        
    def draw_pick(self, card=None):
        output = super().draw_pick(card)
        self.__write_card_log_array('draw-pick')
        return output
    
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
    
    def __write_card_log_array(self,state):
        def card_log_turn_dict():
            cardLogTurn = {}
            cardLogTurn['CardDiscardedAndPaired'] = np.zeros(48)
            cardLogTurn['CardDiscardedAndUnpaired'] = np.zeros(48)
            cardLogTurn['CardPairedByDiscardCollect'] = np.zeros(48)
            cardLogTurn['CardPairedByDiscardUncollect'] = np.zeros(48)
            cardLogTurn['CardDrawnAndPaired'] = np.zeros(48)
            cardLogTurn['CardDrawnAndUnpaired'] = np.zeros(48)
            cardLogTurn['CardPairedByDrawnCollect'] = np.zeros(48)
            cardLogTurn['CardPairedByDrawnUncollect'] = np.zeros(48)
            return cardLogTurn
    
        turn = self.turn_16
        if state == 'init':
            for ii in range(1,17):
                self.card_log_dict[ii] = card_log_turn_dict()
                
        elif state == 'discard':
            if self.pairing_card == []:
                self.card_log_dict[turn]['CardDiscardedAndUnpaired'] \
                    = np.array(card_to_multi_hot(self.show))
            else:
                self.card_log_dict[turn]['CardDiscardedAndPaired'] \
                    = np.array(card_to_multi_hot(self.show)) 
            
        elif state == 'discard-pick':
            if self.collect != []:
                pair_by_discard_collect = np.array(card_to_multi_hot(self.field_collect))
                pair_by_discard = np.array(card_to_multi_hot(self.log[f'turn{turn}']['pairCard']))
                self.card_log_dict[turn]['CardPairedByDiscardCollect'] \
                    = pair_by_discard_collect
                self.card_log_dict[turn]['CardPairedByDiscardUncollect'] \
                    = pair_by_discard - pair_by_discard_collect
            
        elif state == 'draw':
            if self.pairing_card == []:
                self.card_log_dict[turn]['CardDrawnAndUnpaired'] \
                    = np.array(card_to_multi_hot(self.show))
            else:
                self.card_log_dict[turn]['CardDrawnAndPaired'] \
                    = np.array(card_to_multi_hot(self.show))
            
        elif state == 'draw-pick':
            if self.collect != []:
                pair_by_discard_collect = np.array(card_to_multi_hot(self.field_collect))
                pair_by_discard = np.array(card_to_multi_hot(self.log[f'turn{turn}']['pairCard2']))
                self.card_log_dict[turn]['CardPairedByDrawnCollect'] \
                    = pair_by_discard_collect
                self.card_log_dict[turn]['CardPairedByDrawnUncollect'] \
                    = pair_by_discard - pair_by_discard_collect
        return
    
    @property
    def action_mask(self):
        if self.state == 'discard':
            mask = card_to_multi_hot(self.hand[self.turn_player])
        elif self.state in ['discard-pick', 'draw-pick']:
            mask = card_to_multi_hot(self.pairing_card)
        elif self.state == 'koikoi':
            mask = [1,1]
        else:
            mask = []
        return np.array(mask)
    
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
    
    #@property
    #def card_log_array(self):
    #    turn_list = [x for x in range(self.turn_16, 0, -1)] + [x for x in range(self.turn_16 + 1, 17)]
    #    # NumPy配列のリストにしてから vstack（マルチプロセス内で最軽量）
    #    arrays = [f for turn in turn_list for _, f in self.card_log_dict[turn].items()]
    #    return np.vstack(arrays)
    
    #@property
    #def card_suit_array(self):
    #    f_array = np.zeros([12,48])
    #    for ii in range(12):
    #        f_array[ii,4*ii:4*ii+4] = 1
    #    return f_array
    
    #@property
    #def card_init_position_array(self):
    #    # C++から戻るリストを np.array() で確実にNumPy配列化
    #    f_dict = {}
    #    f_dict['CardInMyHand'] = np.array(card_to_multi_hot(self.hand[self.turn_player]))
    #    f_dict['CardInBoard'] = np.array(card_to_multi_hot(self.log['basic']['initBoard']))
    #    f_dict['CardUnseen'] = np.array(card_to_multi_hot(self.unseen_card[self.turn_player]))
    #    return np.vstack([value for key, value in f_dict.items()])
    
    #@property
    #def card_current_position_array(self):
    #    f_dict = {}
    #    f_dict['CardInMyHand'] = np.array(card_to_multi_hot(self.hand[self.turn_player]))
    #    f_dict['CardInMyCollect'] = np.array(card_to_multi_hot(self.pile[self.turn_player]))
    #    f_dict['CardInBoard'] = np.array(card_to_multi_hot(self.field))
    #    f_dict['CardInOpCollect'] = np.array(card_to_multi_hot(self.pile[self.idle_player]))
    #    f_dict['CardUnseen'] = np.array(card_to_multi_hot(self.unseen_card[self.turn_player]))
    #    return np.vstack([value for key, value in f_dict.items()])
    
    #@property
    #def card_pairing_state_array(self):
    #    f_dict = {}
    #    if self.state in ['discard-pick','draw-pick']:
    #        f_dict['CardShowed'] = card_to_multi_hot(self.show)
    #        f_dict['CardPaired'] = card_to_multi_hot(self.pairing_card)
    #    else:
    #        f_dict['CardShowed'] = card_to_multi_hot([])
    #        f_dict['CardPaired'] = card_to_multi_hot([])
    #    f_array = np.vstack([value for key,value in f_dict.items()])
    #    return f_array

    #@property
    #def yaku_status_array(self):
    #    my_hand_bb    = koikoicore.cards_to_bitboard(self.hand[self.turn_player])
    #    board_bb      = koikoicore.cards_to_bitboard(self.field)
    #    my_collect_bb = koikoicore.cards_to_bitboard(self.pile[self.turn_player])
    #    op_collect_bb = koikoicore.cards_to_bitboard(self.pile[self.idle_player])
    #    unseen_bb     = koikoicore.cards_to_bitboard(self.hand[self.idle_player] + self.stock)

    #    # C++側から整数リストを受け取る
    #    features = koikoicore.get_yaku_status_features_by_bitboard(
    #        my_hand_bb, board_bb, my_collect_bb, op_collect_bb, unseen_bb
    #    )
    #    f_array_card_state = np.concatenate(features)
    #    f_array_card_state = np.tile(f_array_card_state, (48, 1)).T
    #    
    #    if not hasattr(KoiKoiRoundState, "_cached_card_key"):
    #        card_dict = {
    #            'Crane':KoiKoiCard.crane, 'Curtain':KoiKoiCard.curtain, 'Moon':KoiKoiCard.moon,
    #            'Rainman':KoiKoiCard.rainman, 'Phoenix':KoiKoiCard.phoenix, 'Sake':KoiKoiCard.sake,
    #            'BoarDeerButterfly':KoiKoiCard.boar_deer_butterfly, 'Seed':KoiKoiCard.seed,
    #            'RedRibbon':KoiKoiCard.red_ribbon, 'BlueRibbon':KoiKoiCard.blue_ribbon,
    #            'RedAndBlue':KoiKoiCard.red_blue_ribbon, 'Ribbon':KoiKoiCard.ribbon, 'Dross':KoiKoiCard.dross
    #        }
    #        KoiKoiRoundState._cached_card_key = np.array([
    #            koikoicore.card_to_multi_hot([list(c) for c in card_set]) for _, card_set in card_dict.items()
    #        ])
    #        
    #    return np.vstack([f_array_card_state, KoiKoiRoundState._cached_card_key])

    
class KoiKoiGameState(KoiKoiGameStateBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # ★ 特徴量ゼロアロケーションのための固定バッファ群
        self._temp_55 = np.zeros(55, dtype=np.float32)
        self._temp_65 = np.zeros(65, dtype=np.float32)
        self._feat_buf_48 = np.zeros((300, 48), dtype=np.float32)
        self._feat_buf_50 = np.zeros((300, 50), dtype=np.float32)
        
        # 普遍的な定数配列の事前キャッシュ＆事前埋め込み
        KoiKoiRoundState._init_caches()
        self._feat_buf_48[150:162, :] = KoiKoiRoundState._card_suit_array
        self._feat_buf_50[150:162, 2:] = KoiKoiRoundState._card_suit_array
        self._feat_buf_48[137:150, :] = KoiKoiRoundState._cached_card_key
        self._feat_buf_50[137:150, 2:] = KoiKoiRoundState._cached_card_key

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
        
    #@property
    #def game_status_array(self):
    #    
    #    def feature_tuple(x, power=[0.5,1,2], weight=[1,1,1]):
    #        return np.abs(float(x)) ** np.array(power) * np.sign(x) * np.array(weight)
    #    
    #    def feature_one_hot(pos, feature_length):
    #        x = np.zeros(feature_length)
    #        x[pos] = 1
    #        return x
    #    
    #    f_dict = {}
    #    
    #    turn_player = self.round_state.turn_player
    #    idle_player = self.round_state.idle_player
    #    
    #    point_diff = self.point[turn_player] - self.point[idle_player]
    #    f_dict['GamePoint'] = feature_tuple(float(point_diff)/2, [0.5,1,1.5], [1,0.5,0.1])

    #    f_dict['MyYakuPoint'] = feature_tuple(
    #        self.round_state.yaku_point(turn_player), [0.5,1,1.5], [1,0.5,0.1])
    #    f_dict['OpYakuPoint'] = feature_tuple(
    #        self.round_state.yaku_point(idle_player), [0.5,1,1.5], [1,0.5,0.1])
    #    
    #    f_dict['Round'] = feature_one_hot(self.round-1, 8)
    #    f_dict['Turn'] = feature_one_hot(self.round_state.turn_16-1, 16)
    #    f_dict['Dealer'] = feature_one_hot(self.round_state.dealer-1, 2)
    #    
    #    f_dict['MyKoiKoiNum'] = feature_tuple(
    #        self.round_state.koikoi_num[turn_player], [1,2], [1,1])
    #    f_dict['OpKoiKoiNum'] = feature_tuple(
    #        self.round_state.koikoi_num[idle_player], [1,2], [1,1])
    #    
    #    f_dict['MyKoiKoi'] = np.array(self.round_state.koikoi[turn_player])
    #    f_dict['OpKoiKoi'] = np.array(self.round_state.koikoi[idle_player])
    #    
    #    f_array = np.concatenate([value for key,value in f_dict.items()])
    #    f_array = f_array[:, np.newaxis] * np.ones((1, 48))
    #    return f_array
    
    #@property
    #def reserve_array(self):
    #    f_array = np.zeros([17,48])
    #    return f_array
        
    @property
    def feature_tensor(self):
        # 毎ステップの配列新規作成 (np.vstack等) を完全に廃止し、
        # 確保済みの巨大バッファに必要な箇所だけを「上書き」します。
        
        is_koikoi = (self.round_state.state == 'koikoi')
        buf = self._feat_buf_50 if is_koikoi else self._feat_buf_48
        out_view = buf[:, 2:] if is_koikoi else buf[:, :]
        
        # 1. reserve (0:17) はゼロのまま
        
        # 2. game_status (17:72)
        self._fill_game_status(self._temp_55)
        out_view[17:72, :] = self._temp_55[:, None] # ブロードキャスト代入 (超高速)
        
        # 3. yaku_status_array (72:137)
        self.round_state._fill_yaku_status(self._temp_65)
        out_view[72:137, :] = self._temp_65[:, None]
        
        # 4. _cached_card_key (137:150) と suit (150:162) は __init__ で書き込み済み
        
        # 5. init_pos (162:165)
        self.round_state._fill_card_init_position(out_view[162:165, :])
        
        # 6. current_pos (165:170)
        self.round_state._fill_card_current_position(out_view[165:170, :])
        
        # 7. pairing (170:172)
        self.round_state._fill_card_pairing_state(out_view[170:172, :])
        
        # 8. log (172:300)
        self.round_state._fill_card_log(out_view[172:300, :])
        
        if is_koikoi:
            # koikoi専用処理 (左2列の f_token の設定)
            buf[:, 0:2] = 0
            buf[0:137, 0:2] = out_view[0:137, 0:2]
            buf[0, 0] = 1
            buf[1, 1] = 1
            
        # 最後にバッファのメモリを共有したまま PyTorch Tensor 化 (コピーなし)
        return torch.from_numpy(buf)
    
    #    f = np.vstack([
    #        self.reserve_array,
    #        self.game_status_array,
    #        self.round_state.yaku_status_array,
    #        self.round_state.card_suit_array,
    #        self.round_state.card_init_position_array,
    #        self.round_state.card_current_position_array,
    #        self.round_state.card_pairing_state_array,
    #        self.round_state.card_log_array
    #    ])

    #    if self.round_state.state == 'koikoi':
    #        f_token = np.zeros([f.shape[0], 2])
    #        f_token[0:137, :] = f[0:137, 0:2]
    #        f_token[0, 0] = 1
    #        f_token[1, 1] = 1
    #        f = np.hstack([f_token, f])
    #        
    #    # 結合が完全に終わった最後のこの瞬間だけ、torch.Tensor に変換する
    #    return torch.Tensor(f)