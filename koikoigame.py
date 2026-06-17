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

#card_to_multi_hot = koikoicore.card_to_multi_hot

#class KoiKoiCard():
#    crane = {(1,1)}
#    curtain = {(3,1)}
#    moon = {(8,1)}
#    rainman = {(11,1)}
#    phoenix = {(12,1)}
#    sake = {(9,1)}
#    
#    light  = {(1,1),(3,1),(8,1),(11,1),(12,1)}
#    seed   = {(2,1),(4,1),(5,1),(6,1),(7,1),(8,2),(9,1),(10,1),(11,2)}
#    ribbon = {(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),(9,2),(10,2),(11,3)}
#    dross  = {(1,3),(1,4),(2,3),(2,4),(3,3),(3,4),(4,3),(4,4),(5,3),(5,4),(6,3),(6,4),(7,3),
#              (7,4),(8,3),(8,4),(9,3),(9,4),(10,3),(10,4),(11,4),(12,2),(12,3),(12,4),(9,1)}
#            
#    boar_deer_butterfly = {(6,1),(7,1),(10,1)}
#    flower_sake = {(3,1),(9,1)}
#    moon_sake = {(8,1),(9,1)}
#    red_ribbon = {(1,2),(2,2),(3,2)}
#    blue_ribbon = {(6,2),(9,2),(10,2)}
#    red_blue_ribbon = {(1,2),(2,2),(3,2),(6,2),(9,2),(10,2)}
#
#YAKU_ID_TO_NAME = {
#    1: 'Five Lights', 2: 'Four Lights', 3: 'Rainy Four Lights', 4: 'Three Lights',
#    5: 'Boar-Deer-Butterfly', 6: 'Flower Viewing Sake', 7: 'Flower Viewing Sake',
#    8: 'Moon Viewing Sake', 9: 'Moon Viewing Sake', 10: 'Tane',
#    11: 'Red & Blue Ribbons', 12: 'Red Ribbons', 13: 'Blue Ribbons', 14: 'Tan',
#    15: 'Kasu', 16: 'Koi-Koi'
#}

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

#    def yaku(self, player):
#        pile_bb = koikoicore.cards_to_bitboard(self.pile[player])
#        yaku_ids = koikoicore.evaluate_yaku_id_by_bitboard(pile_bb, self.koikoi_num[player])
#        return [(y_id, YAKU_ID_TO_NAME.get(y_id, 'Unknown'), pt) for y_id, pt in yaku_ids]
    
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
    
    #def __call__(self, view=None):
    #    view = self.turn_player if view == None else view
    #    op_view = 3-view
    #    pile = set([tuple(card) for card in self.pile[view]])
    #    op_pile = set([tuple(card) for card in self.pile[op_view]])
    #    
    #    print('Turn: '+str(self.turn_8)+',  State: '+self.state)
    #    print('-----------------------------------------------')
    #    print('Opponent\'s Yaku:')
    #    print([[yaku[1],yaku[2]] for yaku in self.yaku(op_view)])
    #    print('Total Point: '+str(self.yaku_point(op_view)))
    #    print('-----------------------------------------------')
    #    print('Opponent\'s Pile:')
    #    print('Light: '+ str(list(op_pile & KoiKoiCard.light)))
    #    print('Seed: '+ str(list(op_pile & KoiKoiCard.seed)))
    #    print('Ribbon: '+ str(list(op_pile & KoiKoiCard.ribbon)))
    #    print('Dross: '+ str(list(op_pile & KoiKoiCard.dross)))
    #    print('-----------------------------------------------')
   #     print('Opponent\'s Hand:')
   #     print([[0,0] for card in self.hand[op_view]])
   #     print('-----------------------------------------------')
   #     print('Field:')
   #     print(self.field)
   #     print('-----------------------------------------------')
   #     print('Your Hand:')
   #     print(self.hand[view])
   #     print('-----------------------------------------------')
   #     print('Your Pile:')
   #     print('Light: '+ str(list(pile & KoiKoiCard.light)))
   #     print('Seed: '+ str(list(pile & KoiKoiCard.seed)))
   #     print('Ribbon: '+ str(list(pile & KoiKoiCard.ribbon)))
   #     print('Dross: '+ str(list(pile & KoiKoiCard.dross)))
   #     print('-----------------------------------------------')
   #     print('Your Yaku:')
   #     print([[yaku[1],yaku[2]] for yaku in self.yaku(view)])
   #     print('Total Point: '+str(self.yaku_point(view)))
   #     print('-----------------------------------------------')
   #     
   #     if view != self.turn_player:
   #         print('Opponent\'s turn, waiting action...')
   #         return
   #         
   #     if self.state == 'discard':
   #         print('Use discard(card) to discard from hand.')
   #     elif self.state == 'discard-pick':
   #         print('Discard: '+str(self.show[0]))
   #         print('Pairing: '+str(self.pairing_card))
   #         if self.wait_action:
   #             print('Use discard_pick(card) to pick a pairing field card.')
   #         else:
   #             print('Use discard_pick() to continue.')
   #     elif self.state == 'draw':
   #         print('Use draw() to draw from stock.')
   #     elif self.state == 'draw-pick':
   #         print('Draw: '+str(self.show[0]))
   #         print('Pairing: '+str(self.pairing_card))
   #         if self.wait_action:
   #             print('Use draw_pick(card) to pick a pairing field card.')
   #         else:
   #             print('Use draw_pick() to continue.')
   #     elif self.state == 'koikoi':
#            if self.wait_action:
#                print('Use claim_koikoi(bool) to koikoi or stop.')
#            else:
#                print('Use claim_koikoi() to continue.')        
#        elif self.state == 'round-over':
#            print('Round Over')
#            print('Round Point: You '+str(self.round_point[view])+\
#                  ', Opponent '+str(self.round_point[op_view]))
#        return
        
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

#    def __call__(self):
#        print('-----------------------------------------------')
#        print('Round: '+str(self.round)+' / '+str(self.round_total))
#        print(self.log['info']['player1Name']+': '+str(self.point[1])+', '+\
#              self.log['info']['player2Name']+': '+str(self.point[2]))
#        if self.game_over:
#            print('Game Over')
#        print('-----------------------------------------------')
#        return
       
class KoiKoiRoundState(KoiKoiRoundStateBase):
    _cached_card_key = None
    _card_suit_array = None
    
    @classmethod
    def _init_caches(cls):
        if cls._cached_card_key is None:
            #card_dict = {
            #    'Crane':KoiKoiCard.crane, 'Curtain':KoiKoiCard.curtain, 'Moon':KoiKoiCard.moon,
            #    'Rainman':KoiKoiCard.rainman, 'Phoenix':KoiKoiCard.phoenix, 'Sake':KoiKoiCard.sake,
            #    'BoarDeerButterfly':KoiKoiCard.boar_deer_butterfly, 'Seed':KoiKoiCard.seed,
            #    'RedRibbon':KoiKoiCard.red_ribbon, 'BlueRibbon':KoiKoiCard.blue_ribbon,
            #    'RedAndBlue':KoiKoiCard.red_blue_ribbon, 'Ribbon':KoiKoiCard.ribbon, 'Dross':KoiKoiCard.dross
            #}
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
            ], dtype=np.float16)
            
        if cls._card_suit_array is None:
            arr = np.zeros([12, 48], dtype=np.float16)
            for ii in range(12):
                arr[ii, 4*ii:4*ii+4] = 1.0
            cls._card_suit_array = arr
    
    def __init__(self, dealer=None):
        super().__init__(dealer)
        # ★辞書を廃止し、3D NumPyバッファで履歴を全管理（16ターン × 8状態 × 48カード）
        self._card_log_buf = np.zeros((17, 8, 48), dtype=np.float16)
    
    def _to_multi_hot(self, cards):
        arr = np.zeros(48, dtype=np.float16)
        if cards:
            arr[cards] = 1.0
        return arr
    
    def discard(self, card=None): 
        out = super().discard(card)
        idx = 0 if self.pairing_card else 1
        self._card_log_buf[self.turn_16, idx, :] = self._to_multi_hot(self.show)
        return out
        
    def discard_pick(self, card=None):
        out = super().discard_pick(card)
        if self.collect:
            p_collect = self._to_multi_hot(self.field_collect)
            p_discard = self._to_multi_hot(self.log[f'turn{self.turn_16}']['pairCard'])
            self._card_log_buf[self.turn_16, 2, :] = p_collect
            self._card_log_buf[self.turn_16, 3, :] = p_discard - p_collect
        return out
        
    def draw(self, card=None):
        out = super().draw(card)
        idx = 4 if self.pairing_card else 5
        self._card_log_buf[self.turn_16, idx, :] = self._to_multi_hot(self.show)
        return out
    
    def draw_pick(self, card=None):
        out = super().draw_pick(card)
        if self.collect:
            p_collect = self._to_multi_hot(self.field_collect)
            p_discard = self._to_multi_hot(self.log[f'turn{self.turn_16}']['pairCard2'])
            self._card_log_buf[self.turn_16, 6, :] = p_collect
            self._card_log_buf[self.turn_16, 7, :] = p_discard - p_collect
        return out

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
        mask = np.zeros(48, dtype=np.bool_)
        if self.state == 'discard':
            if self.hand[self.turn_player]:
                mask[self.hand[self.turn_player]] = True
        elif self.state in ['discard-pick', 'draw-pick']:
            if self.pairing_card:
                mask[self.pairing_card] = True
        elif self.state == 'koikoi':
            return np.array([True, True], dtype=np.bool_)
        return mask
    
#    def __write_card_log_array(self,state):
#        def card_log_turn_dict():
#            cardLogTurn = {}
#            cardLogTurn['CardDiscardedAndPaired'] = np.zeros(48)
#            cardLogTurn['CardDiscardedAndUnpaired'] = np.zeros(48)
#            cardLogTurn['CardPairedByDiscardCollect'] = np.zeros(48)
#            cardLogTurn['CardPairedByDiscardUncollect'] = np.zeros(48)
#            cardLogTurn['CardDrawnAndPaired'] = np.zeros(48)
#            cardLogTurn['CardDrawnAndUnpaired'] = np.zeros(48)
#            cardLogTurn['CardPairedByDrawnCollect'] = np.zeros(48)
#            cardLogTurn['CardPairedByDrawnUncollect'] = np.zeros(48)
#            return cardLogTurn
#    
#        turn = self.turn_16
#        if state == 'init':
#            for ii in range(1,17):
#                self.card_log_dict[ii] = card_log_turn_dict()
#                
#        elif state == 'discard':
#            if self.pairing_card == []:
#                self.card_log_dict[turn]['CardDiscardedAndUnpaired'] \
#                    = np.array(card_to_multi_hot(self.show))
#            else:
#                self.card_log_dict[turn]['CardDiscardedAndPaired'] \
#                    = np.array(card_to_multi_hot(self.show)) 
#            
#        elif state == 'discard-pick':
#            if self.collect != []:
#                pair_by_discard_collect = np.array(card_to_multi_hot(self.field_collect))
#                pair_by_discard = np.array(card_to_multi_hot(self.log[f'turn{turn}']['pairCard']))
#                self.card_log_dict[turn]['CardPairedByDiscardCollect'] \
#                    = pair_by_discard_collect
#                self.card_log_dict[turn]['CardPairedByDiscardUncollect'] \
#                    = pair_by_discard - pair_by_discard_collect
#            
#        elif state == 'draw':
#            if self.pairing_card == []:
#                self.card_log_dict[turn]['CardDrawnAndUnpaired'] \
#                    = np.array(card_to_multi_hot(self.show))
#            else:
#                self.card_log_dict[turn]['CardDrawnAndPaired'] \
#                    = np.array(card_to_multi_hot(self.show))
#            
#        elif state == 'draw-pick':
#            if self.collect != []:
#                pair_by_discard_collect = np.array(card_to_multi_hot(self.field_collect))
#                pair_by_discard = np.array(card_to_multi_hot(self.log[f'turn{turn}']['pairCard2']))
#                self.card_log_dict[turn]['CardPairedByDrawnCollect'] \
#                    = pair_by_discard_collect
#                self.card_log_dict[turn]['CardPairedByDrawnUncollect'] \
#                    = pair_by_discard - pair_by_discard_collect
#        return
#    
#    @property
#    def action_mask(self):
#        if self.state == 'discard':
#            mask = card_to_multi_hot(self.hand[self.turn_player])
#        elif self.state in ['discard-pick', 'draw-pick']:
#            mask = card_to_multi_hot(self.pairing_card)
#        elif self.state == 'koikoi':
#            mask = [1,1]
#        else:
#            mask = []
#        return np.array(mask)
    
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
    
    _idx_map = {(s, r): (s-1)*4 + (r-1) for s in range(1, 13) for r in range(1, 5)}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temp_55 = np.zeros(55, dtype=np.float16)
        self._feat_buf_48 = np.zeros((300, 48), dtype=np.float16)
        self._feat_buf_50 = np.zeros((300, 50), dtype=np.float16)
        
        KoiKoiRoundState._init_caches()
        self._feat_buf_48[150:162, :] = KoiKoiRoundState._card_suit_array
        self._feat_buf_50[150:162, 2:] = KoiKoiRoundState._card_suit_array
        self._feat_buf_48[137:150, :] = KoiKoiRoundState._cached_card_key
        self._feat_buf_50[137:150, 2:] = KoiKoiRoundState._cached_card_key

    @property
    def feature_np(self):
        is_koikoi = (self.round_state.state == 'koikoi')
        buf = self._feat_buf_50 if is_koikoi else self._feat_buf_48
        out_view = buf[:, 2:] if is_koikoi else buf[:, :]
        rs = self.round_state
        
        turn_p = rs.turn_player
        idle_p = rs.idle_player
        
        p_diff = float(self.point[turn_p] - self.point[idle_p]) / 2.0
        sgn0 = 1.0 if p_diff > 0 else (-1.0 if p_diff < 0 else 0.0)
        ax0 = abs(p_diff)
        self._temp_55[0:3] = (ax0**0.5 * sgn0, ax0 * sgn0 * 0.5, ax0**1.5 * sgn0 * 0.1)
        
        yp_t = float(rs.yaku_point(turn_p))
        sgn1 = 1.0 if yp_t > 0 else (-1.0 if yp_t < 0 else 0.0)
        ax1 = abs(yp_t)
        self._temp_55[3:6] = (ax1**0.5 * sgn1, ax1 * sgn1 * 0.5, ax1**1.5 * sgn1 * 0.1)
        
        yp_i = float(rs.yaku_point(idle_p))
        sgn2 = 1.0 if yp_i > 0 else (-1.0 if yp_i < 0 else 0.0)
        ax2 = abs(yp_i)
        self._temp_55[6:9] = (ax2**0.5 * sgn2, ax2 * sgn2 * 0.5, ax2**1.5 * sgn2 * 0.1)
        
        self._temp_55[9:35] = 0
        self._temp_55[9 + self.round - 1] = 1
        self._temp_55[17 + rs.turn_16 - 1] = 1
        self._temp_55[33 + rs.dealer - 1] = 1
        
        k1, k2 = float(rs.koikoi_num[turn_p]), float(rs.koikoi_num[idle_p])
        s1 = 1.0 if k1 > 0 else (-1.0 if k1 < 0 else 0.0)
        s2 = 1.0 if k2 > 0 else (-1.0 if k2 < 0 else 0.0)
        ak1, ak2 = abs(k1), abs(k2)
        self._temp_55[35:37] = (ak1 * s1, (ak1**2) * s1)
        self._temp_55[37:39] = (ak2 * s2, (ak2**2) * s2)
        self._temp_55[39:47] = rs.koikoi[turn_p]
        self._temp_55[47:55] = rs.koikoi[idle_p]
        out_view[17:72, :] = self._temp_55[:, None]
        
        # feature_np() 内に追加する軽量な変換関数
        def to_bb(cards):
            bb = 0
            for c in cards:
                bb |= (1 << c)
            return bb
        
        # 重かった koikoicore.cards_to_bitboard() を Pythonの軽量ループに置き換え
        mh = to_bb(rs.hand[turn_p])
        b  = to_bb(rs.field)
        mc = to_bb(rs.pile[turn_p])
        oc = to_bb(rs.pile[idle_p])
        u  = to_bb(rs.hand[idle_p] + rs.stock)
        # Pythonの int は C++の uint64_t へキャストなし（ゼロコスト）で渡る
        out_view[72:137, :] = koikoicore.get_yaku_status_features_np(mh, b, mc, oc, u)[:, None]
        
        out_view[162:172, :] = 0  # 一括クリア
        
        # 内包表記と辞書(_idx_map)を全廃し、整数リストをそのままインデックスとして渡す
        if rs.hand[turn_p]: out_view[162, rs.hand[turn_p]] = 1.0
        if rs.log['basic']['initBoard']: out_view[163, rs.log['basic']['initBoard']] = 1.0
        if rs.unseen_card[turn_p]: out_view[164, rs.unseen_card[turn_p]] = 1.0
        
        out_view[165, :] = out_view[162, :]
        if rs.pile[turn_p]: out_view[166, rs.pile[turn_p]] = 1.0
        if rs.field: out_view[167, rs.field] = 1.0
        if rs.pile[idle_p]: out_view[168, rs.pile[idle_p]] = 1.0
        out_view[169, :] = out_view[164, :]
        
        if rs.state in ['discard-pick', 'draw-pick']:
            if rs.show: out_view[170, rs.show] = 1.0
            if rs.pairing_card: out_view[171, rs.pairing_card] = 1.0
        
        order = KoiKoiGameState._order_cache[rs.turn_16]
        out_view[172:300, :] = rs._card_log_buf[order].reshape(128, 48)
        
        if is_koikoi:
            buf[:, 0:2] = 0
            buf[0:137, 0:2] = out_view[0:137, 0:2]
            buf[0, 0] = 1
            buf[1, 1] = 1
        
        result = buf.copy()
        
        return buf.copy()

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
        return torch.from_numpy(self.feature_np)
    
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