#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch  # LibTorchのDLL群を先にシステムにロードさせる
import PySimpleGUI as sg
import koikoicore

# リソースパス
IMG_DIR = 'resource/cardpng/'
IMG_DARK_DIR = 'resource/cardpngdark/'
IMG_SMALL_DIR = 'resource/cardpngsmall/'

# 静的属性ビットマスク
BRIGHT_MASK = 0x1101000101
SEED_MASK = 0x2436213010
RIBBON_MASK = 0x460c222222
DROSS_MASK = 0xffffffffffffffff & ~(BRIGHT_MASK | SEED_MASK | RIBBON_MASK)

def get_img(card_id, size='normal'):
    if card_id == -1:
        return IMG_SMALL_DIR + 'null.png' if size == 'small' else IMG_DIR + 'null.png'
    name = f"{card_id // 4 + 1}-{card_id % 4 + 1}.png"
    return IMG_SMALL_DIR + name if size == 'small' else IMG_DIR + name

def json_to_id(card_list):
    return [(c[0] - 1) * 4 + (c[1] - 1) if c != [0, 0] else -1 for c in card_list]

def update_ui_collection(window, prefix, pile_list):
    """獲得札を4属性スロットへソートしてUI反映"""
    ids = [c for c in pile_list if c != -1]
    bb = sum(1 << c for c in ids)
    b_bb, s_bb, r_bb, d_bb = bb & BRIGHT_MASK, bb & SEED_MASK, bb & RIBBON_MASK, bb & DROSS_MASK
    
    def fill_slots(key_base, current_bb, max_slots):
        idx = 1
        while current_bb and idx <= max_slots:
            c = koikoicore.ctz64(current_bb) if hasattr(koikoicore, 'ctz64') else int(current_bb & -current_bb).bit_length() - 1
            window[f"{prefix}{key_base}{idx}"].update(get_img(c, size='small'))
            current_bb &= current_bb - 1
            idx += 1
        for i in range(idx, max_slots + 1):
            window[f"{prefix}{key_base}{i}"].update(get_img(-1, size='small'))

    fill_slots('Brights', b_bb, 5)
    fill_slots('Seeds', s_bb, 10)
    fill_slots('Ribbons', r_bb, 10)
    fill_slots('Dross', d_bb, 26)

def create_replay_window():
    """洗練されたコントロールパネルを持つGUIレイアウト"""
    layout_score = [
        [sg.Text('Round', font=('Helvetica', 16)), sg.Text('1 / 6', font=('Helvetica', 20, 'bold'), key='RoundCounter')],
        [sg.Text('Step', font=('Helvetica', 12)), sg.Text('0 / 0', font=('Helvetica', 12), key='StepCounter')],
        [sg.T('')],
        [sg.Text('Player 2 (Op)', font=('Helvetica', 14, 'bold'), key='opName')],
        [sg.Text('30 Points', font=('Helvetica', 12), key='opPoints')],
        [sg.Text('', font=('Helvetica', 11, 'italic'), key='opDealer', text_color='blue')],
        [sg.T('')],
        [sg.Text('▼ 山札めくり', font=('Helvetica', 10))],
        [sg.Button(image_filename=get_img(-1), key='PileCard')],
        [sg.T('')],
        [sg.Text('Player 1 (My)', font=('Helvetica', 14, 'bold'), key='myName')],
        [sg.Text('30 Points', font=('Helvetica', 12), key='myPoints')],
        [sg.Text('', font=('Helvetica', 11, 'italic'), key='myDealer', text_color='blue')],
        [sg.T('')],
        # コントロール部
        [sg.Button('◀ 1手戻す (Prev)', size=(14, 2), key='BtnPrev', button_color=('white', '#34495e'))],
        [sg.Button('1手進める (Next) ▶', size=(14, 2), key='BtnNext', button_color=('white', '#2c3e50'))],
        [sg.T('')],
        [sg.Button('終了 (Quit)', size=(14, 1), key='Quit', button_color=('white', '#c0392b'))]
    ]

    def collection_column(prefix):
        return sg.Column([
            [sg.Image(get_img(-1, 'small'), key=f"{prefix}Brights{i}") for i in range(1, 6)],
            [sg.Image(get_img(-1, 'small'), key=f"{prefix}Seeds{i}") for i in range(1, 11)],
            [sg.Image(get_img(-1, 'small'), key=f"{prefix}Ribbons{i}") for i in range(1, 11)],
            [sg.Image(get_img(-1, 'small'), key=f"{prefix}Dross{i}") for i in range(1, 27)]
        ])

    layout_board = [
        [sg.Button(image_filename=get_img(-1), key=f"OpHand{i}") for i in range(1, 9)],
        [sg.T('')],
        [sg.Button(image_filename=get_img(-1), key=f"Board{i}") for i in range(1, 17)],
        [sg.T('')],
        [sg.Button(image_filename=get_img(-1), key=f"MyHand{i}") for i in range(1, 9)]
    ]

    layout_yaku = [
        [sg.Text('【確定役リスト】', font=('Helvetica', 11, 'bold'))],
        *[[sg.Text('', size=(24, 1), key=f"YakuLine{i}", font=('Helvetica', 10))] for i in range(1, 11)]
    ]

    layout = [[
        sg.Column(layout_score, element_justification='center'),
        sg.Column([
            [sg.Text('--- Player 2 (相手) 獲得札 ---')], [collection_column('Op')],
            [sg.Column(layout_board), sg.Column(layout_yaku, vertical_alignment='top')],
            [sg.Text('--- Player 1 (自分) 獲得札 ---')], [collection_column('My')]
        ])
    ]]
    return sg.Window('Koi-Koi Record Replay V2', layout, finalize=True)

def build_timeline(game_log):
    """JSONの歴史記録をフラットなタイムラインステップに分解する"""
    timeline = []
    p1_pts, p2_pts = game_log['info']['player1InitPts'], game_log['info']['player2InitPts']
    
    for r_idx in range(1, 9):
        r_key = f'round{r_idx}'
        if r_key not in game_log['record']: break
        r_data = game_log['record'][r_key]
        
        # ラウンド開始ステップ
        timeline.append({
            'type': 'round_start', 'round': r_idx, 'p1_pts': p1_pts, 'p2_pts': p2_pts, 'data': r_data
        })
        
        for t_idx in range(1, 17):
            t_key = f'turn{t_idx}'
            if t_key not in r_data: break
            t_data = r_data[t_key]
            
            # フェーズ1: 手札プレイ
            timeline.append({
                'type': 'phase_hand', 'round': r_idx, 'turn': t_idx, 'data': t_data
            })
            # フェーズ2: 山札めくりプレイ
            timeline.append({
                'type': 'phase_deck', 'round': r_idx, 'turn': t_idx, 'data': t_data
            })
            
        p1_pts += r_data['basic']['player1RoundPts']
        p2_pts += r_data['basic']['player2RoundPts']
        
    return timeline

def render_state(window, timeline, step_idx):
    """現在のステップまで最初から盤面を高速に再現し表示を更新"""
    my_hand, op_hand, board = [], [], [-1] * 16
    my_pile, op_pile = [], []
    my_koikoi, op_koikoi = 0, 0
    p1_pts, p2_pts, dealer, r_num = 0, 0, 1, 1
    
    hl_cards = set()
    hl_pile_card = -1

    # 現在のステップまで順に盤面を早送り演算
    for idx in range(step_idx + 1):
        step = timeline[idx]
        r_num = step['round']
        
        if step['type'] == 'round_start':
            p1_pts, p2_pts = step['p1_pts'], step['p2_pts']
            dealer = step['data']['basic']['Dealer']
            my_hand = json_to_id(step['data']['basic']['initHand1'])
            op_hand = json_to_id(step['data']['basic']['initHand2'])
            board = json_to_id(step['data']['basic']['initBoard']) + [-1] * 8
            my_pile, op_pile = [], []
            my_koikoi, op_koikoi = 0, 0
            hl_cards.clear()
            hl_pile_card = -1
            
        elif step['type'] == 'phase_hand':
            hl_pile_card = -1
            t_data = step['data']
            p_turn = t_data['playerInTurn']
            discard = json_to_id([t_data['discardCard']])[0]
            collect = json_to_id(t_data['collectCard'])
            
            hl_cards = set([discard] + collect)

            if p_turn == 1:
                if discard in my_hand: my_hand.remove(discard)
                my_pile.extend(collect)
            else:
                if discard in op_hand: op_hand.remove(discard)
                op_pile.extend(collect)

            if not collect:
                if -1 in board: board[board.index(-1)] = discard
            else:
                for c in collect:
                    if c != discard and c in board: board[board.index(c)] = -1
                    
        elif step['type'] == 'phase_deck':
            t_data = step['data']
            p_turn = t_data['playerInTurn']
            draw = json_to_id([t_data['drawCard']])[0]
            collect = json_to_id(t_data['collectCard2'])
            
            hl_pile_card = draw
            hl_cards = set([draw] + collect)

            if p_turn == 1:
                my_pile.extend(collect)
            else:
                op_pile.extend(collect)

            if not collect:
                if -1 in board: board[board.index(-1)] = draw
            else:
                for c in collect:
                    if c != draw and c in board: board[board.index(c)] = -1

            if t_data.get('isKoiKoi') is True:
                if p_turn == 1: my_koikoi += 1
                else: op_koikoi += 1

    # --- UI要素の一斉更新描画 ---
    window['RoundCounter'].update(f"{r_num} / {timeline[-1]['round']}")
    window['StepCounter'].update(f"{step_idx} / {len(timeline) - 1}")
    window['myPoints'].update(f"{p1_pts} Points")
    window['opPoints'].update(f"{p2_pts} Points")
    window['myDealer'].update('Dealer' if dealer == 1 else '')
    window['opDealer'].update('Dealer' if dealer == 2 else '')

    # 手札・場札の描画とハイライト
    for i in range(8):
        c_my = my_hand[i] if i < len(my_hand) else -1
        bg_my = '#e74c3c' if c_my in hl_cards and c_my != -1 else '#ecf0f1'
        window[f'MyHand{i+1}'].update(image_filename=get_img(c_my), button_color=('white', bg_my))
        
        c_op = op_hand[i] if i < len(op_hand) else -1
        bg_op = '#e74c3c' if c_op in hl_cards and c_op != -1 else '#ecf0f1'
        window[f'OpHand{i+1}'].update(image_filename=get_img(c_op), button_color=('white', bg_op))

    for i in range(16):
        c_brd = board[i]
        bg_brd = '#e74c3c' if c_brd in hl_cards and c_brd != -1 else '#ecf0f1'
        window[f'Board{i+1}'].update(image_filename=get_img(c_brd), button_color=('white', bg_brd))

    window['PileCard'].update(image_filename=get_img(hl_pile_card), button_color=('white', '#e74c3c' if hl_pile_card != -1 else '#ecf0f1'))

    update_ui_collection(window, 'My', my_pile)
    update_ui_collection(window, 'Op', op_pile)
    
    # 役テキストの更新
    my_yaku = koikoicore.evaluate_yaku_by_bitboard(sum(1 << c for c in my_pile if c != -1), my_koikoi)
    for i in range(10):
        text = f"• {my_yaku[i][1]} ({my_yaku[i][2]}文)" if i < len(my_yaku) else ""
        window[f"YakuLine{i+1}"].update(text, text_color='#2980b9')

def main():
    filename = 'gamerecords_dataset/1.json'
    if not os.path.isfile(filename):
        print(f"棋譜ファイルが見つかりません: {filename}")
        sys.exit(1)

    with open(filename, 'r') as f:
        game_log = json.load(f)

    # タイムラインの構築
    timeline = build_timeline(game_log)
    current_step = 0

    window = create_replay_window()
    
    # 初期状態を描画
    render_state(window, timeline, current_step)

    # イベントループ
    while True:
        event, _ = window.read()
        
        if event in (None, 'Quit'):
            break
            
        elif event == 'BtnNext':
            if current_step < len(timeline) - 1:
                current_step += 1
                render_state(window, timeline, current_step)
                
                step_data = timeline[current_step]
                if step_data['type'] == 'phase_deck':
                    is_kk = step_data['data'].get('isKoiKoi')
                    p = step_data['data']['playerInTurn']
                    if is_kk is True:
                        sg.popup(f"Player {p}: こいこい！", title='Koi-Koi', button_color=('white', '#2980b9'))
                    elif is_kk == False:
                        sg.popup(f"Player {p}: 勝負！", title='Stop Game', button_color=('white', '#e74c3c'))
            else:
                sg.popup("最期の手です（棋譜の終端）", title='Notice')
                
        elif event == 'BtnPrev':
            if current_step > 0:
                current_step -= 1
                render_state(window, timeline, current_step)
            else:
                sg.popup("最初の手です（棋譜の始端）", title='Notice')

    window.close()

if __name__ == '__main__':
    main()