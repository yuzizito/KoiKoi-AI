# author: shguan3
import sys
from typing import Dict, List, Optional, Tuple

import PySimpleGUI as sg

PATH_CARD = "resource/cardpng/"
PATH_CARD_DARK = "resource/cardpngdark/"
PATH_CARD_LIGHT = "resource/cardpnglight/"
PATH_CARD_SMALL = "resource/cardpngsmall/"
PATH_CARD_SMALL_DARK = "resource/cardpngsmalldark/"
PATH_CARD_SMALL_LIGHT = "resource/cardpngsmalllight/"

_CARD_TYPE_MAP = (
    1, 3, 0, 0, 2, 3, 0, 0, 1, 3, 0, 0, 2, 3, 0, 0,
    2, 3, 0, 0, 2, 3, 0, 0, 2, 3, 0, 0, 1, 2, 0, 0,
    4, 3, 0, 0, 2, 3, 0, 0, 1, 2, 3, 0, 1, 0, 0, 0,
)


def get_card_image_name(card_id: int) -> str:
    if card_id == -1:
        return "null.png"
    suit = card_id // 4 + 1
    rank = card_id % 4 + 1
    return f"{suit}-{rank}.png"


def classify_cards(card_list: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
    brights, seeds, ribbons, dross = [], [], [], []
    for card in card_list:
        c_type = _CARD_TYPE_MAP[card]
        if c_type == 1:
            brights.append(card)
        elif c_type == 2:
            seeds.append(card)
        elif c_type == 3:
            ribbons.append(card)
        elif c_type == 4:
            seeds.append(card)
            dross.append(card)
        else:
            dross.append(card)
    return brights, seeds, ribbons, dross


def init_gui() -> sg.Window:
    sg.theme("Material1")

    layout_scoreboard = [
        [sg.Text("Round", font=("Helvetica", 20), pad=((2, 2), (0, 0)))],
        [sg.Text("12 / 12", font=("Helvetica", 25), pad=((2, 2), (0, 3)), key="RoundCounter")],
        [sg.Text("            ", font=("Helvetica", 12), key="gameNum")],
        [sg.T("")],
        [sg.Text("Player2Name", font=("Helvetica", 20), key="opName")],
        [sg.Text("30 Points", font=("Helvetica", 18), key="opPoints")],
        [sg.Text("            ", font=("Helvetica", 12), key="opDealer")],
        [sg.T("")],
        [sg.T(""), sg.Button(image_filename=f"{PATH_CARD}0-0.png", key="PileCard")],
        [sg.T("")],
        [sg.Text("Player1Name", font=("Helvetica", 20), key="myName")],
        [sg.Text("30 Points", font=("Helvetica", 18), key="myPoints")],
        [sg.Text("            ", font=("Helvetica", 12), key="myDealer")],
        [sg.T("")],
        [sg.T("", size=(3, 1), key=f"PointsRound{i}") for i in (1, 2, 3)],
        [sg.T("", size=(3, 1), key=f"PointsRound{i}") for i in (4, 5, 6)],
        [sg.T("", size=(3, 1), key=f"PointsRound{i}") for i in (7, 8, 9)],
        [sg.T("", size=(3, 1), key=f"PointsRound{i}") for i in (10, 11, 12)],
        [sg.T("")],
        [sg.T("")],
        [sg.Button("Quit", size=(10, 1))],
    ]

    # =========================================================================
    # 獲得札エリア：元コード通りの「正しい横並びリスト」に完全修復
    # =========================================================================
    layout_op_brights = [
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 0)))],
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 8)), key=f"OpBrights{i}") for i in range(1, 6)],
    ]
    layout_op_seeds = [
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 0)), key=f"OpSeeds{i}") for i in range(6, 11)],
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 8)), key=f"OpSeeds{i}") for i in range(1, 6)],
    ]
    layout_op_ribbons = [
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 0)), key=f"OpRibbons{i}") for i in range(6, 11)],
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 8)), key=f"OpRibbons{i}") for i in range(1, 6)],
    ]
    layout_op_dross = [
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 0)), key=f"OpDross{i}") for i in (6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 24, 25, 26)],
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 8)), key=f"OpDross{i}") for i in (1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23)],
    ]

    layout_op_collected = [[
        sg.Column(layout_op_brights), sg.Column(layout_op_seeds),
        sg.Column(layout_op_ribbons), sg.Column(layout_op_dross),
    ]]

    layout_op_hand = [[sg.Button(image_filename=f"{PATH_CARD}0-0.png", key=f"OpHand{i}") for i in range(1, 9)]]
    layout_board_cards = [
        [sg.T("")],
        [sg.Button(image_filename=f"{PATH_CARD}null.png", key=f"Board{i}") for i in (1, 3, 5, 7, 9, 11, 13, 15)],
        [sg.Button(image_filename=f"{PATH_CARD}null.png", key=f"Board{i}") for i in (2, 4, 6, 8, 10, 12, 14, 16)],
        [sg.T("")],
    ]
    layout_my_hand = [[sg.Button(image_filename=f"{PATH_CARD}0-0.png", key=f"MyHand{i}") for i in range(1, 9)]]

    layout_my_brights = [
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (8, 0)), key=f"MyBrights{i}") for i in range(1, 6)],
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 0)))],
    ]
    layout_my_seeds = [
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (8, 0)), key=f"MySeeds{i}") for i in range(1, 6)],
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 0)), key=f"MySeeds{i}") for i in range(6, 11)],
    ]
    layout_my_ribbons = [
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (8, 0)), key=f"MyRibbons{i}") for i in range(1, 6)],
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 0)), key=f"MyRibbons{i}") for i in range(6, 11)],
    ]
    layout_my_dross = [
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (8, 0)), key=f"MyDross{i}") for i in (1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23)],
        [sg.Image(f"{PATH_CARD_SMALL}null.png", pad=((0, 0), (0, 0)), key=f"MyDross{i}") for i in (6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 24, 25, 26)],
    ]

    layout_my_collected = [[
        sg.Column(layout_my_brights), sg.Column(layout_my_seeds),
        sg.Column(layout_my_ribbons), sg.Column(layout_my_dross),
    ]]

    layout_op_yakus = [[sg.Text("", size=(16, 1), key=f"OpYaku{i}"), sg.Text("", size=(2, 1), key=f"OpYakuPt{i}")] for i in range(1, 11)]
    layout_hint = [[sg.Text("", size=(17, 1), key="Hint", text_color="blue")]]
    layout_my_yakus = [[sg.Text("", size=(16, 1), key=f"MyYaku{i}"), sg.Text("", size=(2, 1), key=f"MyYakuPt{i}")] for i in range(1, 11)]

    layout_center = [
        [sg.Column(layout_op_hand + layout_board_cards + layout_my_hand),
         sg.Column(layout_op_yakus + layout_hint + layout_my_yakus)]
    ]
    layout_main = [[sg.Column(layout_scoreboard), sg.Column(layout_op_collected + layout_center + layout_my_collected)]]

    window = sg.Window(
        "Koi-Koi",
        layout_main,
        location=(50, 10),
        grab_anywhere=False,
        finalize=True
    )
    window.bind("<Button-1>", "Any Click")
    for i in range(1, 9):
        window[f"MyHand{i}"].bind("<Enter>", "-Enter")
        window[f"MyHand{i}"].bind("<Leave>", "-Leave")

    return window


def update_game_status_gui(window: sg.Window, game_state) -> sg.Window:
    round_state = game_state.round_state
    rounds, num_round = game_state.round, game_state.round_total
    p1_name, p2_name = game_state.player_name[1], game_state.player_name[2]
    p1_pts, p2_pts = game_state.point[1], game_state.point[2]
    dealer = round_state.dealer
    game_log = game_state.log

    window["RoundCounter"].update(value=f"{rounds} / {num_round}")
    window["gameNum"].update(value="  ")
    window["myName"].update(value=p1_name)
    window["opName"].update(value=p2_name)
    window["myPoints"].update(value=f"{p1_pts} Points")
    window["opPoints"].update(value=f"{p2_pts} Points")

    window["myDealer"].update(value="Dealer" if dealer == 1 else "      ")
    window["opDealer"].update(value="Dealer" if dealer == 2 else "      ")

    for i in range(1, rounds):
        window[f"PointsRound{i}"].update(value=game_log["record"][f"round{i}"]["basic"]["player1RoundPts"])
    return window


def update_turn_player(window: sg.Window, game_state) -> sg.Window:
    p1_name, p2_name = game_state.player_name[1], game_state.player_name[2]
    is_p1 = game_state.round_state.turn_player == 1
    window["myName"].update(value=p1_name, text_color="blue" if is_p1 else "black")
    window["opName"].update(value=p2_name, text_color="black" if is_p1 else "blue")
    return window


def clear_board_gui(window: sg.Window) -> sg.Window:
    for pre in ("My", "Op"):
        for i in range(1, 6): window[f"{pre}Brights{i}"].update(filename=f"{PATH_CARD_SMALL}null.png")
        for i in range(1, 11): window[f"{pre}Seeds{i}"].update(filename=f"{PATH_CARD_SMALL}null.png")
        for i in range(1, 11): window[f"{pre}Ribbons{i}"].update(filename=f"{PATH_CARD_SMALL}null.png")
        for i in range(1, 27): window[f"{pre}Dross{i}"].update(filename=f"{PATH_CARD_SMALL}null.png")
        for i in range(1, 9): window[f"{pre}Hand{i}"].update(image_filename=f"{PATH_CARD}null.png", visible=True)
        for i in range(1, 11):
            window[f"{pre}Yaku{i}"].update(value="")
            window[f"{pre}YakuPt{i}"].update(value="")
    return window


def update_card_and_yaku(window: sg.Window, game_state) -> sg.Window:
    update_hand_cards_gui(window, game_state)
    update_board_cards_gui(window, game_state)
    update_collect_cards_gui(window, game_state)
    update_pile_card_gui(window, game_state)
    update_yaku_gui(window, game_state)
    return window


def update_collect_cards_gui(window: sg.Window, game_state) -> sg.Window:
    round_state = game_state.round_state
    for pre, collected in (("My", round_state.pile[1]), ("Op", round_state.pile[2])):
        brights, seeds, ribbons, dross = classify_cards(collected)
        for i, c in enumerate(brights, 1): window[f"{pre}Brights{i}"].update(filename=PATH_CARD_SMALL + get_card_image_name(c))
        for i, c in enumerate(seeds, 1): window[f"{pre}Seeds{i}"].update(filename=PATH_CARD_SMALL + get_card_image_name(c))
        for i, c in enumerate(ribbons, 1): window[f"{pre}Ribbons{i}"].update(filename=PATH_CARD_SMALL + get_card_image_name(c))
        for i, c in enumerate(dross, 1): window[f"{pre}Dross{i}"].update(filename=PATH_CARD_SMALL + get_card_image_name(c))
    return window


def update_collect_cards_highlight_gui(window: sg.Window, game_state, card: int) -> sg.Window:
    round_state = game_state.round_state
    target_suit = card // 4
    for pre, collected in (("My", round_state.pile[1]), ("Op", round_state.pile[2])):
        brights, seeds, ribbons, dross = classify_cards(collected)
        for i, c in enumerate(brights, 1):
            if c // 4 == target_suit: window[f"{pre}Brights{i}"].update(filename=PATH_CARD_SMALL_DARK + get_card_image_name(c))
        for i, c in enumerate(seeds, 1):
            if c // 4 == target_suit: window[f"{pre}Seeds{i}"].update(filename=PATH_CARD_SMALL_DARK + get_card_image_name(c))
        for i, c in enumerate(ribbons, 1):
            if c // 4 == target_suit: window[f"{pre}Ribbons{i}"].update(filename=PATH_CARD_SMALL_DARK + get_card_image_name(c))
        for i, c in enumerate(dross, 1):
            if c // 4 == target_suit: window[f"{pre}Dross{i}"].update(filename=PATH_CARD_SMALL_DARK + get_card_image_name(c))
    return window


def update_hand_cards_gui(window: sg.Window, game_state) -> sg.Window:
    round_state = game_state.round_state
    my_cards, op_cards = round_state.hand[1], round_state.hand[2]
    board_suits = {c // 4 for c in round_state.field_slot if c != -1}

    for i, c in enumerate(my_cards, 1):
        img_prefix = PATH_CARD if (c // 4) in board_suits else PATH_CARD_DARK
        window[f"MyHand{i}"].update(image_filename=img_prefix + get_card_image_name(c), visible=True)
    for i in range(len(my_cards) + 1, 9): window[f"MyHand{i}"].update(image_filename=f"{PATH_CARD}null.png", visible=True)

    for i in range(1, len(op_cards) + 1): window[f"OpHand{i}"].update(image_filename=f"{PATH_CARD}0-0.png", visible=True)
    for i in range(len(op_cards) + 1, 9): window[f"OpHand{i}"].update(image_filename=f"{PATH_CARD}null.png", visible=True)
    return window


def update_my_discard_card_gui(window: sg.Window, game_state) -> sg.Window:
    round_state = game_state.round_state
    card = round_state.show[0]
    for i, c in enumerate(round_state.hand[1], 1):
        prefix = PATH_CARD if c == card else PATH_CARD_DARK
        window[f"MyHand{i}"].update(image_filename=prefix + get_card_image_name(c))
    for i in range(len(round_state.hand[1]) + 1, 9): window[f"MyHand{i}"].update(image_filename=f"{PATH_CARD}null.png")
    return window


def update_op_discard_card_gui(window: sg.Window, game_state) -> sg.Window:
    card = game_state.round_state.show[0]
    window["OpHand1"].update(image_filename=PATH_CARD + get_card_image_name(card))
    return window


def update_board_cards_gui(window: sg.Window, game_state) -> sg.Window:
    for i, c in enumerate(game_state.round_state.field_slot[:16], 1):
        fname = "null.png" if c == -1 else get_card_image_name(c)
        window[f"Board{i}"].update(image_filename=PATH_CARD + fname)
    return window


def update_board_cards_highlight_gui(window: sg.Window, game_state, card: int) -> sg.Window:
    target_suit = card // 4
    for i, c in enumerate(game_state.round_state.field_slot[:16], 1):
        if c == -1: continue
        prefix = PATH_CARD if c // 4 == target_suit else PATH_CARD_DARK
        window[f"Board{i}"].update(image_filename=prefix + get_card_image_name(c))
    return window


def update_yaku_gui(window: sg.Window, game_state) -> sg.Window:
    round_state = game_state.round_state
    for pre, yakus, pts in (("My", round_state.yaku(1), round_state.yaku_point(1)), ("Op", round_state.yaku(2), round_state.yaku_point(2))):
        if len(yakus) >= 10:
            window[f"{pre}Yaku1"].update(value="Too Many Yakus")
            window[f"{pre}Yaku2"].update(value="--------TOTAL--------")
            window[f"{pre}YakuPt2"].update(value=str(pts))
            continue
        for i, y in enumerate(yakus, 1):
            window[f"{pre}Yaku{i}"].update(value=y[1])
            pt_str = f"x{y[2]-2}" if y[0] == 16 and y[2] >= 4 else str(y[2])
            window[f"{pre}YakuPt{i}"].update(value=pt_str)
        if yakus:
            window[f"{pre}Yaku{len(yakus)+1}"].update(value="--------TOTAL--------")
            window[pre + f"YakuPt{len(yakus)+1}"].update(value=str(pts))
    return window


def update_pile_card_gui(window: sg.Window, game_state) -> sg.Window:
    window["PileCard"].update(image_filename=f"{PATH_CARD}0-0.png")
    return window


def show_pile_card_gui(window: sg.Window, game_state) -> sg.Window:
    card = game_state.round_state.show[0]
    window["PileCard"].update(image_filename=PATH_CARD + get_card_image_name(card))
    return window


def wait_discard_gui(window: sg.Window, game_state) -> Tuple[sg.Window, int]:
    my_hand = game_state.round_state.hand[1]
    window["Hint"].update(value="-> Select a Hand Card")
    while True:
        event, _ = window.read() or (None, None)
        if event in {f"MyHand{i}-Enter" for i in range(1, len(my_hand) + 1)}:
            idx = int(event[6]) - 1
            update_board_cards_highlight_gui(window, game_state, my_hand[idx])
            update_collect_cards_highlight_gui(window, game_state, my_hand[idx])
        elif event in {f"MyHand{i}-Leave" for i in range(1, len(my_hand) + 1)}:
            update_board_cards_gui(window, game_state)
            update_collect_cards_gui(window, game_state)
        elif event in {f"MyHand{i}" for i in range(1, len(my_hand) + 1)}:
            discard_idx = int(event[6]) - 1
            update_collect_cards_gui(window, game_state)
            break
        elif event in ("Quit", None):
            window.Close()
            sys.exit(0)
    return window, my_hand[discard_idx]


def wait_pick_gui(window: sg.Window, game_state) -> Tuple[sg.Window, int]:
    board_cards = game_state.round_state.field_slot
    valid_indices = {i + 1 for i in range(16) if board_cards[i] in game_state.round_state.pairing_card}
    window["Hint"].update(value="-> Select a Field Card")
    update_board_cards_highlight_gui(window, game_state, game_state.round_state.show[0])
    while True:
        event, _ = window.read() or (None, None)
        if event in {f"Board{i}" for i in valid_indices}:
            pick_idx = int(event[5:]) - 1
            break
        elif event in ("Quit", None):
            window.Close()
            sys.exit(0)
    return window, board_cards[pick_idx]


def wait_any_click(window: sg.Window) -> sg.Window:
    window["Hint"].update(value="-> Click to Continue")
    while True:
        event, _ = window.read() or (None, None)
        if event in ("Quit", None):
            window.Close()
            sys.exit(0)
        elif event == "Any Click":
            break
    return window


def wait_koikoi(window: sg.Window) -> Tuple[sg.Window, bool]:
    window["Hint"].update(value="-> Koi-Koi?")
    res = sg.popup_yes_no("Koi-Koi?")
    if res is None:
        window.Close()
        sys.exit(0)
    return window, (res == "Yes")


def show_op_koikoi(window: sg.Window, game_state, action: Optional[bool]) -> sg.Window:
    # 役ができてアクション(True/False)が確定した時だけポップアップを出す
    if action is not None:
        p_name = game_state.player_name[game_state.round_state.turn_player]
        msg = f"{p_name}: {'Koi-Koi' if action else 'Stop'}"
        sg.popup(msg, title="Koi-Koi")
    return window


def show_round_over_gui(window: sg.Window, game_state) -> sg.Window:
    rs = game_state.round_state
    p1, p2 = game_state.player_name[1], game_state.player_name[2]
    window["Hint"].update(value="-> Round Over")
    sg.popup(f"{p1}: {rs.round_point[1]}     {p2}: {rs.round_point[2]}", title="Round Over")
    return window


def show_game_over_gui(window: sg.Window, game_state) -> sg.Window:
    p1, p2 = game_state.player_name[1], game_state.player_name[2]
    window["Hint"].update(value="-> Game Over")
    sg.popup(f"{p1}: {game_state.point[1]}     {p2}: {game_state.point[2]}", title="Game Over")
    return window


def close_window(window: sg.Window) -> None:
    window.Close()


# 後方互換性エイリアス
CardClassify = classify_cards
InitGUI = init_gui
UpdateGameStatusGUI = update_game_status_gui
UpdateTurnPlayer = update_turn_player
ClearBoardGUI = clear_board_gui
UpdateCardAndYaku = update_card_and_yaku
UpdateCollectCardsGUI = update_collect_cards_gui
UpdateCollectCardsHighlightGUI = update_collect_cards_highlight_gui
UpdateHandCardsGUI = update_hand_cards_gui
UpdateMyDiscardCardGUI = update_my_discard_card_gui
UpdateOpDiscardCardGUI = update_op_discard_card_gui
UpdateBoardCardsGUI = update_board_cards_gui
UpdateBoardCardsHighlightGUI = update_board_cards_highlight_gui
UpdateYakuGUI = update_yaku_gui
UpdatePileCardGUI = update_pile_card_gui
ShowPileCardGUI = show_pile_card_gui
WaitDiscardGUI = wait_discard_gui
WaitPickGUI = wait_pick_gui
WaitAnyClick = wait_any_click
WaitKoiKoi = wait_koikoi
ShowOpKoiKoi = show_op_koikoi
ShowRoundOverGUI = show_round_over_gui
ShowGameOverGUI = show_game_over_gui
Close = close_window