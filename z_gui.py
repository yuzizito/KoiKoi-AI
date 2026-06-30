# author: shguan3
import sys
from typing import List, Tuple
import PySimpleGUI as sg

P_CARD = "resource/cardpng/"
P_DARK = "resource/cardpngdark/"
P_SMALL = "resource/cardpngsmall/"
P_S_DARK = "resource/cardpngsmalldark/"

# 1次元ID(0~47)対応：カード属性マップ (1:光, 2:タネ, 3:短冊, 4:タネ兼カス, 0:カス)
_CARD_TYPE_MAP = (
    1, 3, 0, 0, 2, 3, 0, 0, 1, 3, 0, 0, 2, 3, 0, 0,
    2, 3, 0, 0, 2, 3, 0, 0, 2, 3, 0, 0, 1, 2, 0, 0,
    4, 3, 0, 0, 2, 3, 0, 0, 1, 2, 3, 0, 1, 0, 0, 0,
)


def get_img(c: int) -> str:
    return "null.png" if c == -1 else f"{c // 4 + 1}-{c % 4 + 1}.png"


def classify(cards: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
    b, s, r, d = [], [], [], []
    for c in cards:
        t = _CARD_TYPE_MAP[c]
        if t == 1: b.append(c)
        if t in (2, 4): s.append(c)
        if t == 3: r.append(c)
        if t in (0, 4): d.append(c)  # 菊に盃(タイプ4)はタネとカスの両方に分配
    return b, s, r, d


def get_yakus(pile: List[int], koi_cnt: int) -> List[Tuple[str, int]]:
    """C++に依存しない純粋な1次元ID役判定ロジック"""
    s = set(pile)
    b_l, s_l, r_l, d_l = classify(pile)
    res = []

    n_b, rain = len(b_l), 40 in s
    if n_b == 5: res.append(("Five Lights", 10))
    elif n_b == 4: res.append(("Rainy Four Lights" if rain else "Four Lights", 7 if rain else 8))
    elif n_b == 3 and not rain: res.append(("Three Lights", 5))

    if {20, 24, 36} <= s: res.append(("Boar-Deer-Butterfly", 5))
    sake_pt = 1 if koi_cnt == 0 else 3
    if {8, 32} <= s: res.append(("Flower Viewing Sake", sake_pt))
    if {28, 32} <= s: res.append(("Moon Viewing Sake", sake_pt))
    if len(s_l) >= 5: res.append(("Tane", len(s_l) - 4))

    aka, ao = {1, 5, 9} <= s, {21, 33, 37} <= s
    if aka and ao: res.append(("Red & Blue Ribbons", 10))
    elif aka: res.append(("Red Ribbons", 5))
    elif ao: res.append(("Blue Ribbons", 5))
    if len(r_l) >= 5: res.append(("Tan", len(r_l) - 4))

    if len(d_l) >= 10: res.append(("Kasu", len(d_l) - 9))
    if koi_cnt > 0: res.append(("Koi-Koi", koi_cnt))
    return res


def init_gui() -> sg.Window:
    sg.theme("Material1")
    sc_board = [
        [sg.Text("Round", font=("Helvetica", 20))],
        [sg.Text("12 / 12", font=("Helvetica", 25), key="RoundCounter")],
        [sg.Text("            ", font=("Helvetica", 12), key="gameNum")],
        [sg.Text("Player2Name", font=("Helvetica", 20), key="opName")],
        [sg.Text("30 Points", font=("Helvetica", 18), key="opPoints")],
        [sg.Text("            ", font=("Helvetica", 12), key="opDealer")],
        [sg.Button(image_filename=f"{P_CARD}0-0.png", key="PileCard")],
        [sg.Text("Player1Name", font=("Helvetica", 20), key="myName")],
        [sg.Text("30 Points", font=("Helvetica", 18), key="myPoints")],
        [sg.Text("            ", font=("Helvetica", 12), key="myDealer")],
        *[ [sg.T("", size=(3, 1), key=f"PointsRound{i}") for i in r] for r in ((1,2,3), (4,5,6), (7,8,9), (10,11,12)) ],
        [sg.Button("Quit", size=(10, 1))],
    ]

    def mk_area(pre):
        return [[
            sg.Column([[sg.Image(f"{P_SMALL}null.png", key=f"{pre}Brights{i}") for i in range(1, 6)]]),
            sg.Column([
                [sg.Image(f"{P_SMALL}null.png", key=f"{pre}Seeds{i}") for i in range(1, 6)],
                [sg.Image(f"{P_SMALL}null.png", key=f"{pre}Seeds{i}") for i in range(6, 11)]
            ]),
            sg.Column([
                [sg.Image(f"{P_SMALL}null.png", key=f"{pre}Ribbons{i}") for i in range(1, 6)],
                [sg.Image(f"{P_SMALL}null.png", key=f"{pre}Ribbons{i}") for i in range(6, 11)]
            ]),
            sg.Column([
                [sg.Image(f"{P_SMALL}null.png", key=f"{pre}Dross{i}") for i in range(1, 14)],
                [sg.Image(f"{P_SMALL}null.png", key=f"{pre}Dross{i}") for i in range(14, 27)]
            ]),
        ]]

    center = [[
        sg.Column([
            [sg.Button(image_filename=f"{P_CARD}0-0.png", key=f"OpHand{i}") for i in range(1, 9)],
            [sg.Button(image_filename=f"{P_CARD}null.png", key=f"Board{i}") for i in (1,3,5,7,9,11,13,15)],
            [sg.Button(image_filename=f"{P_CARD}null.png", key=f"Board{i}") for i in (2,4,6,8,10,12,14,16)],
            [sg.Button(image_filename=f"{P_CARD}0-0.png", key=f"MyHand{i}") for i in range(1, 9)],
        ]),
        sg.Column([
            *[[sg.T("", size=(16, 1), key=f"OpYaku{i}"), sg.T("", size=(2, 1), key=f"OpYakuPt{i}")] for i in range(1, 11)],
            [sg.T("", size=(17, 1), key="Hint", text_color="blue")],
            *[[sg.T("", size=(16, 1), key=f"MyYaku{i}"), sg.T("", size=(2, 1), key=f"MyYakuPt{i}")] for i in range(1, 11)],
        ])
    ]]

    win = sg.Window("Koi-Koi", [[sg.Column(sc_board), sg.Column(mk_area("Op") + center + mk_area("My"))]], location=(50, 10), finalize=True)
    win.bind("<Button-1>", "Any Click")
    for i in range(1, 9):
        win[f"MyHand{i}"].bind("<Enter>", "-Enter")
        win[f"MyHand{i}"].bind("<Leave>", "-Leave")
    return win


def update_status(win: sg.Window, g) -> sg.Window:
    rs = g.cur
    win["RoundCounter"].update(value=f"{g.rnd} / {g.tot}")
    win["myName"].update(value=g.names[1]); win["opName"].update(value=g.names[2])
    win["myPoints"].update(value=f"{g.pts[1]} Points"); win["opPoints"].update(value=f"{g.pts[2]} Points")
    win["myDealer"].update(value="Dealer" if rs.dlr == 1 else "      ")
    win["opDealer"].update(value="Dealer" if rs.dlr == 2 else "      ")

    for i in range(1, g.rnd):
        win[f"PointsRound{i}"].update(g.log["rec"][f"rnd{i}"]["basic"]["p1_rnd_pt"])
    return win


def update_turn_p(win: sg.Window, g) -> sg.Window:
    p1 = (g.cur.tp == 1)
    win["myName"].update(value=g.names[1], text_color="blue" if p1 else "black")
    win["opName"].update(value=g.names[2], text_color="black" if p1 else "blue")
    return win


def clear_board(win: sg.Window) -> sg.Window:
    for p in ("My", "Op"):
        for i in range(1, 6): win[f"{p}Brights{i}"].update(filename=f"{P_SMALL}null.png")
        for i in range(1, 11): win[f"{p}Seeds{i}"].update(filename=f"{P_SMALL}null.png"); win[f"{p}Ribbons{i}"].update(filename=f"{P_SMALL}null.png")
        for i in range(1, 27): win[f"{p}Dross{i}"].update(filename=f"{P_SMALL}null.png")
        for i in range(1, 9): win[f"{p}Hand{i}"].update(image_filename=f"{P_CARD}null.png", visible=True)
        for i in range(1, 11): win[f"{p}Yaku{i}"].update(value=""); win[f"{p}YakuPt{i}"].update(value="")
    return win


def update_card_yaku(win: sg.Window, g) -> sg.Window:
    update_hand(win, g); update_board(win, g); update_collect(win, g); win["PileCard"].update(image_filename=f"{P_CARD}0-0.png"); update_yaku(win, g)
    return win


def update_collect(win: sg.Window, g) -> sg.Window:
    for pre, pile in (("My", g.cur.pile[1]), ("Op", g.cur.pile[2])):
        for name, lst in zip(("Brights", "Seeds", "Ribbons", "Dross"), classify(pile)):
            for i, c in enumerate(lst, 1): win[f"{pre}{name}{i}"].update(filename=P_SMALL + get_img(c))
    return win


def update_collect_hl(win: sg.Window, g, card: int) -> sg.Window:
    st = card // 4
    for pre, pile in (("My", g.cur.pile[1]), ("Op", g.cur.pile[2])):
        for name, lst in zip(("Brights", "Seeds", "Ribbons", "Dross"), classify(pile)):
            for i, c in enumerate(lst, 1):
                if c // 4 == st: win[f"{pre}{name}{i}"].update(filename=P_S_DARK + get_img(c))
    return win


def update_hand(win: sg.Window, g) -> sg.Window:
    rs = g.cur
    f_suits = {c // 4 for c in rs.f_slots if c != -1}
    for i, c in enumerate(rs.hand[1], 1):
        win[f"MyHand{i}"].update(image_filename=(P_CARD if (c // 4) in f_suits else P_DARK) + get_img(c), visible=True)
    for i in range(len(rs.hand[1]) + 1, 9): win[f"MyHand{i}"].update(image_filename=f"{P_CARD}null.png", visible=True)

    for i in range(1, len(rs.hand[2]) + 1): win[f"OpHand{i}"].update(image_filename=f"{P_CARD}0-0.png", visible=True)
    for i in range(len(rs.hand[2]) + 1, 9): win[f"OpHand{i}"].update(image_filename=f"{P_CARD}null.png", visible=True)
    return win


def update_my_dis(win: sg.Window, g) -> sg.Window:
    shw = g.cur.shw[0]
    for i, c in enumerate(g.cur.hand[1], 1): win[f"MyHand{i}"].update(image_filename=(P_CARD if c == shw else P_DARK) + get_img(c))
    return win


def update_op_dis(win: sg.Window, g) -> sg.Window:
    win["OpHand1"].update(image_filename=P_CARD + get_img(g.cur.shw[0]))
    return win


def update_board(win: sg.Window, g) -> sg.Window:
    for i, c in enumerate(g.cur.f_slots[:16], 1): win[f"Board{i}"].update(image_filename=P_CARD + get_img(c))
    return win


def update_board_hl(win: sg.Window, g, card: int) -> sg.Window:
    st = card // 4
    for i, c in enumerate(g.cur.f_slots[:16], 1):
        if c != -1: win[f"Board{i}"].update(image_filename=(P_CARD if c // 4 == st else P_DARK) + get_img(c))
    return win


def update_yaku(win: sg.Window, g) -> sg.Window:
    rs = g.cur
    for pre, p in (("My", 1), ("Op", 2)):
        yakus, pts = get_yakus(rs.pile[p], sum(rs.koi[p])), rs.yaku_pt(p)
        if len(yakus) >= 10:
            win[f"{pre}Yaku1"].update(value="Too Many Yakus"); win[f"{pre}Yaku2"].update(value="TOTAL"); win[f"{pre}YakuPt2"].update(value=str(pts))
            continue
        for i, (nm, pt) in enumerate(yakus, 1): win[f"{pre}Yaku{i}"].update(value=nm); win[f"{pre}YakuPt{i}"].update(value=str(pt))
        if yakus: win[f"{pre}Yaku{len(yakus)+1}"].update(value="TOTAL"); win[f"{pre}YakuPt{len(yakus)+1}"].update(value=str(pts))
    return win


def show_pile(win: sg.Window, g) -> sg.Window:
    win["PileCard"].update(image_filename=P_CARD + get_img(g.cur.shw[0]))
    return win


def wait_dis(win: sg.Window, g) -> Tuple[sg.Window, int]:
    h = g.cur.hand[1]
    win["Hint"].update(value="-> Select a Card")
    while True:
        ev, _ = win.read() or (None, None)
        if ev in {f"MyHand{i}-Enter" for i in range(1, len(h) + 1)}:
            ix = int(ev[6]) - 1
            update_board_hl(win, g, h[ix]); update_collect_hl(win, g, h[ix])
        elif ev in {f"MyHand{i}-Leave" for i in range(1, len(h) + 1)}:
            update_board(win, g); update_collect(win, g)
        elif ev in {f"MyHand{i}" for i in range(1, len(h) + 1)}:
            choice = h[int(ev[6]) - 1]; update_collect(win, g); break
        elif ev in ("Quit", None): sys.exit(0)
    return win, choice


def wait_pic(win: sg.Window, g) -> Tuple[sg.Window, int]:
    f = g.cur.f_slots
    v_ix = {i + 1 for i in range(16) if f[i] in g.cur.pairs}
    win["Hint"].update(value="-> Select Field Card"); update_board_hl(win, g, g.cur.shw[0])
    while True:
        ev, _ = win.read() or (None, None)
        if ev in {f"Board{i}" for i in v_ix}: choice = f[int(ev[5:]) - 1]; break
        elif ev in ("Quit", None): sys.exit(0)
    return win, choice


def wait_click(win: sg.Window) -> sg.Window:
    win["Hint"].update(value="-> Click to Continue")
    while True:
        ev, _ = win.read() or (None, None)
        if ev in ("Quit", None): sys.exit(0)
        elif ev == "Any Click": break
    return win


# =============================================================================
# ★最重要改修：Optional[bool] を全廃した文字列リテラル3値インターフェース
# =============================================================================
def wait_koi(win: sg.Window) -> Tuple[sg.Window, str]:
    win["Hint"].update(value="-> こいこいしますか？")
    res = sg.popup_yes_no("こいこいしますか？\n(Yes: 続行 / No: 上がり)", title="こいこい判断")
    if res is None: sys.exit(0)
    return win, ("koi" if res == "Yes" else "stop")


def show_op_koi(win: sg.Window, g, choice: str) -> sg.Window:
    if choice in ("koi", "stop"):
        sg.popup(f"{g.names[g.cur.tp]}: {'Koi-Koi' if choice == 'koi' else 'Stop'}", title="Koi-Koi")
    return win


def show_rnd_over(win: sg.Window, g) -> sg.Window:
    win["Hint"].update(value="-> Round Over")
    sg.popup(f"{g.names[1]}: {g.cur.pts[1]}     {g.names[2]}: {g.cur.pts[2]}", title="Round Over")
    return win


def show_game_over(win: sg.Window, g) -> sg.Window:
    win["Hint"].update(value="-> Game Over")
    sg.popup(f"{g.names[1]}: {g.pts[1]}     {g.names[2]}: {g.pts[2]}", title="Game Over")
    return win


# 後方互換性エイリアス
CardClassify = classify; InitGUI = init_gui; UpdateGameStatusGUI = update_status; UpdateTurnPlayer = update_turn_p
ClearBoardGUI = clear_board; UpdateCardAndYaku = update_card_yaku; UpdateCollectCardsGUI = update_collect
UpdateCollectCardsHighlightGUI = update_collect_hl; UpdateHandCardsGUI = update_hand; UpdateMyDiscardCardGUI = update_my_dis
UpdateOpDiscardCardGUI = update_op_dis; UpdateBoardCardsGUI = update_board; UpdateBoardCardsHighlightGUI = update_board_hl
UpdateYakuGUI = update_yaku; UpdatePileCardGUI = lambda w, g: w; ShowPileCardGUI = show_pile; WaitDiscardGUI = wait_dis
WaitPickGUI = wait_pic; WaitAnyClick = wait_click; WaitKoiKoi = wait_koi; ShowOpKoiKoi = show_op_koi
ShowRoundOverGUI = show_rnd_over; ShowGameOverGUI = show_game_over; Close = lambda w: w.Close()