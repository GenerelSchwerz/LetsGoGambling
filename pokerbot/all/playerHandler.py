from typing import Union
from pokerbot.abstract.pokerEventHandler import PokerStages
from pokerbot.all.utils import calc_af
from ..abstract.pokerDetection import Player


# Note: this does not currently handle blinds. Small bug.
class BetHandler:
    def __init__(self, sb: int, bb: int) -> None:
        self.reset(sb, bb)
        self.sb = sb
        self.bb = bb

    def reset(self, sb: Union[int, None] = None, bb: Union[int, None] = None) -> None:
        self._preflop_bets: list[tuple[str, float]] = []
        self._flop_bets: list[tuple[str, float]] = []
        self._turn_bets: list[tuple[str, float]] = []
        self._river_bets: list[tuple[str, float]] = []

        if sb is not None:
            self.sb = sb
        if bb is not None:
            self.bb = bb

    def add_bets(
        self,
        *bets: tuple[
            str, float
        ],  # assume sorted into betting order (first is first to act)
        round: int,
    ) -> None:
        assert (
            PokerStages.PREFLOP <= round <= PokerStages.RIVER
        ), f"Invalid round {round}"

        if round == PokerStages.PREFLOP:
            self._preflop_bets.extend(bets)
        elif round == PokerStages.FLOP:
            self._flop_bets.extend(bets)
        elif round == PokerStages.TURN:
            self._turn_bets.extend(bets)
        elif round == PokerStages.RIVER:
            self._river_bets.extend(bets)

    def get_bets(self, round: int) -> list[tuple[str, float]]:
        assert (
            PokerStages.PREFLOP <= round <= PokerStages.RIVER
        ), f"Invalid round {round}"

        if round == PokerStages.PREFLOP:
            return self._preflop_bets
        elif round == PokerStages.FLOP:
            return self._flop_bets
        elif round == PokerStages.TURN:
            return self._turn_bets
        elif round == PokerStages.RIVER:
            return self._river_bets

    def get_all_bets(
        self,
    ) -> tuple[
        list[tuple[str, float]],
        list[tuple[str, float]],
        list[tuple[str, float]],
        list[tuple[str, float]],
    ]:
        return self._preflop_bets, self._flop_bets, self._turn_bets, self._river_bets

    def get_bets_for(self, player: str) -> list[list[float]]:
        # separate bets for each round

        return [
            [bet for name, bet in self._preflop_bets if name == player],
            [bet for name, bet in self._flop_bets if name == player],
            [bet for name, bet in self._turn_bets if name == player],
            [bet for name, bet in self._river_bets if name == player],
        ]

    def calculate_aggression_factor(self, player: str) -> tuple[int, int, float]:

        # again, assume bets are in order of occurring
        def af_info_for_street(
            wanted_name: str, bets: list[tuple[str, float]]
        ) -> tuple[int, int]:
            calls = 0
            bets_or_raises = 0
            cur_call_amt = 0

            for p_name, bet in bets:
                if p_name == wanted_name:
                    if bet == cur_call_amt:
                        calls += 1
                    else:
                        bets_or_raises += 1

                cur_call_amt = bet

            return calls, bets_or_raises

        tot_c = 0
        tot_br = 0
        for stage_bet in [
            # self._preflop_bets, # not needed for AF
            self._flop_bets,
            self._turn_bets,
            self._river_bets,
        ]:

            # check if target bet on this stage
            if player in map(lambda x: x[0], stage_bet):

                res = af_info_for_street(player, stage_bet)
                tot_c += res[0]
                tot_br += res[1]

        return tot_c, tot_br, calc_af(tot_c, tot_br)

    def vpip_this_hand(self, player: str) -> bool:
        cur_call_amt = self.bb

        for idx, (player_name, bet) in enumerate(self._preflop_bets):
            if player_name == player:

                was_sb = idx == 0
                was_bb = idx == 1

                if (
                    bet > cur_call_amt
                    or (was_sb and bet == cur_call_amt)
                    or (not was_bb and bet == cur_call_amt)
                ):
                    return True

                cur_call_amt = bet

        return False

    def pfr_this_hand(self, player: str) -> bool:

        cur_call_amt = self.bb
        for player_name, bet in self._preflop_bets:
            if player_name == player:
                if bet > cur_call_amt:
                    return True

                cur_call_amt = bet

        return False


class PlayerHandler:
    def __init__(self, sb: int, bb: int, distance_thresh=30) -> None:
        self._internal_locs: dict[tuple[int, int, int, int], Player] = {}
        self.distance_thresh = distance_thresh
        self.bet_handler = BetHandler(sb, bb)

    def reset(self):
        self._internal_locs.clear()
        self.bet_handler.reset()

    def close_enough(
        self, src: tuple[int, int, int, int], dest: tuple[int, int, int, int]
    ) -> bool:
        return (
            sum((src[i] - dest[i]) ** 2 for i in range(4)) ** 0.5 < self.distance_thresh
        )

    def update_players(
        self, *player_info: tuple[Player, tuple[int, int, int, int]], update_names=False
    ):
        known_locs = self._internal_locs.keys()

        for player, ploc in player_info:
            for kloc in known_locs:
                if self.close_enough(kloc, ploc):
                    self._internal_locs[kloc].stack = player.stack
                    self._internal_locs[kloc].active = player.active

                    if update_names:
                        self._internal_locs[kloc].name = player.name

                    break  # inner loop

            else:
                assert isinstance(
                    ploc, tuple
                ), f"Expected tuple for ploc, got {type(ploc)}"

                # create new object to avoid reference issues
                self._internal_locs[ploc] = Player(
                    player.name, player.stack, player.active
                )

    def remove_player(self, player: str):
        for loc, p in self._internal_locs.items():
            if p.name == player:
                del self._internal_locs[loc]
                break

    def update_bets(
        self,
        *bets: tuple[tuple[Player, tuple[int, int, int, int]], float],
        round: int,
        # assume sorted into betting order (first is first to act)
    ):
        known_locs = self._internal_locs.keys()
        for (player, ploc), bet in bets:
            for kloc in known_locs:
                if self.close_enough(kloc, ploc):
                    # get old name
                    old_name = self._internal_locs[kloc].name

                    self.bet_handler.add_bets((old_name, bet), round=round)
                    break

            else:
                assert isinstance(
                    ploc, tuple
                ), f"Expected tuple for ploc, got {type(ploc)}"

                # create new object to avoid reference issues
                self._internal_locs[ploc] = Player(
                    player.name, player.stack, player.active
                )
                self.bet_handler.add_bets((player.name, bet), round=round)

    def get_bets_for(
        self, loc: tuple[int, int, int, int]
    ) -> Union[list[list[float]], None]:
        if (val := self._internal_locs.get(loc)) is not None:
            return self.bet_handler.get_bets_for(val.name)

    def get_bets_for_player(self, player: str) -> Union[list[list[float]], None]:
        for loc, p in self._internal_locs.items():
            if p.name == player:
                return self.bet_handler.get_bets_for(player)

    def get_af_for_name(self, player: str) -> Union[tuple[int, int, float], None]:
        for loc, p in self._internal_locs.items():
            if p.name == player:
                return self.bet_handler.calculate_aggression_factor(player)

    def get_af_for(
        self, loc: tuple[int, int, int, int]
    ) -> Union[tuple[int, int, float], None]:
        if (val := self._internal_locs.get(loc)) is not None:
            return self.bet_handler.calculate_aggression_factor(val.name)

    def get_vpip_for_name(self, player: str) -> Union[bool, None]:
        for loc, p in self._internal_locs.items():
            if p.name == player:
                return self.bet_handler.vpip_this_hand(player)

    def get_vpip_for(self, loc: tuple[int, int, int, int]) -> Union[bool, None]:
        if (val := self._internal_locs.get(loc)) is not None:
            return self.bet_handler.vpip_this_hand(val.name)

    def get_pfr_for_name(self, player: str) -> Union[bool, None]:
        for loc, p in self._internal_locs.items():
            if p.name == player:
                return self.bet_handler.pfr_this_hand(player)

    def get_pfr_for(self, loc: tuple[int, int, int, int]) -> Union[bool, None]:
        if (val := self._internal_locs.get(loc)) is not None:
            return self.bet_handler.pfr_this_hand(val.name)


def generate_player_bets(
    players: list[str], sb: int, bb: int, rand_shift=False
) -> list[tuple[str, float]]:
    # generate a valid bet spread for each player, given standard poker rules.
    # preflop to river.
    bets = [[], [], [], []]

    import random

    # shift players in circular order by random number
    # ex: shift by 2 | 0,1,2,3 -> 2,3,0,1

    if rand_shift:
        shift = random.randint(0, len(players) - 1)
        players = players[shift:] + players[:shift]

    bets[0].append((players[0], sb))
    bets[0].append((players[1], bb))

    folded_players = []

    total_pot = sb + bb

    player_len = len(players)

    for stage in range(4):  # preflop, flop, turn, river
        last_facing = sb if stage == 0 else 0
        cur_facing = bb if stage == 0 else 0

        p_played_this_bet = 0

        if stage == PokerStages.PREFLOP:
            p = 2
            bet_dict = {players[0]: sb, players[1]: bb}
        else:
            p = 0
            bet_dict = {}

        while p_played_this_bet < player_len:
            idx = p % player_len
            player = players[idx]
            p += 1

            # print(p_played_this_bet, PokerStages.to_str(stage), player, last_facing, cur_facing, total_pot, folded_players)

            if player in folded_players:
                p_played_this_bet += 1
                continue

            if len(folded_players) == player_len - 1:
                print(f"player {player} wins! Everyone else folded.")
                return bets

            choice = random.random()  # 0 to 1

            # raise decision
            if choice > 0.75:
                # min raise

                p_played_this_bet = 0
                if cur_facing > 0:
                    print(f"raising {player} at stage", PokerStages.to_str(stage))
                    tmp = last_facing
                    bets[stage].append((player, cur_facing + tmp))
                    bet_dict[player] = cur_facing + tmp
                    last_facing = cur_facing
                    cur_facing = cur_facing + tmp
                    total_pot += tmp
                else:
                    print(f"betting {player} at stage", PokerStages.to_str(stage))
                    tmp = total_pot // 2
                    bets[stage].append((player, cur_facing + tmp))
                    bet_dict[player] = cur_facing + tmp
                    last_facing = cur_facing
                    cur_facing = cur_facing + tmp
                    total_pot += tmp

            # call decision
            elif 0.25 < choice < 0.75:
                last_bet = bet_dict.get(player, 0)
                if last_bet < cur_facing and cur_facing > 0:
                    bets[stage].append((player, cur_facing))
                    print(f"calling {player} at stage", PokerStages.to_str(stage))
                else:
                    print(f"checking {player} at stage", PokerStages.to_str(stage))

            elif cur_facing == 0:
                print(f"checking {player} at stage", PokerStages.to_str(stage))
                pass

            # fold
            else:
                print(f"folding {player} at stage", PokerStages.to_str(stage))
                folded_players.append(player)

            p_played_this_bet += 1

        print()

    return bets


def bet_handler_test():

    players = ["first", "second", "third", "fourth"]

    sb = 50
    bb = 100
    target = players[1]

    bh = BetHandler(sb, bb)

    preflop, flop, turn, river = generate_player_bets(
        players, 50, 100, rand_shift=False
    )
    bh.add_bets(*preflop, round=PokerStages.PREFLOP)
    bh.add_bets(*flop, round=PokerStages.FLOP)
    bh.add_bets(*turn, round=PokerStages.TURN)
    bh.add_bets(*river, round=PokerStages.RIVER)

    print(bh.get_all_bets())
    print()
    print(bh.get_bets_for(target))
    print(bh.calculate_aggression_factor(target))
    print(bh.vpip_this_hand(target))
    print(bh.pfr_this_hand(target))

    # bh.add_bets(
    #     *[
    #         (players[0], 50),  # sb (post)
    #         (players[1], 100),  # bb (post)
    #         (players[2], 100),  # call
    #         (players[3], 300),  # raise
    #         (players[0], 600),  # raise
    #         (players[1], 600),  # call
    #         # player[2] folds
    #         # player[3] folds
    #     ],
    #     round=PokerStages.PREFLOP,
    # )

    # # just player[0] and player[1] left
    # bh.add_bets(
    #     *[
    #         (players[0], 900),  # bet
    #         (players[1], 900),  # call
    #     ],
    #     round=PokerStages.FLOP,
    # )

    # bh.add_bets(
    #     *[
    #         # player[0] checks, so no action here
    #         (players[1], 2700),  # bet
    #         (players[0], 2700),  # call
    #     ],
    #     round=PokerStages.TURN,
    # )

    # bh.add_bets(
    #     *[
    #         # player[0] checks, so no action here
    #         (players[1], 5400),  # bet
    #         (players[0], 10800),  # raise
    #         # player[1] folds
    #     ],
    #     round=PokerStages.RIVER,
    # )


def player_handler_test():

    ph = PlayerHandler()

    players = [
        # x, x1, y, y1
        (
            Player("first", 100, True),
            (200, 400, 200, 400),
        ),  # 200x200 box, bottom left is 200, 200
        (
            Player("second", 100, True),
            (200, 400, 600, 800),
        ),  # 200x200 box, bottom left is 200, 600
        (
            Player("third", 100, True),
            (800, 1000, 200, 400),
        ),  # 200x200 box, bottom left is 800, 200
        (
            Player("fourth", 100, True),
            (800, 1000, 600, 800),
        ),  # 200x200 box, bottom left is 800, 600
    ]

    # same locations as players, but misdetected names
    scuffed_players = [
        (Player("D --", 100, True), (200, 400, 200, 400)),
        (Player("", 100, True), (200, 400, 600, 800)),
        (Player("thi3d", 100, True), (800, 1000, 200, 400)),
        (Player("rewr", 100, True), (800, 1000, 600, 800)),
        # (Player("fifth", 100, True), (1000, 1200, 1000, 1200)),
    ]

    bets = [
        (scuffed_players[0], 10),
        (scuffed_players[1], 20),
        (scuffed_players[2], 30),
        (scuffed_players[3], 40),
    ]

    target = players[0]
    target_name = target[0].name
    target_pos = target[1]

    ph.update_players(*players)
    ph.update_players(*scuffed_players)

    ph.update_bets(*bets, round=PokerStages.PREFLOP)

    print(ph.get_bets_for_player(target_name))
    print(ph.get_bets_for(target_pos))

    print(ph.get_af_for_name(target_name))
    print(ph.get_af_for(target_pos))

    new_stack = 200
    target[0].stack = new_stack
    ph.update_players(target)

    print(ph._internal_locs)


def main():

    bet_handler_test()
    # player_handler_test()
    pass


if __name__ == "__main__":
    main()
