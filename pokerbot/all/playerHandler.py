from typing import Union
from pokerbot.abstract.pokerEventHandler import PokerStages
from pokerbot.all.utils import calc_af
from ..abstract.pokerDetection import Player


# Note: this does not currently handle blinds. Small bug.
class BetHandler:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._preflop_bets: list[tuple[str, float]] = []
        self._flop_bets: list[tuple[str, float]] = []
        self._turn_bets: list[tuple[str, float]] = []
        self._river_bets: list[tuple[str, float]] = []

    def add_bets(
        self,
        *bets: 
            tuple[str, float]
        ,  # assume sorted into betting order (first is first to act)
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
            self._preflop_bets,
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


class PlayerHandler:
    def __init__(self, distance_thresh=30) -> None:
        self._internal_locs: dict[tuple[int, int, int, int], Player] = {}
        self.distance_thresh = distance_thresh

        self.bet_handler = BetHandler()

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
                self._internal_locs[ploc] = Player(player.name, player.stack, player.active)

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
                self._internal_locs[ploc] = Player(player.name, player.stack, player.active)
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


def bet_handler_test():

    players = ["first", "second", "third", "fourth"]

    target = players[1]

    bh = BetHandler()

    bh.add_bets(
        [
            (players[0], 10),
            (players[1], 10),
            (players[2], 30),
            (players[3], 30),
            (players[0], 60),
            (players[1], 60),
        ],
        PokerStages.PREFLOP,
    )

    bh.add_bets(
        [
            (players[0], 10),
            (players[1], 10),
            (players[2], 30),
            (players[3], 60),
        ],
        PokerStages.FLOP,
    )

    bh.add_bets(
        [
            (players[0], 10),
            # (players[1], 10),
            (players[2], 30),
            (players[3], 30),
        ],
        PokerStages.TURN,
    )

    bh.add_bets(
        [
            (players[0], 10),
            (players[1], 30),
            (players[2], 30),
            (players[3], 60),
        ],
        PokerStages.RIVER,
    )

    print(bh.get_bets(PokerStages.RIVER))

    print(bh.get_bets_for(target))

    print(bh.calculate_aggression_factor(target))


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
    player_handler_test()
    pass


if __name__ == "__main__":
    main()
