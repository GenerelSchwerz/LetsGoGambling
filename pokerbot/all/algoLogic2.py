from typing import Union
from treys import Card
import numpy as np

from pokerbot.all.playerHandler import PlayerHandler


from ..abstract.pokerEventHandler import PokerStages
from ..abstract.pokerDecisions import PokerDecisionMaking, PokerDecisionChoice
from ..abstract.pokerDetection import Player

from treys import Card, Deck, Evaluator, PLOEvaluator
from ..all.utils import PokerHands, is_hand_possible, is_in_percentile


import time

# hello


def has_flush_draw(community_cards: list[Card], hole_cards: list[Card] = []) -> bool:
    return is_hand_possible(community_cards + hole_cards, PokerHands.FLUSH)


def has_straight_draw(community_cards: list[Card], hole_cards: list[Card] = []) -> bool:
    return is_hand_possible(community_cards + hole_cards, PokerHands.STRAIGHT)


# checks if list of cards with length 3 or 4 has three of a kind
def has_three_of_a_kind(community_cards: list[Card], hole_cards: list[Card] = []):
    ranks = [Card.get_rank_int(card) for card in community_cards + hole_cards]
    for rank in set(ranks):
        if ranks.count(rank) == 3:
            return True
    return False


# checks if list of cards with length 4 has two pair
def has_two_pair(community_cards: list[Card], hole_cards: list[Card] = []):
    ranks = [Card.get_rank_int(card) for card in community_cards + hole_cards]
    pairs = 0
    for rank in set(ranks):
        if ranks.count(rank) >= 2:
            pairs += 1
    return pairs >= 2


def determine_current_threshold(
    community_cards: list[Card], hole_cards: list[Card] = []
) -> int:
    if has_flush_draw(community_cards, hole_cards):
        return PokerHands.FLUSH_DRAW
    elif has_straight_draw(community_cards, hole_cards):
        return 2
    elif has_three_of_a_kind(community_cards, hole_cards):
        return 1
    elif has_two_pair(community_cards, hole_cards):
        return 0
    else:
        return -1


# TODO make everything numpy? Could probably see more performance that way.
# @numba.jit()
def fast_calculate_equity(
    hole_cards: list[Card],
    community_cards: list[Card],
    sim_time=4000,
    runs=1500,
    threshold_classes: Union[int, list[int]] = PokerHands.HIGH_CARD,
    threshold_players=1,
    opponents=1,
    opps_satisfy_thresh_now=False,
) -> float:
    wins = 0
    known_cards = hole_cards + community_cards

    if opps_satisfy_thresh_now:
        if isinstance(threshold_classes, tuple) or isinstance(threshold_classes, list):
            if any(
                not is_hand_possible(community_cards, threshold)
                for threshold in threshold_classes
            ):
                str_it = ", ".join(
                    PokerHands.to_str(threshold) for threshold in threshold_classes
                )

                raise ValueError(
                    f"opps_satisfy_thresh_now is True but no opponents can satisfy threshold ({str_it}) given the board{Card.ints_to_pretty_str(community_cards)}"
                )
        elif isinstance(threshold_classes, int):
            if not is_hand_possible(community_cards, threshold_classes):

                raise ValueError(
                    f"opps_satisfy_thresh_now is True but no opponents can satisfy threshold {PokerHands.to_str(threshold_classes)} given the board{Card.ints_to_pretty_str(community_cards)}"
                )
        else:
            raise ValueError("threshold_classes must be an int or a tuple of two ints")

    # set up full deck beforehand to save cycles.
    SEMI_FULL_DECK = Deck.GetFullDeck().copy()

    for card in known_cards:
        SEMI_FULL_DECK.remove(card)

    hand_len = len(hole_cards)

    hand_evaluator = PLOEvaluator() if hand_len == 4 else Evaluator()
    board_evaluator = Evaluator()
    board_len = len(community_cards)

    # introducing stochastic variance here

    if is_range := (
        isinstance(threshold_classes, list) or isinstance(threshold_classes, tuple)
    ):
        idx1 = np.random.randint(0, len(threshold_classes))
        threshold_class = threshold_classes[idx1]
    elif isinstance(threshold_classes, int):
        threshold_class = threshold_classes
    else:
        raise ValueError("threshold_classes must be an int or a tuple of two ints")

    start = time.time()
    run_c = 0
    run_it = 0

    deck = Deck()

    # evals_list = list()
    evals = [100_000_000 for _ in range(opponents)]

    while run_c < runs:

        if run_it % 100 == 0:
            if (time.time() - start) * 1000 >= sim_time:
                break

        run_it += 1
        # deck needs to be modified
        # deck = Deck(shuffle=False)
        deck.cards = SEMI_FULL_DECK.copy()
        deck._random.shuffle(deck.cards)

        opponents_cards = [deck.draw(hand_len) for _ in range(opponents)]
        full_board = community_cards + deck.draw(5 - board_len)

        board_rank = board_evaluator.evaluate([], full_board)
        board_class = board_evaluator.get_rank_class(board_rank)

        opps_satisfy_thresh_now = opps_satisfy_thresh_now and board_len != 5
        opp_board = community_cards if opps_satisfy_thresh_now else full_board

        thresh_sat = 0
        opp_count = 0

        for idx, opponent in enumerate(opponents_cards):

            opponent_rank = hand_evaluator.evaluate(opponent, opp_board)

            # if opponent has a worse hand than the board AND the board is worse than the threshold
            # if opponent_rank >= board_rank and board_class >= threshold_class:
            #     evals[idx] = 100_000_000
            #     continue

            if board_class <= threshold_class:

                # if threshhold is high card, just evaluate every hand.
                if threshold_class == PokerHands.HIGH_CARD:
                    opponent_rank = hand_evaluator.evaluate(opponent, full_board)

                # # if opponent rank is CURRENTLY stronger than RAN OUT board rank, then threshold is satisfied
                if opponent_rank <= board_rank:  # include chops, if not then only do <

                    thresh_sat += 1
            else:
                opponent_class = hand_evaluator.get_rank_class(opponent_rank)
                if opponent_class <= threshold_class:
                    thresh_sat += 1

            # evaluate entire board out since threshold met.
            evals[idx] = (
                hand_evaluator.evaluate(opponent, full_board)
                if opps_satisfy_thresh_now  # and board_class > threshold_class
                else opponent_rank
            )
            opp_count += 1

            if thresh_sat >= threshold_players:
                break

        if thresh_sat >= threshold_players:
            run_c += 1

            our_rank = hand_evaluator.evaluate(hole_cards, full_board)

            if is_range:
                threshold_class = threshold_classes[idx1]
                idx1 = (idx1 + 1) % len(threshold_classes)

            for idx in range(opp_count, opponents):
                evals[idx] = hand_evaluator.evaluate(opponents_cards[idx], full_board)

            for opp_rank in evals:
                if our_rank > opp_rank:
                    # for idx, opp_rank in enumerate(evals):
                    #     if our_rank > opp_rank:
                    #         print(f"For value: {threshold_class}, We lost to hand {idx + 1}: {Card.ints_to_pretty_str(opponents_cards[idx])} | {Card.ints_to_pretty_str(full_board)}, {evaluator.get_rank_class(opp_rank)}, {evaluator.get_rank_class(our_rank)}")
                    break  # mid level loop

            else:
                # print(f"For value: {threshold_class}, We won with: {Card.ints_to_pretty_str(hole_cards)} | {Card.ints_to_pretty_str(full_board)}, {evaluator.get_rank_class(our_rank)}")
                # for opp in range(opp_count):
                #     print(f"Beat hand {opp + 1}: {Card.ints_to_pretty_str(opponents_cards[opp])} | {Card.ints_to_pretty_str(full_board)}, {evaluator.get_rank_class(evals[opp])}, {evaluator.get_rank_class(our_rank)}")
                wins += 1

    print(f"Successful simulations: {run_c}, wins: {wins}, runs: {run_it}")
    return wins / run_c if run_c > 0 else 0


def calculate_equity_preflop(
    hole_cards: list[Card],
    opponents: int,
    community_cards: list[Card] = [],
    sim_time=4000,  # how much time in ms spent on calculations MAX
    runs=1500,
    # num successful simulations to be satisfied & return before simulation_time is up
    threshold_percentile=60,  # percentile of hands opponents are playing
    threshold_players=1,  # min players we assume satisfy threshold_percentile
    # for example, with default arguments, we calculate our equity
    # (the chance that our hand beats or chops with everyone's hands after a random run-out)
    # assuming at least ONE player has hole cards in the top 60th percentile
    # if threshold_hand_strength were 20 and threshold_players was 2,
    # then we calculate the chance that we beat/chop with everyone after a random run-out,
    # assuming AT LEAST two players have hands in the top 20th percentile
) -> float:
    start_time = time.time()

    com_card_len = len(community_cards)
    hole_card_len = len(hole_cards)
    deck = Deck()
    evaluator = PLOEvaluator() if hole_card_len == 4 else Evaluator()

    SEMI_FULL_DECK = Deck.GetFullDeck().copy()
    for card in hole_cards + community_cards:
        SEMI_FULL_DECK.remove(card)

    wins = 0
    run_c = 0
    run_it = 0
    while run_c < runs:

        if run_it % 100 == 0:
            if (time.time() - start_time) * 1000 >= sim_time:
                break

        run_it += 1

        deck.cards = SEMI_FULL_DECK.copy()
        deck._random.shuffle(deck.cards)

        opp_cards = [deck.draw(hole_card_len) for _ in range(opponents)]

        thresh_sat = 0

        for i in range(opponents):
            if is_in_percentile(threshold_percentile, opp_cards[i], opponents > 1):
                thresh_sat += 1

                # runs once.
                if thresh_sat >= threshold_players:
                    run_c += 1
                    board = community_cards + deck.draw(5 - com_card_len)
                    our_rank = evaluator.evaluate(hole_cards, board)

                    for i in range(opponents):
                        opp = evaluator.evaluate(opp_cards[i], board)
                        if opp < our_rank:
                            break
                    else:
                        wins += 1

                    break  # inner loop

    print(f"Successful simulations: {run_c} ({time.time() - start_time}s) {wins}")
    # watch for div by 0
    return wins / run_c if run_c != 0 else 0


class AlgoDecisions2(PokerDecisionMaking):

    def __init__(self) -> None:
        super().__init__()

        self.is_bluffing = False

        self.current_stage = PokerStages.PREFLOP
        self.was_reraised = False
        self.current_stage_bet = 0

        self.player_handler: PlayerHandler = None

    def wanted_on_turn(
        self,
        hole_cards: list[Card],
        community_cards: list[Card],
        stage: int,
        facing_bets: list[tuple[Player, int]],
        facing_players: list[Player],  # players may not have bet yet.
        mid_pot: int,
        min_bet: int,
        big_blind: int,
        stack_size: int,
    ) -> PokerDecisionChoice:
        if stage != self.current_stage:
            self.current_stage = stage
            self.current_stage_bet = 0
            self.was_reraised = False

        max_facing = max([bet for _, bet in facing_bets])

        # first, refine the data to match accordingly.
        actual_facing_bet = max_facing - self.current_stage_bet
        self.current_stage_bet = actual_facing_bet

        if actual_facing_bet < max_facing:
            self.was_reraised = True

        # let's find some basic info about this hand.

    def on_turn(
        self,
        hole_cards: list[Card],
        community_cards: list[Card],
        stage: int,
        facing_bet: int,
        min_bet: int,
        mid_pot: int,
        total_pot: int,
        big_blind: int,
        stack_size: int,
        active_opponents: int,
    ) -> PokerDecisionChoice:
        pass

        if stage != self.current_stage:
            self.current_stage = stage
            self.current_stage_bet = 0
            self.was_reraised = False

        # first, refine the data to match accordingly.
        actual_facing_bet = facing_bet - self.current_stage_bet

        if actual_facing_bet < facing_bet:
            self.was_reraised = True

        self.current_stage_bet = actual_facing_bet
