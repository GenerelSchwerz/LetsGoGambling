from typing import Union
from treys import Card

from ..abstract.pokerEventHandler import PokerStages
from ..abstract.pokerDecisions import PokerDecisionMaking, PokerDecisionChoice
from ..abstract.pokerDetection import Player

from ..all.utils import PokerHands

import random

from treys import Deck, Evaluator

import numba


import time
# hello


def has_flush_draw(community_cards: list[Card], hole_cards: list[Card] = []) -> bool:
    suits = [Card.get_suit_int(card) for card in hole_cards + community_cards]
    for suit in suits:
        if suits.count(suit) >= 4:
            return True
    return False


def has_straight_draw(community_cards: list[Card], hole_cards: list[Card] = []) -> bool:
    # straight draw is defined as having 4 cards in a row
    ranks = [Card.get_rank_int(card) for card in hole_cards + community_cards]
    ranks.sort()
    count = 0
    for i in range(1, len(ranks)):
        if ranks[i] - ranks[i - 1] == 1:
            count += 1
            if count == 3:
                return True
        else:
            return False
    return False


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
    threshold_classes: Union[int, tuple[int, int]] = PokerHands.HIGH_CARD,
    threshold_players=1,
    opponents=1,
    opps_satisfy_thresh_now=False,
) -> float:
    wins = 0
    known_cards = hole_cards + community_cards

    # set up full deck beforehand to save cycles.
    SEMI_FULL_DECK = Deck.GetFullDeck().copy()

    for card in known_cards:
        SEMI_FULL_DECK.remove(card)

    evaluator = Evaluator()
    board_len = len(community_cards)

    # introducing stochastic variance here
    if is_lower := (is_middle := isinstance(threshold_classes, tuple)):
        threshold_class = threshold_classes[0]
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

        opponents_cards = [deck.draw(2) for _ in range(opponents)]
        full_board = community_cards + deck.draw(5 - board_len)

        board_rank = evaluator.evaluate([], full_board)
        board_class = evaluator.get_rank_class(board_rank)

        opps_satisfy_thresh_now = opps_satisfy_thresh_now and board_len != 5
        opp_board = community_cards if opps_satisfy_thresh_now else full_board

        thresh_sat = 0
        opp_count = 0

        for idx, opponent in enumerate(opponents_cards):

            opponent_rank = evaluator.evaluate(opponent, opp_board)

            # if opponent has a worse hand than the board AND the board is worse than the threshold
            # if opponent_rank >= board_rank and board_class >= threshold_class:
            #     evals[idx] = 100_000_000
            #     continue

            if board_class <= threshold_class:

                # if opponent rank is CURRENTLY stronger than RAN OUT board rank, then threshold is satisfied
                if opponent_rank <= board_rank:  # include chops, if not then only do <
                    thresh_sat += 1
            else:
                opponent_class = evaluator.get_rank_class(opponent_rank)
                if opponent_class <= threshold_class:
                    thresh_sat += 1

            # evaluate entire board out since threshold met.
            evals[idx] = (
                evaluator.evaluate(opponent, full_board)
                if opps_satisfy_thresh_now
                else opponent_rank
            )
            opp_count += 1

            if thresh_sat >= threshold_players:
                break

        if thresh_sat >= threshold_players:
            run_c += 1

            our_rank = evaluator.evaluate(hole_cards, full_board)

            if is_middle:
                threshold_class = threshold_classes[int(is_lower)]
                is_lower = not is_lower

            for idx in range(opp_count, opponents):
                evals[idx] = evaluator.evaluate(opponents_cards[idx], full_board)

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


class AlgoDecisions2(PokerDecisionMaking):

    def __init__(self) -> None:
        super().__init__()

        self.is_bluffing = False

        self.current_stage = PokerStages.PREFLOP
        self.was_reraised = False
        self.current_stage_bet = 0



    def wanted_on_turn(
            self, 
            hole_cards: list[Card],
            community_cards: list[Card],
            stage: int,
            facing_bets: list[tuple[Player, int]],
            facing_players: list[Player], # players may not have bet yet.
            mid_pot: int,
            min_bet: int,
            big_blind: int,
            stack_size: int
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

        if stage != self.current_stage:
            self.current_stage = stage
            self.current_stage_bet = 0
            self.was_reraised = False

        # first, refine the data to match accordingly.
        actual_facing_bet = facing_bet - self.current_stage_bet

        if actual_facing_bet < facing_bet:
            self.was_reraised = True

        self.current_stage_bet = actual_facing_bet
