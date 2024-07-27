from treys import Card

from ..abstract.pokerEventHandler import PokerStages
from ..abstract.pokerDecisions import PokerDecisionMaking, PokerDecisionChoice
from ..abstract.pokerDetection import Player

from ..all.utils import PokerHands

import random

from treys import Deck, Evaluator

import numba

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
@numba.njit()
def fast_calculate_equity(
    hole_cards: list[Card],
    community_cards: list[Card],
    sim_time=4000,
    runs=1000,
    threshold_class=PokerHands.HIGH_CARD,
    threshold_players=1,
    opponents=1,
) -> float:
    wins = 0
    known_cards = hole_cards + community_cards

    # set up full deck beforehand to save cycles.
    SEMI_FULL_DECK = Deck.GetFullDeck().copy()
    for card in known_cards:
        SEMI_FULL_DECK.remove(card)

    evaluator = Evaluator()
    board_len = len(community_cards)


    for idx in range(runs):
        deck = Deck()
        deck.cards = SEMI_FULL_DECK.copy()
        deck._random.shuffle(deck.cards)

        opponents_cards = [deck.draw(2) for _ in range(opponents)]
        full_board = community_cards + deck.draw(5 - board_len)

        our_score = evaluator.evaluate(hole_cards, full_board)
        our_rank = evaluator.get_rank_class(our_score)  

        # if our own hand is worse than the threshold, we'll skip the comparison.
        if our_rank > threshold_class:
            continue # outer loop

        board_rank = evaluator.evaluate([], full_board)
        board_class = evaluator.get_rank_class(board_rank)

        thresh_count = 0
        opp_count = 0
        evals = []
        for opponent in opponents_cards:
            opponent_rank = evaluator.evaluate(opponent, full_board)
         
            # weird amelia logic inbound (idk what this is tbh)
            if board_class >= PokerHands.PAIR:
                opponent_class = evaluator.get_rank_class(opponent_rank)

                # this code is retarded. this is literally ONLY a pair.
                if PokerHands.THREE_OF_A_KIND < threshold_class < PokerHands.HIGH_CARD:
                    adj_thresh = threshold_class - (PokerHands.HIGH_CARD - board_class)
                else:
                    adj_thresh = threshold_class

                # opponent's category is stronger than the threshold
                if opponent_class < adj_thresh:
                    thresh_count += 1

            # opponent has literally anything on this board.
            elif opponent_rank < board_rank:
                thresh_count += 1

            evals.append(opponent_rank)
            opp_count += 1

            if thresh_count >= threshold_players:
                break # inner loop
        else:
            print("No one met the threshold")
            continue # outer loop

        # fill in all other opponents
        for idx in range(opp_count, opponents):
            evals.append(evaluator.evaluate(opponents_cards[idx], full_board))

        # if we are stronger than all opponents, we win.
        if all([our_score <= opp_score for opp_score in evals]):
            wins += 1

    return wins / runs if runs > 0 else 0
      




class AlgoDecisions2(PokerDecisionMaking):

    def __init__(self) -> None:
        super().__init__()

        self.is_bluffing = False

        self.current_stage = PokerStages.PREFLOP
        self.was_reraised = False
        self.current_stage_bet = 0

    def is_board_draw_heavy(self, community_cards: list[Card]) -> bool:
        pass


    def wanted_on_turn(
            self, 
            hole_cards: list[Card],
            community_cards: list[Card],
            stage: int,
            facing_bets: list[tuple[Player, int]],
            facing_players: list[Player],
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

        if actual_facing_bet < max_facing:
            self.was_reraised = True

        self.current_stage_bet = actual_facing_bet

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
