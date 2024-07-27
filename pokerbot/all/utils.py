from poker import Range
from treys import Card, Evaluator

from ..abstract.pokerDecisions import PokerDecisionChoice
from ..abstract.pokerEventHandler import PokerStages

poker_ranks = '23456789TJQKA'
poker_suits = 'shdc'

treys_lookup_table = {}

for r in poker_ranks:
    for s in poker_suits:
        card_str = r + s
        treys_card = Card.new(card_str)
        treys_lookup_table[treys_card] = card_str


class PokerHands:
    HIGH_CARD = 9
    PAIR = 8
    TWO_PAIR = 7
    THREE_OF_A_KIND = 6
    STRAIGHT = 5
    FLUSH = 4
    FULL_HOUSE = 3
    FOUR_OF_A_KIND = 2
    STRAIGHT_FLUSH = 1
    ROYAL_FLUSH = 0

    @staticmethod
    def merge(low: int, high: int) -> float:
        """
        Merge two hand strengths into one.
        """
        return (low + high) / 2
    

perc_ranges_multiple_ops = {
    60: Range("22+, A2s+, K2s+, Q2s+, J2s+, T2s+, 94s+, 84s+, 74s+, 64s+, 54s, A2o+, K3o+, Q5o+, J7o+, T7o+, 97o+"),
    40: Range("33+, A2s+, K2s+, Q4s+, J6s+, T7s+, 97s+, 87s, A3o+, K7o+, Q8o+, J9o+, T9o"),
    30: Range("44+, A2s+, K2s+, Q6s+, J7s+, T7s+, 98s, A7o+, K9o+, Q9o+, JTo"),
    20: Range("55+, A3s+, K7s+, Q8s+, J9s+, T9s, A9o+, KTo+, QJo"),
    10: Range("77+, A9s+, KTs+, QJs, AJo+, KQo"),
    5: Range("99+, AJs+, KQs, AKo")
}
perc_ranges_single_op = {
    60: Range("22+, A2s+, K2s+, Q2s+, J2s+, T4s+, 96s+, 87s, A2o+, K2o+, Q2o+, J6o+, T7o+, 98o"),
    40: Range("33+, A2s+, K2s+, Q5s+, J7s+, T8s+, A2o+, K5o+, Q8o+, J9o+"),
    30: Range("44+, A2s+, K4s+, Q8s+, J9s+, A4o+, K8o+, Q9o+"),
    20: Range("55+, A4s+, K8s+, Q9s+, A7o+, KTo+, QJo"),
    10: Range("66+, A9s+, KTs+, AJo+"),
    5: Range("88+, AQs+, AKo")
}


def ceil_half(num):
    return -(num // -2)


def fold_or_check(current_bet) -> PokerDecisionChoice:
    if current_bet == 0:
        return PokerDecisionChoice.check()
    return PokerDecisionChoice.fold()


def bet(ideal_bet, stack_size):
    return PokerDecisionChoice.bet(min(int(ideal_bet), stack_size))


def get_stage(community_cards):
    if len(community_cards) == 0:
        current_stage = PokerStages.PREFLOP
    elif len(community_cards) == 3:
        current_stage = PokerStages.FLOP
    elif len(community_cards) == 4:
        current_stage = PokerStages.TURN
    elif len(community_cards) == 5:
        current_stage = PokerStages.RIVER
    else:
        raise ValueError("Malformed community cards")
    return current_stage


### CARD/BOARD EVALUATION ###

# returns hand rank from 1-7000ish and hand strength from 0-9
def evaluate_hand(hole_cards, community_cards):
    # check if community cards is empty
    if not community_cards:
        return 9999, PokerHands.HIGH_CARD
    evaluator = Evaluator()
    player_hand = evaluator.evaluate(hole_cards, community_cards)
    player_hand_strength = evaluator.get_rank_class(player_hand)
    return player_hand, player_hand_strength


# checks if board has 4 or more of one suit
def has_flush_draw(board_cards):
    flush_draw = False
    suits = [Card.get_suit_int(card) for card in board_cards]
    for suit in suits:
        if suits.count(suit) >= 4:
            flush_draw = True
    return flush_draw


# checks if list of cards with length 3 or 4 has three of a kind
def has_three_of_a_kind(cards):
    cards = [Card.get_rank_int(card) for card in cards]
    for card in cards:
        if cards.count(card) == 3:
            return True
    return False


# checks if list of cards with length 4 has two pair
def has_two_pair(cards):
    cards = [Card.get_rank_int(card) for card in cards]
    cards_set = set(cards)
    pairs = 0
    for card in cards_set:
        if cards.count(card) >= 2:
            pairs += 1
    return pairs >= 2


def is_in_percentile(percentile, hole_cards, multiple_opponents=True):
    return is_in_range(
        perc_ranges_multiple_ops[percentile] if multiple_opponents else perc_ranges_single_op[percentile],
        hole_cards)


def is_in_range(hand_range, hole_cards):
    # Convert treys Card objects to poker Card objects
    hole_cards_poker = treys_to_poker(hole_cards)
    hole_cards_poker = "".join(hole_cards_poker)
    value = hole_cards_poker in hand_range
    return value


def treys_to_poker(cards):
    # use lookup_table
    return [treys_lookup_table[card] for card in cards]
