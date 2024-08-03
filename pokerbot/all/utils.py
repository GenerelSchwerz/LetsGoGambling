import math
from poker import Range
from treys import Card, Evaluator

from pokerbot.abstract.pokerDetection import Player

from ..abstract.pokerDecisions import PokerDecisionChoice
from ..abstract.pokerEventHandler import PokerStages

poker_ranks = "23456789TJQKA"
poker_suits = "shdc"

treys_lookup_table: dict[str, str] = {}

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

    str_dict = {
        HIGH_CARD: "HIGH_CARD",
        PAIR: "PAIR",
        TWO_PAIR: "TWO_PAIR",
        THREE_OF_A_KIND: "THREE_OF_A_KIND",
        STRAIGHT: "STRAIGHT",
        FLUSH: "FLUSH",
        FULL_HOUSE: "FULL_HOUSE",
        FOUR_OF_A_KIND: "FOUR_OF_A_KIND",
        STRAIGHT_FLUSH: "STRAIGHT_FLUSH",
        ROYAL_FLUSH: "ROYAL_FLUSH",
    }

    @staticmethod
    def merge(low: int, high: int) -> float:
        """
        Merge two hand strengths into one.
        """
        return (low + high) / 2
    
    @staticmethod
    def to_str(hand: int | list[int]) -> str:
        if isinstance(hand, list) or isinstance(hand, tuple):
            return ", ".join([PokerHands.str_dict[h] for h in hand])
        
        return PokerHands.str_dict[hand]



perc_ranges_multiple_ops = {
    60: Range(
        "22+, A2s+, K2s+, Q2s+, J2s+, T2s+, 94s+, 84s+, 74s+, 64s+, 54s, A2o+, K3o+, Q5o+, J7o+, T7o+, 97o+"
    ),
    40: Range(
        "33+, A2s+, K2s+, Q4s+, J6s+, T7s+, 97s+, 87s, A3o+, K7o+, Q8o+, J9o+, T9o"
    ),
    30: Range("44+, A2s+, K2s+, Q6s+, J7s+, T7s+, 98s, A7o+, K9o+, Q9o+, JTo"),
    20: Range("55+, A3s+, K7s+, Q8s+, J9s+, T9s, A9o+, KTo+, QJo"),
    10: Range("77+, A9s+, KTs+, QJs, AJo+, KQo"),
    5: Range("99+, AJs+, KQs, AKo"),
}
perc_ranges_single_op = {
    60: Range(
        "22+, A2s+, K2s+, Q2s+, J2s+, T4s+, 96s+, 87s, A2o+, K2o+, Q2o+, J6o+, T7o+, 98o"
    ),
    40: Range("33+, A2s+, K2s+, Q5s+, J7s+, T8s+, A2o+, K5o+, Q8o+, J9o+"),
    30: Range("44+, A2s+, K4s+, Q8s+, J9s+, A4o+, K8o+, Q9o+"),
    20: Range("55+, A4s+, K8s+, Q9s+, A7o+, KTo+, QJo"),
    10: Range("66+, A9s+, KTs+, AJo+"),
    5: Range("88+, AQs+, AKo"),
}


def ceil_half(num):
    return -(num // -2)


import math

def calc_af(calls: int, bets_or_raises: int) -> float:
    return  bets_or_raises / calls if calls != 0 else math.inf if bets_or_raises != 0 else 0


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



def is_hand_possible(board: list[Card], expected: int) -> bool:

    if len(board) == 5:
        return True
    
    # # board is always at least a high card
    # if expected == PokerHands.HIGH_CARD:
    #     return True
    
    # # it is always possible for a hand to pair a board
    # if expected == PokerHands.PAIR:
    #     return True
    
    # # it is always possible for a hand to two pair a board
    # if expected == PokerHands.TWO_PAIR:
    #     return True
    
    # # it is always possible for a hand to three of a kind a board
    # if expected == PokerHands.THREE_OF_A_KIND:
    #     return True
    
    if expected == PokerHands.STRAIGHT:
        # straight is defined as having 5 cards in a row
        ranks = [Card.get_rank_int(card) for card in board]
   
        # Create a set of card values to remove duplicates
        card_set = set(ranks)
        
        # Add special case for Ace (12), which can be part of low (0-4) and high (8-12) straights
        if 12 in card_set:
            card_set.add(-1)  # -1 represents Ace as 1
        
        # Sort the card values
        sorted_cards = sorted(card_set)
        
        # Check for straight possibilities in the sorted card values
        for i in range(len(sorted_cards)):
            count = 1
            for j in range(i + 1, len(sorted_cards)):
                if sorted_cards[j] <= sorted_cards[i] + 4:
                    count += 1
                    if count >= 3:
                        return True  # Potential straight found
                else:
                    break  # No need to check further if gap is found
            
        return False
                
    if expected == PokerHands.FLUSH:
        suits = [Card.get_suit_int(card) for card in board]
        for suit in suits:
            # require the board to have at least 3 of the same suit (suited hand can complete it, otherwise flush needed.)
            if suits.count(suit) >= 3:
                return True
        return False
    
    if expected == PokerHands.FULL_HOUSE:
        ranks = [Card.get_rank_int(card) for card in board]
        for rank in set(ranks):
            # require the board to be paired.
            if ranks.count(rank) >= 2:
                return True
            
        return False
    
    if expected == PokerHands.FOUR_OF_A_KIND:
        ranks = [Card.get_rank_int(card) for card in board]
        for rank in set(ranks):
            # require the board to at least be paired (pocket pairs can complete it, otherwise set needed.)
            if ranks.count(rank) >= 2:
                return True
        return False
    
    if expected == PokerHands.STRAIGHT_FLUSH:
        suits = [Card.get_suit_int(card) for card in board]
        for card in board:
            suit = Card.get_suit_int(card)

            if suits.count(suit) >= 3:
                new_board = [card for card in board if Card.get_suit_int(card) == suit]
                ranks = [Card.get_rank_int(card) for card in new_board]
                card_set = set(ranks)
                if 12 in card_set:
                    card_set.add(-1)
                sorted_cards = sorted(card_set)
                for i in range(len(sorted_cards)):
                    count = 1
                    for j in range(i + 1, len(sorted_cards)):
                        if sorted_cards[j] <= sorted_cards[i] + 4:
                            count += 1
                            if count >= 3:
                                return True
                        else:
                            break
        return False
            
    if expected == PokerHands.ROYAL_FLUSH:
        suits = [Card.get_suit_int(card) for card in board]
        for card in board:
            suit = Card.get_suit_int(card)

            if suits.count(suit) >= 3:
                new_board = [card for card in board if Card.get_suit_int(card) == suit]
                ranks = [Card.get_rank_int(card) for card in new_board]
                if len(list(filter(lambda x: x >= 8, ranks))) < 3:
                    return False
                
                card_set = set(ranks)
                if 12 in card_set:
                    card_set.add(-1)
                sorted_cards = sorted(card_set)
                for i in range(len(sorted_cards)):
                    count = 1
                    for j in range(i + 1, len(sorted_cards)):
                        if sorted_cards[j] <= sorted_cards[i] + 4:
                            count += 1
                            if count >= 3:
                                return True
                        else:
                            break
        return False
    
    # it is otherwise possible to have the expected value on the board.
    if PokerHands.ROYAL_FLUSH <= expected <= PokerHands.HIGH_CARD:
        return True
    else:
        raise ValueError("Invalid expected hand value, needs to be between 0 and 9.")
    

def is_in_percentile(percentile, hole_cards, multiple_opponents=True):

    if (card_len := len(hole_cards)) == 2:

        return is_in_range(
            (
                perc_ranges_multiple_ops[percentile]
                if multiple_opponents
                else perc_ranges_single_op[percentile]
            ),
            hole_cards,
        )
    elif card_len == 4:
        # implement later.
        return True


def is_in_range(hand_range: int, hole_cards: list[Card]):
    # Convert treys Card objects to poker Card objects
    hole_cards_poker = treys_to_poker(hole_cards)
    hole_cards_poker = "".join(hole_cards_poker)
    value = hole_cards_poker in hand_range
    return value

def is_in_range_plo(hand_range: int, hole_cards: list[Card]):
    # Convert treys Card objects to poker Card objects
    hole_cards_poker = treys_to_poker(hole_cards)
    hole_cards_poker = "".join(hole_cards_poker)
    value = hole_cards_poker in hand_range
    return value

def treys_to_poker(cards):
    # use lookup_table
    return [treys_lookup_table[card] for card in cards]


from typing import Any, Union
from ..abstract.pokerEventHandler import PokerStages


def suit_full_name_to_abbrev(suit_full_name: str) -> str:
    if suit_full_name == "hearts":
        return "h"
    elif suit_full_name == "diamonds":
        return "d"
    elif suit_full_name == "clubs":
        return "c"
    elif suit_full_name == "spades":
        return "s"

    raise ValueError("Invalid suit name")


def card_to_abbrev(card: str) -> str:
    if card == "10":
        return "T"
    return card


def pretty_str_to_float(str: str) -> int:
    number = str.lower()
    # remove comma/period
    has_period = "." in number or "," in number
    new_number = number.replace(",", "")
    new_number = new_number.replace(".", "")
    new_number = new_number.replace("k", "00" if has_period else "000")
    new_number = new_number.replace("m", "00000" if has_period else "000000")
    try:
        return int(new_number)
    except ValueError:
        print(f"Could not convert number to int: {number} ({new_number})")
        return 0


def cards_to_stage(cards: list[Any]) -> PokerStages:
    if len(cards) == 0:
        return PokerStages.PREFLOP
    elif len(cards) <= 3:
        return PokerStages.FLOP
    elif len(cards) == 4:
        return PokerStages.TURN
    else:
        return PokerStages.RIVER


def associate_bet_locs(
    players: list[tuple[Player, tuple[int, int, int, int]]],
    bets: list[tuple[float, tuple[int, int, int, int]]],
) -> dict[tuple[int,int,int,int], tuple[float, tuple[int, int, int, int]]]:
    """
    Associate bets with players
    """

    # sort players by x position
    players.sort(key=lambda x: x[1][0])

    # sort bets by x position
    bets.sort(key=lambda x: x[1][0])

    ret = {}

    already_used_bets = set()

    # for each player, associate closest bet on both X and Y axis (bets can be on either side of player)
    # outer loop is bets
    # inner loop is players

    for bet_amt, bet_loc in bets:
        # find closest player to bet
        closest_player_loc = None
        closest_dist = math.inf

        b_mid_x = (bet_loc[0] + bet_loc[2]) / 2
        b_mid_y = (bet_loc[1] + bet_loc[3]) / 2

        for player, player_loc in players:
            # if player_loc[1] > bet_loc[3] or player_loc[3] < bet_loc[1]:
            #     continue

            p_mid_x = (player_loc[0] + player_loc[2]) / 2
            p_mid_y = (player_loc[1] + player_loc[3]) / 2

            dist = math.sqrt((b_mid_x - p_mid_x) ** 2 + (b_mid_y - p_mid_y) ** 2)

            if dist < closest_dist:
                closest_dist = dist
                closest_player_loc = player_loc

        if closest_player_loc is not None:
            ret[closest_player_loc] = (bet_amt, bet_loc)
            # already_used_bets.add(bet_amt)

    return ret

def order_players_by_sb(
    players: list[tuple[Player, tuple[int, int, int, int]]],
    bets: dict[tuple[int, int, int, int], tuple[float, tuple[int, int, int, int]]], # player position, bet amount, bet location
    sb_amount: Union[float, None] = None,
    board_bb: tuple[int, int, int, int] = None,
) -> list[Player]:
    """
    Order players by their position relative to the dealer.

    :param players: list of tuples containing Player objects and their bounding box coordinates.
    :param bets: Dictionary mapping player locations to their bet amounts and coordinates.
    :param sb_amount: The amount of the small blind.
    :return: list of Player objects ordered by their position relative to the dealer.
    """


    if sb_amount is None:
        sb_amount = min([bet[0] for bet in bets.values()])

        # verify there is only one instance of the small blind amount
        if len([bet[0] for bet in bets.values() if bet[0] == sb_amount]) > 1:
            raise ValueError("Multiple players with the same small blind amount")

    # Identify the small blind player
    sb_player_loc = next(
        loc for loc, (amount, _) in bets.items() if amount == sb_amount
    )

    if board_bb:
        # Define the center of the table
        center_x, center_y = (board_bb[0] + board_bb[2]) // 2, (
            board_bb[1] + board_bb[3]
        ) // 2
    else:
        # get center of bounding box containing all players
        x1 = min([box[0] for _, box in players])
        y1 = min([box[1] for _, box in players])
        x2 = max([box[2] for _, box in players])
        y2 = max([box[3] for _, box in players])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Calculate angles for each player relative to the center
    def calculate_angle(box):
        x1, y1, x2, y2 = box
        player_center_x = (x1 + x2) / 2
        player_center_y = (y1 + y2) / 2
        angle = math.atan2(player_center_y - center_y, player_center_x - center_x)
        return angle

    # Get active players and calculate angles
    player_info = [(player, box, calculate_angle(box)) for player, box in players]

    # Sort players based on their angle in a clockwise manner
    player_info.sort(key=lambda item: item[2])

   
    # Rotate list to start with the small blind
    sb_index = next(
        i for i, (_, loc, _) in enumerate(player_info) if loc == sb_player_loc
    )
    ordered_players = player_info[sb_index:] + player_info[:sb_index]

    # Return only the player objects in order
    return [player for player, _, _ in ordered_players]


def associate_bets(
    players: list[tuple[Player, tuple[int, int, int, int]]],
    bets: list[tuple[float, tuple[int, int, int, int]]],
) -> dict[tuple[int, int, int, int], float]:

    ret = associate_bet_locs(players, bets)

    for player_name, info in ret.items():
        ret[player_name] = info[0]


    return ret
