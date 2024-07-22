import time

from treys import Card, Evaluator, Deck
from poker.hand import Range
from poker.card import Card as PokerCard


# hand rankings:
# 9: high card
# 8: one pair
# 7: two pair
# 6: three of a kind
# 5: straight
# 4: flush
# 3: full house
# 2: four of a kind
# 1: straight flush
# 0: royal flush


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

poker_ranks = '23456789TJQKA'
poker_suits = 'shdc'

lookup_table = {}

for r in poker_ranks:
    for s in poker_suits:
        card_str = r + s
        treys_card = Card.new(card_str)
        lookup_table[treys_card] = card_str


# expects hole_cards as a list of two 2-tuples, where cards are represented like (4, 'c') for 4 of clubs
# or (13, 'h') for the king of hearts
# expects community_cards to be a list of 5 similarly represented cards
def evaluate_hand_from_string(hole_cards, community_cards):
    # Define a mapping from integer ranks to character ranks
    int_to_char_rank = {14: 'A', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}
    # For ranks 2-9, the character rank is the same as the integer rank
    int_to_char_rank.update({i: str(i) for i in range(2, 10)})

    # Convert hole cards and community cards to treys.Card objects
    hole_cards = [Card.new(f"{int_to_char_rank[hole_card[0]]}{hole_card[1]}") for hole_card in hole_cards]
    community_cards = [Card.new(f"{int_to_char_rank[community_card[0]]}{community_card[1]}") for community_card
                       in community_cards]

    return evaluate_hand(community_cards, hole_cards)


def tuple_to_treys_card(tup):
    int_to_char_rank = {14: 'A', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}
    int_to_char_rank.update({i: str(i) for i in range(2, 10)})
    return Card.new(f"{int_to_char_rank[tup[0]]}{tup[1]}")


# returns hand rank from 1-7000ish and hand strength from 0-9
def evaluate_hand(community_cards, hole_cards):
    # create an evaluator object
    evaluator = Evaluator()
    # get the best 5-card hand for the player
    player_hand = evaluator.evaluate(hole_cards, community_cards)
    # get the player's hand strength
    player_hand_strength = evaluator.get_rank_class(player_hand)
    return player_hand, player_hand_strength


def decide_winner(hole_cards1, hole_cards2, community_cards):
    player1_hand, player1_hand_strength = evaluate_hand(hole_cards1, community_cards)
    player2_hand, player2_hand_strength = evaluate_hand(hole_cards2, community_cards)
    if player1_hand_strength < player2_hand_strength:
        return 1
    elif player1_hand_strength > player2_hand_strength:
        return 2
    else:
        return 0


def had_pair_or_draw_on_flop(hole_cards, board, evaluator):
    # returns true if the player had at least pair or a draw on the flop or two overcards
    flop = board[:3]
    rank = evaluator.evaluate(hole_cards, flop)
    rank_class = evaluator.get_rank_class(rank)
    least_a_pair = rank_class <= 8
    if least_a_pair:
        return True

    # overcards meaning values greater than all board cards
    overcards = all(Card.get_rank_int(hole_card) > max([Card.get_rank_int(board_card) for board_card in flop])
                    for hole_card in hole_cards)
    if overcards:
        return True

    # flush draw
    flush_draw = False
    combined = hole_cards + flop
    suits = [Card.get_suit_int(card) for card in combined]
    for suit in range(4):  # 4 suits: 0, 1, 2, 3
        if suits.count(suit + 1) == 4:  # Check for 4 cards of the same suit
            flush_draw = True
    if flush_draw:
        return True

    # straight draw, only takes into account open enders
    straight_draw = False
    combined = hole_cards + flop
    ranks = [Card.get_rank_int(c) for c in combined]
    ranks = list(set(ranks))  # Remove duplicates
    ranks.sort()
    for i in range(len(ranks) - 3):
        if ranks[i + 3] - ranks[i] == 3:
            straight_draw = True
    if straight_draw:
        return True
    return False


def calculate_equity(hole_cards, num_opponents, community_cards=None, simulation_time=4000, chop_is_win=False,
                     threshold_hand_strength=9, threshold_players=None, num_simulations=2000):
    evaluator = Evaluator()
    wins = 0

    if threshold_players is None:
        threshold_players = num_opponents

    start_time = time.time()
    # range_time = 0
    # winner_time = 0
    successful_simulations = 0
    # check if simulation_time ms have passed
    while time.time() - start_time < simulation_time / 1000 and successful_simulations < num_simulations:
        # winner_start_time = time.time()
        deck = Deck()
        # winner_time += time.time() - winner_start_time
        for card in hole_cards:
            deck.cards.remove(card)
        if community_cards is not None:
            for card in community_cards:
                deck.cards.remove(card)

        opponent_hole_cards = [deck.draw(2) for _ in range(num_opponents)]

        if community_cards is None or len(community_cards) == 0:
            game_stage = 0
        elif len(community_cards) == 3:
            game_stage = 1
        elif len(community_cards) == 4:
            game_stage = 2
        else:
            game_stage = 3

        if threshold_hand_strength == 9:
            percentile = 40
        elif threshold_hand_strength == 8:
            percentile = 20
        else:
            percentile = 5

        threshold_satisfieds = 0

        if game_stage == 0:
            # start_range_time = time.time()
            for i in range(num_opponents):
                if is_in_percentile(percentile, opponent_hole_cards[i], num_opponents > 1):
                    threshold_satisfieds += 1
                    if threshold_satisfieds >= threshold_players:
                        break
            # range_time += time.time() - start_range_time
            if threshold_satisfieds >= threshold_players:
                board = community_cards + deck.draw(5 - len(community_cards))
            else:
                continue
        else:
            if community_cards is None:
                board = deck.draw(5)
            else:
                board = community_cards + deck.draw(5 - len(community_cards))

        our_rank = evaluator.evaluate(hole_cards, board)

        if game_stage != 0:
            board_rank = evaluator.evaluate([], board)
            board_class = evaluator.get_rank_class(board_rank)
            diff = 9 - board_class
            for i in range(num_opponents):
                eval_result = evaluator.evaluate(opponent_hole_cards[i], board)
                if (evaluator.get_rank_class(eval_result) <= threshold_hand_strength - diff
                if board_class >= 8 else eval_result < board_rank):
                    # the if-else here means if the board has two pair or better on it,
                    # instead of assuming opponent has better than two pair,
                    # just assume he has better than the board (so maybe a better two pair)
                    # and (eval_result >= our_rank if chop_is_win else eval_result > our_rank)
                    # and had_pair_or_draw_on_flop(opponent_hole_cards[i], board,
                    #                              evaluator) if not is_flop else True):
                    threshold_satisfieds += 1
                    if threshold_satisfieds >= threshold_players:
                        break

        if threshold_satisfieds >= threshold_players:
            successful_simulations += 1
            if all((eval_result := evaluator.evaluate(opponent_hole_cards[i], board),
                    (eval_result >= our_rank if chop_is_win else eval_result > our_rank)
                    )[1] for i in range(num_opponents)):
                wins += 1
    print(f"Successful simulations: {successful_simulations} ({time.time() - start_time}s)")
    # watch for div by 0
    return wins / successful_simulations if successful_simulations != 0 else 0


def is_in_range(hand_range, hole_cards):
    # Convert treys Card objects to poker Card objects
    hole_cards_poker = treys_to_poker(hole_cards)
    hole_cards_poker = "".join(hole_cards_poker)
    value = hole_cards_poker in hand_range
    return value


def treys_to_poker(cards):
    # use lookup_table
    return [lookup_table[card] for card in cards]


def is_in_percentile(percentile, hole_cards, multiple_opponents=True):
    return is_in_range(
        perc_ranges_multiple_ops[percentile] if multiple_opponents else perc_ranges_single_op[percentile],
        hole_cards)
