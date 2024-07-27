from typing import Union
import numba
import numpy as np

from treys import Card, Deck, Evaluator
from pokerbot.abstract.pokerDetection import Player
from pokerbot.all.utils import PokerHands


# TODO make everything numpy? Could probably see more performance that way.
# @numba.jit()
def fast_calculate_equity(
    hole_cards: list[Card],
    community_cards: list[Card],
    sim_time=4000,
    runs=1000,
    threshold_classes: Union[int, tuple[int, int]]=PokerHands.HIGH_CARD,
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
    while (time.time() - start) * 1000 < sim_time and run_c < runs:
        run_it += 1
        # deck needs to be modified 
        # deck = Deck(shuffle=False)
        # deck.cards = SEMI_FULL_DECK.copy()
        # deck._random.shuffle(deck.cards)

        deck = Deck()
        for card in hole_cards:
            deck.cards.remove(card)
        if community_cards is not None:
            for card in community_cards:
                deck.cards.remove(card)

        opponents_cards = [deck.draw(2) for _ in range(opponents)]
        full_board = community_cards + deck.draw(5 - board_len)

        our_rank = evaluator.evaluate(hole_cards, full_board)
        board_rank = evaluator.evaluate([], full_board)
        board_class = evaluator.get_rank_class(board_rank)

        thresh_sat = 0
        opp_count = 0
        evals = []
        for opponent in opponents_cards:
            opponent_rank = evaluator.evaluate(opponent, full_board)

            # weird amelia logic inbound (idk what this is tbh)
            if board_class >= PokerHands.PAIR:
                opponent_class = evaluator.get_rank_class(opponent_rank)

                # this code is retarded. this is literally ONLY a pair and two pair.
                if PokerHands.THREE_OF_A_KIND < threshold_class < PokerHands.HIGH_CARD:
                    adj_thresh = threshold_class - (PokerHands.HIGH_CARD - board_class)
                else:
                    adj_thresh = threshold_class

                # opponent's category is stronger than the threshold
                if opponent_class <= adj_thresh:
                    thresh_sat += 1

            # Also, isn't this just high card higher than what is on board?
            elif opponent_rank < board_rank:
                thresh_sat += 1

            evals.append(opponent_rank)
            opp_count += 1

            # exit out code
            if thresh_sat >= threshold_players:
                break

        if thresh_sat >= threshold_players:
            run_c += 1

            if is_middle:
                threshold_class = threshold_classes[int(is_lower)]
                is_lower = not is_lower

            for idx in range(opp_count, opponents):
                evals.append(evaluator.evaluate(opponents_cards[idx], full_board))

            for opp_rank in evals:
                if our_rank > opp_rank:
                    # for idx, opp_rank in enumerate(evals):
                    #     if our_rank > opp_rank:
                    #         print(f"We lost to hand {idx + 1}: {Card.ints_to_pretty_str(opponents_cards[idx])} | {Card.ints_to_pretty_str(full_board)}, {evaluator.get_rank_class(opp_rank)}, {evaluator.get_rank_class(our_rank)}")
                    break # mid level loop

            else:
                wins += 1

            

            

    print(f"Successful simulations: {run_c}, wins: {wins}, runs: {run_it}")
    return wins / run_c if run_c > 0 else 0


def calculate_equity(
    hole_cards: list[Card],
    community_cards: list[Card],
    active_opponents: int,
    simulation_time=4000,  # how much time in ms spent on calculations MAX
    num_simulations=2000,
    # num successful simulations to be satisfied & return before simulation_time is up
    threshold_hand_strength=PokerHands.HIGH_CARD,  # min strength of hands opponents will have at river
    threshold_players=1,  # min players we assume satisfy threshold_hand_strength
    # for example, with default arguments, we calculate our equity
    # (the chance that our hand beats or chops with everyone else's hands)
    # assuming that at least ONE player has HIGH_CARD or better by the river
    # (or right now, if it's the river already)
    # if threshold_hand_strength were PokerHands.THREE_OF_A_KIND and threshold_players was 2,
    # then we calculate the chance that we beat/chop with everyone,
    # assuming that AT LEAST two players have AT LEAST three of a kind
    # also supports a threshold_hand_strength of X.5, which means to average 50/50 between X and X+1
) -> float:
    evaluator = Evaluator()
    wins = 0

    if threshold_players is None:
        threshold_players = active_opponents

    start_time = time.time()
    successful_simulations = 0
    threshold = threshold_hand_strength
    ceild = False
    if threshold % 1 == 0.5:
        threshold = threshold + 0.5
        ceild = True
    while (
        time.time() - start_time < simulation_time / 1000
        and successful_simulations < num_simulations
    ):
        deck = Deck()
        for card in hole_cards:
            deck.cards.remove(card)
        if community_cards is not None:
            for card in community_cards:
                deck.cards.remove(card)

        opponent_hole_cards = [deck.draw(2) for _ in range(active_opponents)]

        threshold_satisfieds = 0

        if community_cards is None:
            raise ValueError(
                "Expecting board cards, did you mean to use calculate_equity_preflop?"
            )
        else:
            board = community_cards + deck.draw(5 - len(community_cards))

        our_rank = evaluator.evaluate(hole_cards, board)
        board_rank = evaluator.evaluate([], board)
        board_class = evaluator.get_rank_class(board_rank)

        diff = PokerHands.HIGH_CARD - board_class
        for i in range(active_opponents):
            eval_result = evaluator.evaluate(opponent_hole_cards[i], board)
            if (
                evaluator.get_rank_class(eval_result)
                <= (
                    threshold
                    - (
                        diff
                        if PokerHands.THREE_OF_A_KIND < threshold < PokerHands.HIGH_CARD
                        else 0
                    )
                )
                if board_class >= PokerHands.PAIR
                else eval_result < board_rank
            ):
                # the if-else here means if the board has two pair or better on it,
                # instead of assuming opponent has better than two pair,
                # just assume they have better than the board (so maybe a better two pair)
                threshold_satisfieds += 1
                if threshold_satisfieds >= threshold_players:
                    break

        if threshold_satisfieds >= threshold_players:
            successful_simulations += 1
            if threshold_hand_strength % 1 == 0.5:
                if ceild:
                    threshold -= 1
                    ceild = False
                else:
                    threshold += 1
                    ceild = True
            if all(
                (
                    eval_result := evaluator.evaluate(opponent_hole_cards[i], board),
                    (eval_result >= our_rank),
                )[1]
                for i in range(active_opponents)
            ):
                wins += 1
    print(
        f"Successful simulations: {successful_simulations} ({time.time() - start_time}s)"
    )
    # watch for div by 0
    return wins / successful_simulations if successful_simulations != 0 else 0


import time

start = time.time()


hole_cards = [Card.new("Ah"), Card.new("Kd")]
community_cards = [Card.new("As"), Card.new("3c"), Card.new("2h")]
sim_time = 4000
runs = 1000

num_opponents = 3
threshold_players = 1

threshold = PokerHands.TWO_PAIR

res = fast_calculate_equity(
    hole_cards,
    community_cards,
    sim_time=sim_time,
    runs=runs,
    threshold_classes=threshold,
    threshold_players=threshold_players,
    opponents=num_opponents,
)

print(res)

print(time.time() - start)

start = time.time()

res1 = calculate_equity(
    hole_cards,
    community_cards,
    active_opponents=num_opponents,
    simulation_time=sim_time,
    num_simulations=runs,
    threshold_hand_strength=threshold,
    threshold_players=threshold_players,
)

print(res1)

print(time.time() - start)
