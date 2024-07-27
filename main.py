from typing import Union
import numba
import numpy as np

from treys import Card, Deck, Evaluator
from pokerbot.abstract.pokerDetection import Player
from pokerbot.all.utils import PokerHands, is_in_percentile


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

    card_len = len(community_cards)
    deck = Deck()
    evaluator = Evaluator()

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

        opp_cards = [deck.draw(2) for _ in range(opponents)]

        thresh_sat = 0
        for i in range(opponents):
            if is_in_percentile(threshold_percentile, opp_cards[i], opponents > 1):
                thresh_sat += 1

                # runs once.
                if thresh_sat >= threshold_players:
                    run_c += 1
                    board = community_cards + deck.draw(5 - card_len)
                    our_rank = evaluator.evaluate(hole_cards, board)
                  
                    for i in range(opponents):
                        eval_result = evaluator.evaluate(opp_cards[i], board)
                        if eval_result < our_rank:
                            break
                    else:
                        wins += 1

                    break  # inner loop

    print(f"Successful simulations: {run_c} ({time.time() - start_time}s)")
    # watch for div by 0
    return wins / run_c if run_c != 0 else 0


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
    runs = 0
    threshold = threshold_hand_strength
    ceild = False
    if threshold % 1 == 0.5:
        threshold = threshold + 0.5
        ceild = True
    while (
        time.time() - start_time < simulation_time / 1000
        and successful_simulations < num_simulations
    ):
        runs += 1
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
            else:
                # for i in range(active_opponents):
                #     eval_result = evaluator.evaluate(opponent_hole_cards[i], board)
                #     if eval_result < our_rank:
                #         print(f"Lost to hand {i + 1}: {Card.ints_to_pretty_str(opponent_hole_cards[i])} | {Card.ints_to_pretty_str(board)}, {evaluator.get_rank_class(eval_result)}, {evaluator.get_rank_class(our_rank)}")
                pass

    print(
        f"Successful simulations: {successful_simulations} ({time.time() - start_time}s), runs: {runs}"
    )
    # watch for div by 0
    return wins / successful_simulations if successful_simulations != 0 else 0


import time

start = time.time()


hole_cards = [Card.new("Ks"), Card.new("Ts")]
community_cards = [Card.new("As"), Card.new("3h"), Card.new("2h")]
sim_time = 5000
runs = 1500

num_opponents = 1
threshold_players = 1

threshold = PokerHands.FLUSH


for i in range(9, -1, -1):
    start = time.time()

    res = fast_calculate_equity(
        hole_cards,
        community_cards,
        sim_time=sim_time,
        runs=runs,
        threshold_classes=i,
        threshold_players=threshold_players,
        opponents=num_opponents,
        opps_satisfy_thresh_now=True,
    )

    # can you add this to something outputed after the loop
    print(f"equity for i ({i}):", res, " | took", time.time() - start, "s")
    print()

    # res1 = calculate_equity(
    #     hole_cards,
    #     community_cards,
    #     active_opponents=num_opponents,
    #     simulation_time=sim_time,
    #     num_simulations=runs,
    #     threshold_hand_strength=i,
    #     threshold_players=threshold_players,
    # )

    # print(f"[OLD] equity for i ({i}):", res1, " | took", time.time() - start, "s")
    # print()

start = time.time()
res2 = calculate_equity_preflop(
    hole_cards,
    num_opponents,
    # community_cards=community_cards,
    sim_time=sim_time,
    runs=runs,
    threshold_percentile=60,
    threshold_players=threshold_players,
)

print(f"equity for preflop:", res2, " | took", time.time() - start, "s")
