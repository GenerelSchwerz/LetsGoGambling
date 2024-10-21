import math
import random

from old.PokerHelper import *


def get_threshold(board_cards, num_opponents, game_stage, pot_odds, going_all_in, current_bet, big_blind, pot_value,
                  current_stack,
                  middle_pot_value, flags):
    # odds of winning if the winner were randomly decided among all players
    lotto_chance = 1 / (1 + num_opponents)
    print(f"Chance of winning if randomly decided: {lotto_chance}")

    threatening_pot_odds_threshold = 0.33

    flush_draw = has_flush_draw(board_cards)
    if flush_draw:
        print("Flush draw on board!")
        threshold = 7
        # otherwise, the bot will wayyyyy overvalue low flushes like 4 high
    elif current_bet > middle_pot_value and game_stage != 0:
        print("Threatening bet, someone bet pot, assume 2 pair or better")
        threshold = 7
    elif current_bet > big_blind * 10 and game_stage == 0:
        print("Threatening bet, someone bet > 10 BB, assume high percentile hand")
        threshold = 7
    elif pot_odds < lotto_chance:
        print("Pot odds less than random chance")
        threshold = 8 if ((game_stage == 1 and (current_bet > 0))
                          or (game_stage == 0 and current_bet > (3 * big_blind))
                          or game_stage >= 2) else 9
    elif lotto_chance <= pot_odds < threatening_pot_odds_threshold:
        print(f"Pot odds less than {threatening_pot_odds_threshold}")
        threshold = 9 if game_stage == 0 and current_bet <= 3 * big_blind else 8
    else:
        print("Scary bet!")
        threshold = 7 if game_stage == 3 else 8

    threshold = min(flags["threshold"], threshold)

    flags["threshold"] = threshold

    return threshold


def has_flush_draw(board_cards):
    flush_draw = False
    suits = [Card.get_suit_int(card) for card in board_cards]
    for suit in suits:
        if suits.count(suit) >= 4:
            flush_draw = True
    return flush_draw


def fold_or_check(current_bet):
    if current_bet == 0:
        return "call"
    return "fold"


def make_decision(hole_cards, board_cards, stack_size, pot_value, current_bet,
                  min_bet, num_opponents, big_blind, middle_pot_value, flags):
    # convert hole cards and board cards to treys card objects using tuple_to_treys_cards
    hole_cards = [tuple_to_treys_card(card) for card in hole_cards]
    board_cards = [tuple_to_treys_card(card) for card in board_cards]

    if len(board_cards) == 0:
        game_stage = 0
    elif len(board_cards) == 3:
        game_stage = 1
    elif len(board_cards) == 4:
        game_stage = 2
    else:
        game_stage = 3

    going_all_in = current_bet >= stack_size

    pot_odds = current_bet / (pot_value + current_bet)
    print(f"Pot odds: {pot_odds}")

    bb_stack = stack_size / big_blind
    print(f"Stack size: {bb_stack} BBs ({stack_size})")

    threshold = get_threshold(board_cards, num_opponents, game_stage, pot_odds, going_all_in, current_bet, big_blind,
                              pot_value,
                              stack_size, middle_pot_value, flags)
    print(f"Threshold: {threshold}")

    play_this_hand = False

    # preflop
    if game_stage == 0:
        if going_all_in:
            if bb_stack > 20:
                if num_opponents <= 2:
                    if is_in_percentile(10, hole_cards, num_opponents > 1):
                        return "call"
                    else:
                        return fold_or_check(current_bet)
            else:
                if is_in_percentile(5, hole_cards, num_opponents > 1):
                    return "call"
                else:
                    return fold_or_check(current_bet)
        if bb_stack < 15:
            percentile = 10
        elif bb_stack < 25:
            percentile = 20
        elif bb_stack < 50:
            percentile = 30
        else:
            percentile = 40
        print(f"Playing the best {percentile}% of hands")
        if is_in_percentile(percentile, hole_cards, num_opponents > 1):
            play_this_hand = True
        else:
            if random.random() < 0.2 and bb_stack > 50:
                if is_in_percentile(60, hole_cards, num_opponents > 1):
                    play_this_hand = True
                print("20% chance to play hand outside normal percentile hit")

        if not play_this_hand:
            print("Folding hand outside percentile")
            return fold_or_check(current_bet)

    if play_this_hand or game_stage > 0:
        threshold_players = max(2, -(num_opponents // -2))
        if game_stage == 3:
            threshold_players += 1
        if threshold_players > num_opponents:
            threshold_players = num_opponents

        if game_stage > 0:
            _, hand_rank = evaluate_hand(hole_cards, board_cards)
            if hand_rank <= 7:
                if threshold == 9:
                    threshold = 8

        print(f"Calculating equity with\n\t"
              f"Opponents: {num_opponents}\n\t"
              f"Threshold: {threshold}\n\t"
              f"ThresholdPlayers: {threshold_players}")
        equity = calculate_equity(hole_cards,
                                  num_opponents,
                                  board_cards,
                                  chop_is_win=True,
                                  threshold_hand_strength=threshold,
                                  threshold_players=threshold_players)
        print(f"Equity: {equity}")

        if game_stage > 0:
            if hand_rank <= 3:
                if game_stage == 3:
                    _, board_rank = evaluate_hand([], board_cards)
                    if board_rank != 6 and board_rank != 7:
                        equity = 1
                else:
                    equity = 1

        if equity < pot_odds:
            print(f"{equity} < {pot_odds}, folding")
            return fold_or_check(current_bet)

        # calculating ideal bet becomes negative after 50% equity,
        # and you want to bet not a great amount usually anyway right?
        betting_equity = equity / 2
        if game_stage == 3 and pot_value >= 75 * big_blind:
            print("River, big pot, dont bet a lot pls")
            betting_equity = equity / 3

        if equity > 0.9:
            ideal_bet = stack_size
        elif current_bet <= big_blind and game_stage == 0:
            raise_factor = 1
            if is_in_percentile(30, hole_cards, num_opponents > 1):
                raise_factor = 3
            if is_in_percentile(10, hole_cards, num_opponents > 1):
                raise_factor = 5
            if is_in_percentile(5, hole_cards, num_opponents > 1):
                if random.random() < 0.1:
                    print("Shoving because 10% chance and top 5% hand")
                    raise_factor = stack_size / big_blind
                    equity = 1
            ideal_bet = raise_factor * big_blind
        elif equity < (1 / (num_opponents + 1) / 2):
            ideal_bet = 0
        else:
            if equity < (1 / (num_opponents + 1)):
                betting_equity = equity / 3
            # calculating the bet at which the pot odds are equal to the equity
            ideal_bet = (betting_equity * middle_pot_value) / (1 - (2 * betting_equity))
            print(f"Ideal bet: {ideal_bet}")

        if ideal_bet < 0:
            raise Exception("Ideal bet is negative")

        # if current bet is at least 75% of ideal bet, call
        if 0.75 * ideal_bet <= current_bet or ideal_bet < min_bet:
            # BLUFFS
            if current_bet == 0 and equity > 0.15:
                if pot_value <= 8 * big_blind and random.random() < (0.3 if num_opponents > 1 else 0.5):
                    print(f"Bluffing because no bet, small pot, and {10 if num_opponents > 1 else 30}% hit")
                    return f"bet {int((0.5 + (0.5 * random.random())) * pot_value)}"
                elif pot_value <= 16 * big_blind and game_stage == 3 and random.random() < (
                        0.2 if num_opponents > 1 else 0.4):
                    print(f"Bluffing because no bet, river, small pot and {20 if num_opponents > 1 else 40}% hit")
                    return f"bet {int((0.5 + (0.5 * random.random())) * pot_value)}"

            print("Calling")
            return "call"
        # if current bet is below 75% of ideal bet, raise
        if current_bet < 0.75 * ideal_bet:
            # SLOW ROLLING
            if current_bet == 0:
                if random.random() < 0.3 and (equity < 0.5 or game_stage < 2):
                    print("Slow rolling because no bet and 30% hit and <50% equity")
                    return "call"
            if ideal_bet < 0.1 * pot_value:
                print("Not worth raising")
                return "call"
            else:
                print("Raising")
                if equity < 0.9:
                    return f"bet {min(int(ideal_bet), stack_size, middle_pot_value if game_stage != 0 else stack_size)}"
                else:
                    return f"bet {min(int(ideal_bet), stack_size)}"

    raise Exception("No decision made")
