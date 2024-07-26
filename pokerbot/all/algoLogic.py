import logging
import math
import time

from treys import Card, Deck
import random
from ..abstract.pokerDecisions import PokerDecisionMaking, PokerDecisionChoice
from ..abstract.pokerEventHandler import PokerStages
from ..all.utils import *

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s|%(name)s]: %(message)s',
    datefmt='%m-%d %H:%M:%S'
)
log = logging.getLogger("BotmeliaLogic")


class AlgoDecisionMode:
    HOLDEM = 0
    OMAHA = 1


def calculate_equity(hole_cards: list[Card],
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
    while time.time() - start_time < simulation_time / 1000 and successful_simulations < num_simulations:
        deck = Deck()
        for card in hole_cards:
            deck.cards.remove(card)
        if community_cards is not None:
            for card in community_cards:
                deck.cards.remove(card)

        opponent_hole_cards = [deck.draw(2) for _ in range(active_opponents)]

        threshold_satisfieds = 0

        if community_cards is None:
            raise ValueError("Expecting board cards, did you mean to use calculate_equity_preflop?")
        else:
            board = community_cards + deck.draw(5 - len(community_cards))

        our_rank = evaluator.evaluate(hole_cards, board)
        board_rank = evaluator.evaluate([], board)
        board_class = evaluator.get_rank_class(board_rank)

        diff = PokerHands.HIGH_CARD - board_class
        for i in range(active_opponents):
            eval_result = evaluator.evaluate(opponent_hole_cards[i], board)
            if (evaluator.get_rank_class(eval_result) <=
                (threshold - (diff if PokerHands.THREE_OF_A_KIND < threshold < PokerHands.HIGH_CARD else 0))
            if board_class >= PokerHands.PAIR else eval_result < board_rank):
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
            if all((eval_result := evaluator.evaluate(opponent_hole_cards[i], board),
                    (eval_result >= our_rank)
                    )[1] for i in range(active_opponents)):
                wins += 1
    log.info(f"Successful simulations: {successful_simulations} ({time.time() - start_time}s)")
    # watch for div by 0
    return wins / successful_simulations if successful_simulations != 0 else 0


def calculate_equity_preflop(hole_cards: list[Card],
                             community_cards: list[Card],
                             active_opponents: int,
                             simulation_time=4000,  # how much time in ms spent on calculations MAX
                             num_simulations=2000,
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
    evaluator = Evaluator()
    wins = 0

    if threshold_players is None:
        threshold_players = active_opponents

    start_time = time.time()
    successful_simulations = 0
    percentile = threshold_percentile
    while time.time() - start_time < simulation_time / 1000 and successful_simulations < num_simulations:
        deck = Deck()
        for card in hole_cards:
            deck.cards.remove(card)
        if community_cards is not None:
            for card in community_cards:
                deck.cards.remove(card)

        opponent_hole_cards = [deck.draw(2) for _ in range(active_opponents)]

        threshold_satisfieds = 0
        for i in range(active_opponents):
            if is_in_percentile(percentile, opponent_hole_cards[i], active_opponents > 1):
                threshold_satisfieds += 1
                if threshold_satisfieds >= threshold_players:
                    break
        if threshold_satisfieds >= threshold_players:
            board = community_cards + deck.draw(5 - len(community_cards))
        else:
            continue

        our_rank = evaluator.evaluate(hole_cards, board)

        if threshold_satisfieds >= threshold_players:
            successful_simulations += 1
            if all((eval_result := evaluator.evaluate(opponent_hole_cards[i], board),
                    (eval_result >= our_rank)
                    )[1] for i in range(active_opponents)):
                wins += 1
    log.info(f"Successful simulations: {successful_simulations} ({time.time() - start_time}s)")
    # watch for div by 0
    return wins / successful_simulations if successful_simulations != 0 else 0


class AlgoDecisions(PokerDecisionMaking):

    def __init__(self, mode=int):
        self.mode = mode
        self.currently_bluffing = False
        self.currently_betting = False
        self.was_reraised = False
        self.min_threshold = PokerHands.HIGH_CARD
        self.hole_card_1 = None
        self.hole_card_2 = None
        self.current_street = PokerStages.UNKNOWN

    # this method is only called post flop
    def get_threshold(self,
                      board_cards,
                      num_opponents,
                      game_stage,
                      pot_odds,
                      current_bet,
                      big_blind,
                      pot_value,
                      middle_pot_value,
                      hand_strength):
        # odds of winning if the winner were randomly decided among all players
        lotto_chance = 1 / (1 + num_opponents)
        log.info(f"Chance of winning if randomly decided: {lotto_chance}")

        flush_draw = has_flush_draw(board_cards)
        if flush_draw:
            # without all this, the bot will wayyyyy overvalue low flushes like 4 high
            log.info(">=4 of one suit on board! Uh oh!")
            if current_bet > 0:
                threshold = PokerHands.THREE_OF_A_KIND
                if self.was_reraised:
                    log.info("We've been reraised, assume flush")
                    threshold = PokerHands.STRAIGHT  # eh they could also have a straight
            else:
                threshold = PokerHands.TWO_PAIR

        elif current_bet >= middle_pot_value and game_stage > PokerStages.PREFLOP:
            log.info("Bet is >=pot postflop, assume someone has 2 pair or better")
            threshold = PokerHands.TWO_PAIR
        elif pot_odds < lotto_chance:
            log.info("Pot odds are less than random chance")
            threshold = PokerHands.PAIR
            if game_stage == PokerStages.FLOP and current_bet <= big_blind:
                threshold = PokerHands.HIGH_CARD
                log.info("I know y'all did not hit this board...")
        else:
            log.info("Pot odds greater than random chance")
            threshold = PokerHands.PAIR
            if game_stage == 3:
                # if we're all the way at the river and seeing a hefty bet,
                # assume decent but not amazing hand
                threshold = PokerHands.merge(PokerHands.PAIR, PokerHands.TWO_PAIR)

        if hand_strength <= PokerHands.TWO_PAIR:  # if we have a good hand, don't get too overzealous
            if threshold == PokerHands.HIGH_CARD:
                threshold = PokerHands.PAIR

        elif threshold >= PokerHands.PAIR and self.was_reraised and current_bet > big_blind * 2:
            log.info("Was reraised, assume 2 pair or better 50% of the time")
            threshold = min(PokerHands.merge(PokerHands.PAIR, PokerHands.TWO_PAIR), threshold)

        elif (game_stage == PokerStages.RIVER
              and middle_pot_value >= 50 * big_blind
              or current_bet >= 50 * big_blind):
            log.info("River, either >= 50 BB pot or >= 50 BB bet, assume 2 pair or better 50% of the time")
            threshold = min(PokerHands.merge(PokerHands.PAIR, PokerHands.TWO_PAIR), threshold)

        # idk why this is here i think it's because this is the best approximation i have for assuming top pair?
        elif current_bet >= 0.5 * middle_pot_value:
            log.info("Bet over half pot, assume 2 pair or better 50% of the time")
            threshold = min(PokerHands.merge(PokerHands.PAIR, PokerHands.TWO_PAIR), threshold)

        if threshold > self.min_threshold:
            log.info(
                f"Threshold ({threshold}) is higher than min threshold ({self.min_threshold}), setting it to that")
        threshold = min(self.min_threshold, threshold)

        self.min_threshold = threshold

        return threshold

    def make_preflop_decision(self,
                              going_all_in: bool,
                              bb_stack: int,
                              active_opponents: int,
                              hole_cards: list[Card],
                              community_cards: list[Card],
                              facing_bet: int,
                              big_blind: int,
                              stack_size: int,
                              min_bet: int,
                              pot_odds: float) -> PokerDecisionChoice:
        play_this_hand = False
        if going_all_in:
            if bb_stack > 20:
                if active_opponents <= 2:
                    if is_in_percentile(10, hole_cards, active_opponents > 1):
                        return PokerDecisionChoice.call()
                    else:
                        return fold_or_check(facing_bet)
            if is_in_percentile(5, hole_cards, active_opponents > 1):
                return PokerDecisionChoice.call()
            else:
                return fold_or_check(facing_bet)

        percentile = 40
        if bb_stack < 50:
            percentile = 30
        if bb_stack < 25:
            percentile = 20
        if bb_stack < 15:
            percentile = 10
        log.info(f"Playing the best {percentile}% of hands")
        if is_in_percentile(percentile, hole_cards, active_opponents > 1):
            play_this_hand = True
        else:
            if random.random() < 0.5 and bb_stack > 50:
                if is_in_percentile(60, hole_cards, active_opponents > 1):
                    play_this_hand = True
                    log.info("50% chance to play hand outside normal percentile hit")
        if not play_this_hand:
            log.info("Folding hand outside percentile")
            return fold_or_check(facing_bet)

        if facing_bet <= big_blind:  # if no one has raised pre yet, we're gonna raise, bc that's good poker mmkay?
            raise_factor = 1
            if is_in_percentile(5, hole_cards, active_opponents > 1):
                raise_factor = 6
                # TRIPLEJACK SPECIFIC
                if random.random() < 0.2:
                    log.info("Shoving because 20% chance hit and top 5% hand")
                    raise_factor = stack_size / big_blind
                if active_opponents >= 5 and random.random() < 0.3:
                    log.info("Shoving because 50% chance, >=5 ops, and top 5% hand")
                    raise_factor = stack_size / big_blind
                # / TRIPLEJACK SPECIFIC
            elif is_in_percentile(10, hole_cards, active_opponents > 1):
                raise_factor = 5
            elif is_in_percentile(30, hole_cards, active_opponents > 1):
                raise_factor = 3
            ideal_bet = raise_factor * big_blind
            if ideal_bet >= min_bet:
                log.info(f"Preflop raise to {ideal_bet}")
                return bet(ideal_bet, stack_size)
            else:
                if facing_bet == 0:
                    log.info("Preflop check")
                    return PokerDecisionChoice.check()
                else:
                    log.info("Preflop call")
                    return PokerDecisionChoice.call()

        # rest of preflop logic
        if facing_bet > 10 * big_blind:
            equity = calculate_equity_preflop(hole_cards, community_cards, active_opponents,
                                              threshold_percentile=10,
                                              threshold_players=1 if active_opponents < 4 else 2)
        elif facing_bet > 3 * big_blind:
            equity = calculate_equity_preflop(hole_cards, community_cards, active_opponents,
                                              threshold_percentile=20,
                                              threshold_players=1 if active_opponents < 4 else 2)
        else:
            equity = calculate_equity_preflop(hole_cards, community_cards, active_opponents,
                                              threshold_percentile=60,
                                              threshold_players=2 if active_opponents < 4 else 3)
        log.info(f"Preflop equity: {equity}")
        if equity < pot_odds:
            log.info(f"Preflop equity less than pot odds, folding ({equity} < {pot_odds})")
            return fold_or_check(facing_bet)
        else:
            log.info(f"Preflop equity greater than pot odds, calling ({equity} > {pot_odds})")
            return PokerDecisionChoice.call()

    def on_turn(self,
                hole_cards: list[Card],
                community_cards: list[Card],
                facing_bet: int,
                min_bet: int,
                mid_pot: int,
                total_pot: int,
                big_blind: int,
                stack_size: int,
                active_opponents: int,
                ) -> PokerDecisionChoice:

        print("Our turn!", Card.ints_to_pretty_str(hole_cards), Card.ints_to_pretty_str(community_cards),
              f"facing bet: {facing_bet}, min bet: {min_bet}, mid pot: {mid_pot}, total pot: {total_pot}, big blind: {big_blind}, stack size: {stack_size}, active opponents: {active_opponents}")

        current_street = get_street(community_cards)

        # new hand detection, not necessary if we do something with event listeners, but i'm just porting over code rn

        if (self.current_street > current_street
                or (self.hole_card_1 != hole_cards[0] or self.hole_card_2 != hole_cards[1])):
            log.info("New hand detected")
            self.min_threshold = PokerHands.HIGH_CARD
            self.currently_bluffing = False
            self.currently_betting = False

        self.was_reraised = False
        if self.current_street == current_street and self.currently_betting:
            self.was_reraised = True

        self.current_street = current_street
        self.hole_card_1 = hole_cards[0]
        self.hole_card_2 = hole_cards[1]

        # approximate the pot so that when someone bets right in front of you,
        # you can assume some of the players behind you are gonna call.
        # this paints a more accurate picture of pot odds

        approximate_pot = True
        if community_cards == [] and facing_bet >= 10 * big_blind:
            approximate_pot = False
        if facing_bet >= stack_size:
            approximate_pot = False
        if facing_bet > big_blind * 5 and facing_bet >= mid_pot:
            approximate_pot = False

        if approximate_pot:
            approximate_pot = math.floor(((active_opponents + 1) * facing_bet) /
                                         (1.5 if facing_bet != big_blind else 1))
            if total_pot < approximate_pot:
                log.info(f"Assuming some callers behind, pot value ({total_pot}) "
                         f"less than {approximate_pot}, setting it to that")
                total_pot = approximate_pot

        ###

        going_all_in = facing_bet >= stack_size
        pot_odds = facing_bet / (total_pot + facing_bet)
        log.info(f"Pot odds: {pot_odds}")

        # This is triplejack specific. People raise pre with shit hands. This would not fly on coinpoker.
        # TRIPLEJACK SPECIFIC
        if self.current_street == PokerStages.PREFLOP and facing_bet <= 5 * big_blind:
            pot_odds = pot_odds / 2
            log.info(f"Treating pot odds as {pot_odds} because preflop and low bet")
        # / TRIPLEJACK SPECIFIC

        print("stack size", stack_size, "big_blind", big_blind)
        bb_stack = stack_size // big_blind
        log.info(f"Stack size: {bb_stack} BBs ({stack_size})")

        ### PREFLOP DECISION MAKING

        if self.current_street == PokerStages.PREFLOP:
            return self.make_preflop_decision(going_all_in,
                                              bb_stack,
                                              active_opponents,
                                              hole_cards,
                                              community_cards,
                                              facing_bet,
                                              big_blind,
                                              stack_size,
                                              min_bet,
                                              pot_odds)

        ### POSTFLOP DECISION MAKING BELOW \/

        _, our_hand_strength = evaluate_hand(hole_cards, community_cards)

        threshold = self.get_threshold(community_cards, active_opponents, current_street,
                                       pot_odds, facing_bet, big_blind, total_pot, mid_pot, our_hand_strength)

        # setting threshold_players, the min number of players we assume satisfy threshold hand_strength

        threshold_players = min(max(2, ceil_half(active_opponents)), 3)
        if threshold <= PokerHands.TWO_PAIR:
            threshold_players = 1
        if self.current_street == PokerStages.RIVER:
            threshold_players += 1
        if threshold_players > active_opponents:
            threshold_players = active_opponents

        # calculating equity, with two special cases where equity should be 100% to save calculation

        if our_hand_strength == PokerHands.FULL_HOUSE:
            if self.current_street == PokerStages.RIVER:
                _, board_rank = evaluate_hand([], community_cards)
                if (board_rank != PokerHands.TWO_PAIR
                        and board_rank != PokerHands.THREE_OF_A_KIND
                        and board_rank != PokerHands.FULL_HOUSE):
                    log.info("River, fh when no 2pair, 3kind, fh on board, equity is 100%")
                    equity = 1
            elif self.current_street == PokerStages.TURN:
                if (not has_three_of_a_kind(community_cards)
                        and not has_two_pair(community_cards)):
                    log.info("Turn, fh when no 2pair, 3kind on board, equity is 100%")
                    equity = 1
        elif our_hand_strength < PokerHands.FULL_HOUSE:
            log.info("4 of a kind or better, equity is 100%")
            equity = 1
        else:
            log.info(f"Calculating equity with\n\t"
                     f"Opponents: {active_opponents}\n\t"
                     f"Threshold: {threshold}\n\t"
                     f"Threshold players: {threshold_players}\n\t")

            equity = calculate_equity(hole_cards, community_cards, active_opponents,
                                      threshold_hand_strength=threshold,
                                      threshold_players=threshold_players)
        log.info(f"Equity: {equity}")

        # comparing equity and pot odds

        if equity < pot_odds:
            log.info("Equity less than pot odds, folding")
            if self.current_street == PokerStages.RIVER and facing_bet == big_blind:
                log.info("River and facing min bet, calling")
                return PokerDecisionChoice.call()
            return fold_or_check(facing_bet)
        elif pot_odds > 0:
            log.info(f"{equity} > {pot_odds}, playing")

        # finding the equity to size bet with (betting_equity), special cases described in log() statements

        if (self.current_street == PokerStages.RIVER
                and total_pot >= 50 * big_blind
                and active_opponents <= 2
                and threshold >= PokerHands.PAIR):
            betting_equity = calculate_equity(hole_cards, community_cards, active_opponents,
                                              threshold_hand_strength=PokerHands.merge(PokerHands.PAIR,
                                                                                       PokerHands.TWO_PAIR),
                                              threshold_players=1)
            log.info(f"River, >=50BB pot, <=2 ops, calculate betting equity assuming two pair/pair 50/50 split ({betting_equity})")
            # betting_equity = betting_equity / 2
            # ^ this was in original code but seems rather timid?
        elif equity < 0.7 and self.current_street == PokerStages.RIVER and total_pot >= 50 * big_blind:
            log.info("River, >=50BB pot, <70% equity, don't bet a lot")
            betting_equity = equity / 3
        else:  # general case
            # calculating ideal bet becomes negative after 50% equity,
            # and you want to bet not a great amount usually anyway bc low variance or smth right?
            betting_equity = equity / 2

            if self.currently_betting and self.current_street == PokerStages.FLOP and facing_bet == 0:
                log.info("C-betting")  # not really a c bet more like bet more than usual if c bet criteria is met
                # cause the bot already likes betting flops
                betting_equity = min(0.4, equity)  # the 0.4 is bc of infinity vertical asymptote at 0.5

        # calculating ideal bet from betting equity

        if equity > 0.9:  # shove if we're prolly gonna win
            ideal_bet = stack_size
        elif equity < (1 / (active_opponents + 1) / 2):
            # if we expect to win less than half the chance of randomly winning, yeah don't bet lol
            log.info("I REALLY don't think we're winning this one boys, dont bet")
            ideal_bet = 0
        else:
            if equity < (1 / (active_opponents + 1)) and self.current_street > PokerStages.FLOP:
                # if we expect to win less than often than random chance, bet less, unless it's the flop
                log.info("I don't think we're winning this one boys, bet less")
                betting_equity = betting_equity / 2

            # calculating the bet at which the pot odds are equal to the equity
            ideal_bet = (betting_equity * mid_pot) / (1 - (2 * betting_equity))  # you can derive this equation yourself
            log.info(f"Ideal bet: {ideal_bet}")
        if ideal_bet < 0:
            raise Exception("Ideal bet is negative")

        # now that we have ideal bet, let's decide between calling and raising

        # if current bet is at least 75% of ideal bet, call
        if 0.75 * ideal_bet <= facing_bet or ideal_bet < min_bet:

            # BLUFFS
            if facing_bet == 0:
                bluff_frequency = 0.3 * (
                    2 if self.currently_bluffing else 1) + 0.1 if self.current_street != PokerStages.PREFLOP else 0
                if total_pot <= max(0.025 * stack_size, 12) * big_blind and random.random() < bluff_frequency:
                    log.info(f"Bluffing because no bet, small pot, and {bluff_frequency * 100}% hit")
                    self.currently_bluffing = True
                    return PokerDecisionChoice.bet(int((0.5 + (0.5 * random.random())) * total_pot))
                elif (total_pot <= max(0.05 * stack_size, 16) * big_blind
                      and self.current_street == PokerStages.RIVER and random.random() < bluff_frequency):
                    log.info(f"Bluffing because no bet, river, small pot and {bluff_frequency * 100}% hit")
                    self.currently_bluffing = True
                    return PokerDecisionChoice.bet(int((0.5 + (0.5 * random.random())) * total_pot))

            if facing_bet == 0:
                log.info("Checking")
                return PokerDecisionChoice.check()
            else:
                log.info("Calling")
                return PokerDecisionChoice.call()
            
        # if current bet is below 75% of ideal bet, raise
        if facing_bet < 0.75 * ideal_bet:
            # SLOW ROLLING
            if facing_bet == 0:
                if random.random() < 0.3 and equity < 0.5:
                    log.info("Slow rolling because no bet and 30% hit and <50% equity")
                    ideal_bet = ideal_bet / 2

            if ideal_bet < 0.1 * total_pot:
                log.info("Not worth raising")
                return PokerDecisionChoice.call()
            else:
                log.info("Raising")
                if equity < 0.9:
                    return PokerDecisionChoice.bet(min(int(ideal_bet), stack_size, mid_pot))
                    # don't bet more than pot ever, helps to keep variance low with a kinda dumb bot
                else:
                    return PokerDecisionChoice.bet(min(int(ideal_bet), stack_size))

        raise Exception("No decision made")
