import time


from ..abstract.pokerEventHandler import PokerEventHandler, PokerEvents
from ..abstract.pokerDetection import PokerDetection
from ..abstract.pokerInteract import PokerInteract
from ..abstract.pokerDecisions import PokerDecisionChoice, PokerDecisionMaking
from ..abstract.pokerInit import MultiTableSetup


from treys import Card


class AClient:

    currently_running = 0

    """
    This class is the main class for the poker bot.
    """

    def __init__(
        self,
        event_handler: PokerEventHandler,
        detector: PokerDetection,
        interactor: PokerInteract,
        logic: PokerDecisionMaking,
        debug=False,
    ):
        self.debug = debug

        self.detector = detector
        self.event_handler = event_handler
        self.interactor = interactor
        self.logic = logic

        self.cur_hand_bb = 0
        self.cur_hand_sb = 0

    # fuck around functions
    def fold(self):
        res = self.interactor.fold()
        if not res:
            check = self.interactor.check()
            if not check:
                print("fold failed")
            else:
                print("fold failed, but check succeeded")
        else:
            print("fold succeeded")

    def check(self):
        res = self.interactor.check()
        if not res:
            call = self.interactor.call()
            if not call:
                print("check failed")
            else:
                print("check failed, but call succeeded")
        else:
            print("check succeeded")

    def call(self):
        res = self.interactor.call()
        if not res:
            check = self.interactor.check()
            if not check:
                print("call failed")
            else:
                print("call failed, but check succeeded")
        else:
            print("call succeeded")

    def bet(self, amount: int):
        res = self.interactor.bet(amount, self.cur_hand_sb, self.cur_hand_bb)
        if not res:
            raise_ = self.interactor.reraise(amount)
            if not raise_:
                print("bet failed")
            else:
                print("bet failed, but raise succeeded")
        else:
            print("bet succeeded")

    def on_new_hand(self, hole_cards: list[Card], bb: int, sb: int):
        print("New hand", Card.ints_to_pretty_str(hole_cards), sb, bb)
        self.cur_hand_bb = bb
        self.cur_hand_sb = sb

    def on_turn(
        self,
        hole_cards: list[Card],
        community_cards: list[Card],
        stage: int,
        facing_bet: int,
        min_bet: int,
        mid_pot: int,
        total_pot: int,
        stack_size: int,
        active_opponents: int,
    ):
        print("Our turn")
        start = time.time()
        result = self.logic.on_turn(
            hole_cards,
            community_cards,
            stage,
            facing_bet,
            min_bet,
            mid_pot,
            total_pot,
            self.cur_hand_bb,
            stack_size,
            active_opponents,
        )

        if result.choice == PokerDecisionChoice.FOLD:
            self.fold()
        elif result.choice == PokerDecisionChoice.CHECK:
            self.check()
        elif result.choice == PokerDecisionChoice.CALL:
            self.call()
        elif result.choice == PokerDecisionChoice.BET:
            self.bet(result.amount)

        print("Time taken for decision:", time.time() - start)

    def start(self):
        # if self.interactor is not None:
        #     self.interactor.shutdown()

        if not self.detector.is_initialized():
            self.detector.load_images()

        self.event_handler.on(PokerEvents.OUR_TURN, self.on_turn)
        self.event_handler.on(PokerEvents.NEW_HAND, self.on_new_hand)

    def stop(self):
        if self.interactor is not None:
            self.interactor.shutdown()
            self.interactor = None

        self.event_handler.remove(PokerEvents.OUR_TURN, self.on_turn)
        self.event_handler.remove(PokerEvents.NEW_HAND, self.on_new_hand)



class AMultiTableClient:
    def __init__(self, initializer: MultiTableSetup) -> None:
        pass

def main():
    bot = AClient()

    try:
        bot.start()
    except KeyboardInterrupt:
        print("bot finished")
        bot.stop()
    except Exception as e:
        import traceback

        traceback.print_exc()
        bot.stop()


if __name__ == "__main__":
    main()
