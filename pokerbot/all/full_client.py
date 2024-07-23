import time


from ..abstract.pokerEventHandler import PokerEventHandler, PokerEvents
from ..abstract.pokerDetection import PokerDetection
from ..abstract.pokerInteract import PokerInteract
from ..abstract.pokerDecisions import PokerDecisionChoice, PokerDecisionMaking

from treys import Card

class AClient:

    currently_running = 0

    """
    This class is the main class for the poker bot.
    """

    def __init__(self, event_handler: PokerEventHandler, detector: PokerDetection, interactor: PokerInteract, logic: PokerDecisionMaking, debug=False):
        self.debug = debug

        self.detector = detector
        self.event_handler = event_handler
        self.interactor = interactor
        self.logic = logic

    

    def on_turn(self, cards: list[Card], facing_bet: int, mid_pot: int, total_pot: int):
        
        result = self.logic.on_turn(cards, facing_bet, mid_pot, total_pot)

        if result.choice == PokerDecisionChoice.FOLD:
            res = self.interactor.fold()
            if not res:
                print("fold failed")
        elif result.choice == PokerDecisionChoice.CHECK:
            res = self.interactor.check()
            if not res:
                print("check failed")

        elif result.choice == PokerDecisionChoice.CALL:
            res = self.interactor.call()
            if not res:
                print("call failed")

        elif result.choice == PokerDecisionChoice.BET:
            res = self.interactor.bet(result.amount)
            if not res:
                print("bet failed")
            

    def start(self, username: str, password: str):
        # if self.interactor is not None:
        #     self.interactor.shutdown()

        if not self.detector.is_initialized():
            self.detector.load_images()

        self.interactor.start(username, password)

        self.event_handler.on(PokerEvents.OUR_TURN, self.on_turn)



    def stop(self):
        if self.interactor is not None:
            self.interactor.shutdown()
            self.interactor = None

        self.event_handler.remove(PokerEvents.OUR_TURN, self.on_turn)


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
