

import time

from ..all.full_client import AClient
from .base import TJPokerDetect, TJInteract, TJEventEmitter


def run_ticks(_bot):
    time_split = 2
    last_time = time.time()
    while True:

        _bot.ss = _bot.interactor._ss()
        _bot.event_handler.tick(_bot.ss)
        now = time.time()
        dur = max(0, time_split - (now - last_time))
        print("Sleeping for", round(dur * 1000) / 1000, "seconds. Been", round((now - last_time) * 1000) / 1000, "seconds taken to calculate since last tick.",round(( dur + now - last_time) * 1000) / 1000, "seconds total")
        time.sleep(dur)
        last_time = time.time()

def main():

    from ..all.simpleLogic import SimpleDecisions

    detector = TJPokerDetect()
    detector.load_images()

    event_handler = TJEventEmitter(detector=detector)
    interactor = TJInteract(headless=False, detector=detector)

    logic = SimpleDecisions()

    bot = AClient(
        event_handler=event_handler,
        detector=detector,
        interactor=interactor,
        logic=logic,
        debug=True
    )



    try:
        bot.start("ForTheChips", "WooHoo123!")

        run_ticks(bot)
    except KeyboardInterrupt:
        print("bot finished")
        bot.stop()
    except Exception as e:
        import traceback

        traceback.print_exc()
        bot.stop()


if __name__ == "__main__":
    main()
