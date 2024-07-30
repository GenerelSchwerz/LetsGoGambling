import json
import time

from ..all.abstractClient import AClient
from .base import TJPokerDetect, PSInteract, TJEventEmitter


def run_ticks(_bot: AClient, time_split=2):
    import cv2

    last_time = time.time()
    while True:
        ss = _bot.interactor._ss()

        # cv2.imwrite(f"./midrun/{int(time.time() * 100_000) / 100_000}.png", ss)

        _bot.event_handler.tick(ss)
        now = time.time()
        dur = max(0, time_split - (now - last_time))
        print(
            "Sleeping for",
            round(dur * 1000) / 1000,
            "seconds. Been",
            round((now - last_time) * 1000) / 1000,
            "seconds taken to calculate since last tick.",
            round((dur + now - last_time) * 1000) / 1000,
            "seconds total",
        )
        time.sleep(dur)
        last_time = time.time()


def launch_user(
    username: str, password: str, time_split=2, headless=False, firefox=False
):
    from ..all.algoLogic import AlgoDecisions

    detector = TJPokerDetect(username)
    detector.load_images()

    event_handler = TJEventEmitter(detector=detector)
    interactor = PSInteract(firefox=firefox, headless=headless, detector=detector)

    logic = AlgoDecisions()

    bot = AClient(
        event_handler=event_handler,
        detector=detector,
        interactor=interactor,
        logic=logic,
        debug=True,
    )

    try:
        bot.start(username, password)
        run_ticks(bot, time_split)
    except KeyboardInterrupt:
        print("bot finished")
        bot.stop()
    except Exception as e:
        import traceback

        traceback.print_exc()
        bot.stop()


def main():
    import os

    # clear midrun

    os.makedirs("./midrun", exist_ok=True)

    # for file in os.listdir("./midrun"):
    #     os.remove(f"./midrun/{file}")

    from concurrent.futures import ThreadPoolExecutor

    try:
        accounts = []

        with open("pokerbot/config.json") as config_file:
            tj_cfg = json.load(config_file)["triplejack"]
            firefox = tj_cfg["firefox"]
            accounts = tj_cfg["accounts"]
            acc_amt = tj_cfg.get("accountAmt", len(accounts))

        with ThreadPoolExecutor() as executor:

            for idx in range(acc_amt):
                info = accounts[idx]
                username = info["username"]
                password = info["password"]
                time_split = tj_cfg["secondsPerTick"]
                headless = tj_cfg["headless"]
                executor.submit(
                    launch_user, username, password, time_split, headless, firefox
                )

        # wait until all threads are done
        executor.shutdown(wait=True)

    except FileNotFoundError:
        print("Config file not found")
        return


if __name__ == "__main__":
    main()
