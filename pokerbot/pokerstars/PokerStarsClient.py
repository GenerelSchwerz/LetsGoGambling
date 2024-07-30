import json
import time

from ..abstract.pokerEventHandler import PokerEvents

from ..all.abstractClient import AClient
from .base import PSPokerImgDetect, PSInteract, PSEventEmitter


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


def _print(*args):
    print(*args)

def launch_user(
    username: str, password: str, exec_path: str, time_split=2, 
):
    from ..all.algoLogic import AlgoDecisions

    detector = PSPokerImgDetect()
    detector.load_images()

    event_handler = PSEventEmitter(detector=detector)
    interactor = PSInteract(detector=detector, path_to_exec=exec_path)

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
        bot.event_handler.on(PokerEvents.TICK, _print)
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

    for file in os.listdir("./midrun"):
        os.remove(f"./midrun/{file}")

   

    try:
     
        with open("pokerbot/config.json") as config_file:
            ps_cfg = json.load(config_file)["pokerstars"]
            acc = ps_cfg["account"]
            exec_path = ps_cfg["exec_path"]


        launch_user(acc["username"], acc["password"], exec_path)

    except FileNotFoundError:
        print("Config file not found")
        return


if __name__ == "__main__":
    main()
