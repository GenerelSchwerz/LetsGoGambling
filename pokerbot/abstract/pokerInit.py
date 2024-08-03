from abc import ABC, abstractmethod


from concurrent.futures import ProcessPoolExecutor, Future

class MultiTableSetup(ABC):

    def __init__(self) -> None:
        self.process_executor = ProcessPoolExecutor()

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def login(self):
        pass

    @abstractmethod
    def start_tables(self, amt: int = 1) -> list[Future]:
        pass

    def wait_for_all(self, futures: list[Future], timeout=None):
        """
        Bloat.
        """
        for future in futures:
            future.result(timeout=timeout)

    def shutdown(self):
        self.process_executor.shutdown()