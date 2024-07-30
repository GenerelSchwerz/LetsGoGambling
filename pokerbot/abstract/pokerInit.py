from abc import ABC, abstractmethod




class PokerInitSetup(ABC):

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def login(self):
        pass

    @abstractmethod
    def start_tables(self):
        pass