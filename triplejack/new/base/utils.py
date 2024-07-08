def suit_full_name_to_abbrev(suit_full_name: str) -> str:
    if suit_full_name == "hearts":
        return "h"
    elif suit_full_name == "diamonds":
        return "d"
    elif suit_full_name == "clubs":
        return "c"
    elif suit_full_name == "spades":
        return "s"
    
    raise ValueError("Invalid suit name")
    

def card_to_abbrev(card: str) -> str:
    if card == "10":    
        return "T"
    return card