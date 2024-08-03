from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


@dataclass
class Player:
    name: str
    stack: int
    active: bool


def order_players_by_sb(
    players: list[tuple[Player, tuple[int, int, int, int]]],
    bets: dict[tuple[int, int, int, int], tuple[float, tuple[int, int, int, int]]], # player position, bet amount, bet location
    sb_amount: Union[float, None] = None,
    board_bb: tuple[int, int, int, int] = None,
) -> list[Player]:
    """
    Order players by their position relative to the dealer.

    :param players: list of tuples containing Player objects and their bounding box coordinates.
    :param bets: Dictionary mapping player locations to their bet amounts and coordinates.
    :param sb_amount: The amount of the small blind.
    :return: list of Player objects ordered by their position relative to the dealer.
    """


    if sb_amount is None:
        sb_amount = min([bet[0] for bet in bets.values()])

        # verify there is only one instance of the small blind amount
        if len([bet[0] for bet in bets.values() if bet[0] == sb_amount]) > 1:
            raise ValueError("Multiple players with the same small blind amount")

    # Identify the small blind player
    sb_player_loc = next(
        loc for loc, (amount, _) in bets.items() if amount == sb_amount
    )

    if board_bb:
        # Define the center of the table
        center_x, center_y = (board_bb[0] + board_bb[2]) // 2, (
            board_bb[1] + board_bb[3]
        ) // 2
    else:
        # get center of bounding box containing all players
        x1 = min([box[0] for _, box in players])
        y1 = min([box[1] for _, box in players])
        x2 = max([box[2] for _, box in players])
        y2 = max([box[3] for _, box in players])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Calculate angles for each player relative to the center
    def calculate_angle(box):
        x1, y1, x2, y2 = box
        player_center_x = (x1 + x2) / 2
        player_center_y = (y1 + y2) / 2
        angle = math.atan2(player_center_y - center_y, player_center_x - center_x)
        return angle

    # Get active players and calculate angles
    player_info = [(player, box, calculate_angle(box)) for player, box in players]

    # Sort players based on their angle in a clockwise manner
    player_info.sort(key=lambda item: item[2])

   
    # Rotate list to start with the small blind
    sb_index = next(
        i for i, (_, loc, _) in enumerate(player_info) if loc == sb_player_loc
    )
    ordered_players = player_info[sb_index:] + player_info[:sb_index]

    # Return only the player objects in order
    return [player for player, _, _ in ordered_players]



player_tests = [
    (
        Player(name="TexasFoldEmUsa", stack=3042, active=True),
        (np.int64(40), np.int64(357), np.int64(183), np.int64(430)),
    ),
    (
        Player(name="hiboss777", stack=41183, active=True),
        (np.int64(73), np.int64(133), np.int64(216), np.int64(206)),
    ),
    (
        Player(name="Candalzin_", stack=10, active=False),
        (np.int64(434), np.int64(63), np.int64(577), np.int64(136)),
    ),
    (
        Player(name="GenerelSchwerz", stack=9750, active=True),
        (np.int64(505), np.int64(488), np.int64(648), np.int64(561)),
    ),
    (
        Player(name="paulact28", stack=17627, active=True),
        (np.int64(863), np.int64(133), np.int64(1006), np.int64(207)),
    ),
    (
        Player(name="dumi63", stack=26959, active=True),
        (np.int64(896), np.int64(357), np.int64(1039), np.int64(431)),
    ),
]
bets = {
    (np.int64(73), np.int64(133), np.int64(216), np.int64(206)): (
        50,
        (np.int64(297), np.int64(214), np.int64(325), np.int64(234)),
    ),
    (np.int64(505), np.int64(488), np.int64(648), np.int64(561)): (
        100,
        (np.int64(494), np.int64(409), np.int64(531), np.int64(429)),
    ),
    (np.int64(863), np.int64(133), np.int64(1006), np.int64(207)): (
        100,
        (np.int64(734), np.int64(188), np.int64(771), np.int64(207)),
    ),
    (np.int64(896), np.int64(357), np.int64(1039), np.int64(431)): (
        100,
        (np.int64(760), np.int64(366), np.int64(797), np.int64(385)),
    ),
}
# Small blind amount
# small_blind_amount_test = 50

# Use the function to order players
ordered_players_test = order_players_by_sb(player_tests, bets)

# Output the ordered player names
ordered_player_names_test = [player.name for player in ordered_players_test]
print("Ordered Players:", ordered_player_names_test)

# Extract bounding boxes directly from the players' information for the test case
ordered_boxes_test = [
    next(
        (box for player, box in player_tests if player.name == ordered_player.name),
        None,
    )
    for ordered_player in ordered_players_test
]

# Create a new blank image for the test case
image_test = np.zeros((720, 1080, 3), dtype=np.uint8)

# Draw the bounding boxes on the image and label them with player names and order numbers
for index, ((x1, y1, x2, y2), ordered_player) in enumerate(
    zip(ordered_boxes_test, ordered_players_test), start=1
):
    # Draw rectangle
    cv2.rectangle(image_test, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Label the bounding box with its order number and player name
    label = f"{index}: {ordered_player.name}"
    cv2.putText(
        image_test, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

# Display the image with bounding boxes
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
