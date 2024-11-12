from dataclasses import dataclass
from enum import Enum
import random
from typing import List, Dict, Optional, Set, Tuple


class Color(Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    WHITE = "white"


@dataclass
class Card:
    color: Color
    number: int

    def __str__(self):
        return f"{self.color.value.capitalize()} {self.number}"


class HintType(Enum):
    COLOR = "color"
    NUMBER = "number"


class Hanabi:
    def __init__(self, num_players: int):
        if not 2 <= num_players <= 5:
            raise ValueError("Player count must be between 2 and 5")

        self.num_players = num_players
        self.players = []
        self.hands_size = 4 if num_players >= 4 else 5
        self.hint_tokens = 8
        self.fuse_tokens = 3
        self.current_player = 0
        self.deck = self._create_deck()
        self.discard_pile = []
        self.fireworks = {color: [] for color in Color}
        self.game_over = False
        self.final_round = False
        self.rounds_remaining = num_players

        # Deal initial hands
        self._deal_initial_hands()

    def _create_deck(self) -> List[Card]:
        deck = []
        # For each color, create cards: 1(x3), 2(x2), 3(x2), 4(x2), 5(x1)
        for color in Color:
            for number, count in [(1, 3), (2, 2), (3, 2), (4, 2), (5, 1)]:
                deck.extend([Card(color, number) for _ in range(count)])
        random.shuffle(deck)
        return deck

    def _deal_initial_hands(self):
        for _ in range(self.num_players):
            hand = [self.deck.pop() for _ in range(self.hands_size)]
            self.players.append(hand)

    # self.game.is_hint_valid(self.game.current_player, target, hint_type, value)
    def is_hint_valid(
        self, player: int, target: int, hint_type: HintType, value: any
    ) -> bool:
        """Check if a hint is valid for a player's cards."""
        if player != self.current_player:
            return False
        if target == player:
            return False
        if not 0 <= target < len(self.players):
            return False
        if hint_type == HintType.COLOR and value not in Color:
            return False
        if hint_type == HintType.NUMBER and not 1 <= value <= 5:
            return False
        return True

    def is_play_valid(self, player: int, card_index: int) -> bool:
        """Check if a play is valid for a card in hand."""
        if player != self.current_player:
            return False
        if not 0 <= card_index < len(self.players[player]):
            return False
        return True
    
    def is_discard_valid(self, player: int, card_index: int) -> bool:
        """Check if a discard is valid for a card in hand."""
        if player != self.current_player:
            return False
        if self.hint_tokens >= 8:
            return False
        if not 0 <= card_index < len(self.players[player]):
            return False
        return True

    def give_hint(
        self, from_player: int, to_player: int, hint_type: HintType, value: any
    ) -> List[int]:
        """Give a hint to another player about their cards. Returns indices of matching cards."""
        if self.hint_tokens <= 0:
            raise ValueError("No hint tokens remaining")
        if from_player == to_player:
            raise ValueError("Cannot give hint to yourself")

        matching_indices = []
        for idx, card in enumerate(self.players[to_player]):
            if (hint_type == HintType.COLOR and card.color == value) or (
                hint_type == HintType.NUMBER and card.number == value
            ):
                matching_indices.append(idx)

        if not matching_indices:
            raise ValueError("Hint must match at least one card")

        self.hint_tokens -= 1
        return matching_indices

    def play_card(self, player: int, card_index: int) -> Tuple[bool, str]:
        """Play a card from hand. Returns (success, message)."""
        if player != self.current_player:
            raise ValueError("Not your turn")

        hand = self.players[player]
        if not 0 <= card_index < len(hand):
            raise ValueError("Invalid card index")

        card = hand.pop(card_index)
        firework = self.fireworks[card.color]

        # Check if card can be played
        if not firework and card.number == 1:
            firework.append(card)
            success = True
            msg = f"Successfully played {card}"
        elif firework and card.number == firework[-1].number + 1:
            firework.append(card)
            success = True
            msg = f"Successfully played {card}"
            # Add hint token if completed a firework (played a 5)
            if card.number == 5 and self.hint_tokens < 8:
                self.hint_tokens += 1
                msg += ". Gained a hint token for completing firework!"
        else:
            self.discard_pile.append(card)
            self.fuse_tokens -= 1
            success = False
            msg = f"Invalid play: {card}. Lost a fuse token!"

        # Draw replacement card if possible
        if self.deck:
            hand.append(self.deck.pop())
        elif not self.final_round:
            self.final_round = True
            self.rounds_remaining = self.num_players

        self._advance_turn()
        return success, msg

    def discard_card(self, player: int, card_index: int) -> str:
        """Discard a card from hand. Returns result message."""
        if player != self.current_player:
            raise ValueError("Not your turn")
        if self.hint_tokens >= 8:
            raise ValueError("Cannot discard when hint tokens are full")

        hand = self.players[player]
        if not 0 <= card_index < len(hand):
            raise ValueError("Invalid card index")

        card = hand.pop(card_index)
        self.discard_pile.append(card)
        self.hint_tokens += 1

        # Draw replacement card if possible
        if self.deck:
            hand.append(self.deck.pop())
        elif not self.final_round:
            self.final_round = True
            self.rounds_remaining = self.num_players

        msg = f"Discarded {card}. Gained a hint token!"
        self._advance_turn()
        return msg

    def _advance_turn(self):
        """Advance to next player and check game end conditions."""
        if self.final_round:
            self.rounds_remaining -= 1
            if self.rounds_remaining <= 0:
                self.game_over = True
                return

        if self.fuse_tokens <= 0:
            self.game_over = True
            return

        # Check if all fireworks are complete
        if all(len(stack) == 5 for stack in self.fireworks.values()):
            self.game_over = True
            return

        self.current_player = (self.current_player + 1) % self.num_players

    def get_score(self) -> int:
        """Calculate current score (sum of highest card in each firework)."""
        return sum(len(stack) for stack in self.fireworks.values())

    def get_game_state(self) -> str:
        """Return a string representation of the current game state."""
        state = []
        state.append(f"Hint tokens: {self.hint_tokens}")
        state.append(f"Fuse tokens: {self.fuse_tokens}")
        state.append(f"Cards in deck: {len(self.deck)}")
        state.append("\nFireworks:")
        for color in Color:
            stack = self.fireworks[color]
            top_card = stack[-1].number if stack else 0
            state.append(f"{color.value.capitalize()}: {top_card}")
        state.append(f"\nCurrent Score: {self.get_score()}")
        return "\n".join(state)


def print_hand(hand: list, hide_cards: bool = False) -> None:
    """Print a player's hand, optionally hiding card values."""
    print("\nCards:", end=" ")
    for i, card in enumerate(hand):
        if hide_cards:
            print(f"{i}: [??]", end=" ")
        else:
            print(f"{i}: [{card}]", end=" ")
    print()


def print_game_info(game: Hanabi) -> None:
    """Print current game state information."""
    print("\n" + "=" * 50)
    print(game.get_game_state())
    print("\nDiscard pile:")
    for card in game.discard_pile:
        print(f"[{card}]", end=" ")
    print("\n" + "=" * 50)


def get_valid_action() -> str:
    """Get a valid action choice from the user."""
    while True:
        print("\nChoose action:")
        print("1: Give hint")
        print("2: Play card")
        print("3: Discard card")
        choice = input("Enter choice (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("Invalid choice. Please try again.")


def main():
    # Game setup
    while True:
        try:
            num_players = int(input("Enter number of players (2-5): "))
            game = Hanabi(num_players)
            break
        except ValueError as e:
            print(f"Error: {e}")

    # Main game loop
    while not game.game_over:
        current_player = game.current_player
        print(f"\nPlayer {current_player}'s turn")

        # Show game state
        print_game_info(game)

        # Show other players' hands
        for i in range(num_players):
            if i != current_player:
                print(f"\nPlayer {i}'s hand:")
                print_hand(game.players[i], hide_cards=False)

        # Show current player's hand (hidden)
        print(f"\nYour hand (Player {current_player}):")
        print_hand(game.players[current_player], hide_cards=True)

        # Get action choice
        action = get_valid_action()

        try:
            if action == "1":  # Give hint
                if game.hint_tokens <= 0:
                    print("No hint tokens remaining!")
                    continue

                # Get hint details
                to_player = int(
                    input(
                        f"Which player to hint (0-{num_players-1}, except {current_player}): "
                    )
                )
                hint_type = (
                    input("Hint about color (c) or number (n)? ").lower().strip()
                )

                if hint_type == "c":
                    color_input = input(
                        "Enter color (red/blue/green/yellow/white): "
                    ).upper()
                    try:
                        color = Color[color_input]
                        matches = game.give_hint(
                            current_player, to_player, HintType.COLOR, color
                        )
                        print(f"Revealed {len(matches)} cards of color {color.value}")
                    except (KeyError, ValueError) as e:
                        print(f"Error: {e}")
                        continue

                elif hint_type == "n":
                    try:
                        number = int(input("Enter number (1-5): "))
                        matches = game.give_hint(
                            current_player, to_player, HintType.NUMBER, number
                        )
                        print(f"Revealed {len(matches)} cards of number {number}")
                    except ValueError as e:
                        print(f"Error: {e}")
                        continue

            elif action == "2":  # Play card
                card_index = int(input("Which card to play (0-4)? "))
                success, msg = game.play_card(current_player, card_index)
                print(msg)

            elif action == "3":  # Discard card
                if game.hint_tokens >= 8:
                    print("Cannot discard when hint tokens are full!")
                    continue
                card_index = int(input("Which card to discard (0-4)? "))
                msg = game.discard_card(current_player, card_index)
                print(msg)

        except ValueError as e:
            print(f"Error: {e}")
            continue

    # Game over
    print("\nGame Over!")
    print(f"Final score: {game.get_score()} out of 25")

    if game.get_score() == 25:
        print("Perfect score! Congratulations!")
    elif game.fuse_tokens <= 0:
        print("Lost due to too many mistakes!")
    else:
        print("Game completed!")
        if game.get_score() >= 20:
            print("Legendary performance!")
        elif game.get_score() >= 15:
            print("Good job!")
        else:
            print("Better luck next time!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
