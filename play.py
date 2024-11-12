import torch
from hanabi import Hanabi, Color
from dqn import DQNNetwork, DQNAgent, calculate_state_size


def print_game_state(game: Hanabi):
    print("\n" + "=" * 50)
    print(f"Hint tokens: {game.hint_tokens}")
    print(f"Fuse tokens: {game.fuse_tokens}")
    print(f"Cards in deck: {len(game.deck)}")
    print("\nFireworks:")
    for color in Color:
        stack = game.fireworks[color]
        top_card = stack[-1].number if stack else 0
        print(f"{color.value.capitalize()}: {top_card}")
    print("\nDiscard pile:")
    for card in game.discard_pile:
        print(f"[{card}]", end=" ")
    print("\n" + "=" * 50)


def print_hands(game: Hanabi):
    for i in range(game.num_players):
        print(f"\nPlayer {i}'s hand:")
        for j, card in enumerate(game.players[i]):
            print(f"Card {j}: {card}")


def play_game():
    # Initialize game and agent
    num_players = 2
    state_size = calculate_state_size(num_players)
    action_size = 10  # 5 play + 5 discard actions

    # Create agent and load trained model
    agent = DQNAgent(state_size, action_size)
    agent.epsilon = 0  # No random actions during evaluation
    try:
        agent.policy_net.load_state_dict(
            torch.load("hanabi_dqn.pth", weights_only=True)
        )
        print("Loaded trained model successfully!")
    except FileNotFoundError:
        print("No trained model found. Please train the model first.")
        return

    # Start new game
    game = Hanabi(num_players)
    total_score = 0
    turn = 0

    while not game.game_over:
        print(f"\nTurn {turn} - Player {game.current_player}'s turn")
        print_game_state(game)
        print_hands(game)

        # Get state and action from agent
        state = agent.get_state(game)
        action = agent.act(state)
        card_index = action % 5

        try:
            if action < 5:  # Play card
                card = game.players[game.current_player][card_index]
                print(f"\nAction: Playing card {card_index} ({card})")
                success, msg = game.play_card(game.current_player, card_index)
                print(f"Result: {msg}")
                if success:
                    total_score += 1
            else:  # Discard card
                card = game.players[game.current_player][card_index]
                print(f"\nAction: Discarding card {card_index} ({card})")
                msg = game.discard_card(game.current_player, card_index)
                print(f"Result: {msg}")
        except ValueError as e:
            print(f"Invalid action: {e}")

        turn += 1
        input("\nPress Enter for next turn...")  # Pause between turns

    # Game over - print final state
    print("\nGame Over!")
    print_game_state(game)
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
    play_game()
