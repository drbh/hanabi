# hanabi

this is a tiny repo that contains a simple implementation of the card game hanabi. this can be used to play hanabi with a human player, or to train hanabots? (named by Ivar)

## how to play

```bash
uv run hanabi.py
# Enter number of players (2-5): 2

# Player 0's turn

# ==================================================
# Hint tokens: 8
# Fuse tokens: 3
# Cards in deck: 40

# Fireworks:
# Red: 0
# Blue: 0
# Green: 0
# Yellow: 0
# White: 0

# Current Score: 0

# Discard pile:

# ==================================================

# Player 1's hand:

# Cards: 0: [Green 2] 1: [Green 4] 2: [Red 4] 3: [Green 4] 4: [Red 1]

# Your hand (Player 0):

# Cards: 0: [??] 1: [??] 2: [??] 3: [??] 4: [??]

# Choose action:
# 1: Give hint
# 2: Play card
# 3: Discard card
# Enter choice (1-3): 1
# Which player to hint (0-1, except 0): 1
# Hint about color (c) or number (n)? c
# Enter color (red/blue/green/yellow/white): red
# Revealed 2 cards of color red
```

## train small dqn model

train a small model that outputs to `hanabi_dqn.pth`

```bash
uv run dqn.py
# Starting training with state size: 108
# Action space size: 10
# Episode: 9, Average Score: -3.75, Average Loss: 0.00, Epsilon: 0.95
# Episode: 19, Average Score: -4.25, Average Loss: 0.17, Epsilon: 0.90
# Episode: 29, Average Score: -2.95, Average Loss: 0.38, Epsilon: 0.86
# ...
```

## step through a game with the trained model

then you can run it (although it's not very good yet)

```bash
uv run play.py
# Loaded trained model successfully!

# Turn 0 - Player 0's turn

# ==================================================
# Hint tokens: 8
# Fuse tokens: 3
# Cards in deck: 40

# Fireworks:
# Red: 0
# Blue: 0
# Green: 0
# Yellow: 0
# White: 0

# Discard pile:

# ==================================================

# Player 0's hand:
# Card 0: Blue 2
# Card 1: Green 4
# Card 2: Yellow 1
# Card 3: Red 1
# Card 4: Yellow 2

# Player 1's hand:
# Card 0: White 5
# Card 1: Blue 1
# Card 2: Red 3
# Card 3: Red 4
# Card 4: Red 1

# Action: Playing card 3 (Red 1)
# Result: Successfully played Red 1

# Press Enter for next turn...
```
