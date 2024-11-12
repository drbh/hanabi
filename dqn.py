import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple
from hanabi import Hanabi, HintType, Color


class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQNNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DQN Hyperparameters
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 128
        self.update_target_frequency = 10

        # Networks
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)
        self.steps = 0

    def get_state(self, game: Hanabi) -> np.ndarray:
        state = []

        # Game state information (3 values)
        state.extend([game.hint_tokens / 8, game.fuse_tokens / 3, len(game.deck) / 50])

        # Fireworks state (5 values)
        for color in Color:
            stack = game.fireworks[color]
            top_card = len(stack) / 5 if stack else 0
            state.append(top_card)

        # Encode current player's hand (5 cards * 10 features = 50 values)
        hand = game.players[game.current_player]
        for _ in range(5):  # Always encode 5 card slots
            # Encode card position as unknown (10 zeros)
            state.extend([0] * 10)  # 5 for number + 5 for color

        # Other players' hands (10 features per card)
        for i in range(game.num_players):
            if i != game.current_player:
                hand = game.players[i]
                for _ in range(5):  # Always encode 5 card slots
                    if _ < len(hand):
                        card = hand[_]
                        # One-hot encode number (5 values)
                        number_enc = [0] * 5
                        number_enc[card.number - 1] = 1
                        # One-hot encode color (5 values)
                        color_enc = [0] * 5
                        color_enc[list(Color).index(card.color)] = 1
                        state.extend(number_enc + color_enc)
                    else:
                        # Empty slot
                        state.extend([0] * 10)

        return np.array(state, dtype=np.float32)

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train(self, batch: Tuple):
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]

        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()


def calculate_state_size(num_players: int) -> int:
    # Game state (3) + Fireworks (5) + Current player hand (5 * 10)
    base_size = 3 + 5 + 50
    # Other players' hands (num_players - 1) * 5 cards * 10 features
    other_players_size = (num_players - 1) * 5 * 10
    return base_size + other_players_size


def train_dqn(episodes=5000):
    num_players = 2
    state_size = calculate_state_size(num_players)  # Dynamic calculation
    action_size = 10  # 5 play actions + 5 discard actions

    agent = DQNAgent(state_size, action_size)
    scores = []
    losses = []

    print(f"Starting training with state size: {state_size}")
    print(f"Action space size: {action_size}")

    for episode in range(episodes):
        env = Hanabi(num_players)
        state = agent.get_state(env)
        score = 0
        episode_losses = []

        while not env.game_over:
            action = agent.act(state)

            # Convert action index to game action
            card_index = action % 5
            reward = 0

            try:
                if action < 5:  # Play card
                    success, _ = env.play_card(env.current_player, card_index)
                    reward = 1 if success else -1
                else:  # Discard card
                    if env.hint_tokens < 8:
                        env.discard_card(env.current_player, card_index)
                        reward = 0.1  # Small positive reward for valid discard
                    else:
                        reward = -0.5  # Penalty for invalid discard
            except ValueError as e:
                reward = -0.5

            next_state = agent.get_state(env)
            agent.memory.push(state, action, reward, next_state, env.game_over)

            if len(agent.memory) > agent.batch_size:
                batch = agent.memory.sample(agent.batch_size)
                loss = agent.train(batch)
                episode_losses.append(loss)

            state = next_state
            score += reward
            agent.steps += 1

            # Update target network
            if agent.steps % agent.update_target_frequency == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        scores.append(score)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)

        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(
                f"Episode: {episode}, Average Score: {avg_score:.2f}, "
                f"Average Loss: {avg_loss:.2f}, Epsilon: {agent.epsilon:.2f}"
            )

    return agent, scores, losses


if __name__ == "__main__":
    trained_agent, training_scores, training_losses = train_dqn()

    # Save the trained model
    torch.save(trained_agent.policy_net.state_dict(), "hanabi_dqn.pth")

    # Plot training results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_scores)
    plt.title("Training Scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.subplot(1, 2, 2)
    plt.plot(training_losses)
    plt.title("Training Losses")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()
