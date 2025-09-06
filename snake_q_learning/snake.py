import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import time # For debug prints if needed
import math

# --- PyTorch Device Setup ---
# Check for MPS (Apple Silicon GPU) first, then CUDA, then CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# --- Constants ---
# Screen and Panel Dimensions
SCREEN_WIDTH = 1280  # Increased width for better analytics display
SCREEN_HEIGHT = 720
GAME_PANEL_WIDTH = 700 # Width for the game display
INFO_PANEL_WIDTH = SCREEN_WIDTH - GAME_PANEL_WIDTH # Width for analytics
BORDER_THICKNESS = 5

# Game Grid and Block Size
BLOCK_SIZE = 20
GRID_WIDTH = (GAME_PANEL_WIDTH - 4 * BLOCK_SIZE) // BLOCK_SIZE # Grid cells horizontally
GRID_HEIGHT = (SCREEN_HEIGHT - 4 * BLOCK_SIZE) // BLOCK_SIZE # Grid cells vertically
GAME_AREA_X_OFFSET = 2 * BLOCK_SIZE
GAME_AREA_Y_OFFSET = 2 * BLOCK_SIZE


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN_LIGHT = (0, 255, 0)
GREEN_DARK = (0, 200, 0)
BLUE_LIGHT = (100, 100, 255)
BLUE_DARK = (0, 0, 200)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Snake Game Parameters
INITIAL_SNAKE_LENGTH = 3
FPS_AGENT_PLAYING = 60 # Speed of game visualization

# DQN Agent Parameters
STATE_SIZE = 11  # [danger_straight, danger_right, danger_left, dir_l, dir_r, dir_u, dir_d, food_l, food_r, food_u, food_d]
ACTION_SIZE = 3  # [straight, turn_right, turn_left]
LEARNING_RATE = 0.001
GAMMA = 0.9         # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995 # Multiplicative decay: new_epsilon = old_epsilon * decay
# More aggressive decay for faster learning, might need tuning:
# EPSILON_DECAY_FRAMES = 100000
# EPSILON_DECAY_LINEAR_RATE = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_FRAMES


REPLAY_MEMORY_SIZE = 50000
BATCH_SIZE = 1024 # Larger batch size can stabilize training
TARGET_UPDATE_FREQUENCY = 10  # Update target network every N episodes

# --- Named Tuple for Points ---
Point = namedtuple('Point', ['x', 'y'])
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# --- Helper Functions ---
def draw_text(surface, text, position, font, color=WHITE, antialias=True):
    """Renders and draws text on a surface."""
    text_surface = font.render(text, antialias, color)
    surface.blit(text_surface, position)

def draw_plot(surface, data, plot_rect, color=GREEN_LIGHT, max_val=None, title="Plot"):
    """Draws a simple line plot for analytics."""
    pygame.draw.rect(surface, GRAY, plot_rect, 1) # Border for plot area
    if not data or len(data) < 2:
        return

    if max_val is None:
        max_y = max(data) if data else 1
    else:
        max_y = max_val
    max_y = max(1, max_y) # Avoid division by zero

    points = []
    for i, val in enumerate(data):
        x = plot_rect.left + (i / (len(data) -1)) * plot_rect.width
        y = plot_rect.bottom - (val / max_y) * plot_rect.height
        y = max(plot_rect.top, min(plot_rect.bottom, y)) # Clamp y to plot area
        points.append((x, y))

    if len(points) >= 2:
        pygame.draw.lines(surface, color, False, points, 2)

    # Draw title for the plot
    plot_font = pygame.font.Font(None, 24)
    draw_text(surface, title, (plot_rect.left + 5, plot_rect.top + 5), plot_font, WHITE)


# --- SnakeGame Class ---
class SnakeGame:
    def __init__(self, width_cells, height_cells):
        self.width_cells = width_cells
        self.height_cells = height_cells
        self.font = pygame.font.Font(None, 30) # For displaying score in game panel
        self.reset()

    def reset(self):
        """Resets the game state for a new episode."""
        self.direction = Point(1, 0)  # Start moving right

        # Center the snake initially
        start_x = self.width_cells // 2
        start_y = self.height_cells // 2
        self.head = Point(start_x, start_y)
        self.snake_body = [self.head,
                           Point(self.head.x - 1, self.head.y),
                           Point(self.head.x - 2, self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.game_over = False
        self.steps_since_last_food = 0 # To potentially penalize for not eating

    def _place_food(self):
        """Places food randomly on the grid, not on the snake."""
        while True:
            x = random.randint(0, self.width_cells - 1)
            y = random.randint(0, self.height_cells - 1)
            self.food = Point(x, y)
            if self.food not in self.snake_body:
                break

    def step(self, action_idx): # action_idx is [0: straight, 1: right turn, 2: left turn]
        """
        Performs one game step based on the agent's action.
        Action: 0 (straight), 1 (turn right), 2 (turn left) relative to current direction.
        Returns: (reward, game_over, score_increment)
        """
        self.steps_since_last_food +=1
        reward = 0 # Default reward per step
        score_increment = 0

        # Determine new direction based on action
        # Directions: UP, RIGHT, DOWN, LEFT (clockwise order)
        # Current direction vector: self.direction
        # Relative turns:
        # - Straight: new_direction = current_direction
        # - Right turn: if current is (dx, dy), new is (dy, -dx)
        # - Left turn: if current is (dx, dy), new is (-dy, dx)

        current_dx, current_dy = self.direction.x, self.direction.y
        if action_idx == 0: # Straight
            new_direction = self.direction
        elif action_idx == 1: # Turn Right
            new_direction = Point(current_dy, -current_dx)
        elif action_idx == 2: # Turn Left
            new_direction = Point(-current_dy, current_dx)
        else:
            raise ValueError(f"Invalid action_idx: {action_idx}")
        self.direction = new_direction

        # Update snake head position
        self.head = Point(self.head.x + self.direction.x, self.head.y + self.direction.y)
        self.snake_body.insert(0, self.head)

        # Check for game over conditions
        if self._is_collision():
            self.game_over = True
            reward = -10 # Penalty for collision
            self.snake_body.pop() # Remove the head that caused collision for drawing
            return reward, self.game_over, score_increment

        # Check for food eaten
        if self.head == self.food:
            self.score += 1
            score_increment = 1
            reward = 10  # Reward for eating food
            self._place_food()
            self.steps_since_last_food = 0
        else:
            self.snake_body.pop() # Remove tail if no food eaten
            # Optional: Small penalty for each step to encourage efficiency
            # reward = -0.01
            # Optional: Penalty for taking too long to find food
            # if self.steps_since_last_food > (self.width_cells * self.height_cells) * 0.5: # Heuristic
            #     reward = -1 # Small penalty for wandering
            #     self.game_over = True # End episode if wandering too much
            #     return reward, self.game_over, score_increment


        # Optional: Reward for getting closer to food
        # dist_to_food_before = math.hypot(self.snake_body[1].x - self.food.x, self.snake_body[1].y - self.food.y)
        # dist_to_food_after = math.hypot(self.head.x - self.food.x, self.head.y - self.food.y)
        # if dist_to_food_after < dist_to_food_before:
        #     reward += 0.1
        # elif dist_to_food_after > dist_to_food_before:
        #     reward -= 0.2


        return reward, self.game_over, score_increment

    def _is_collision(self, point=None):
        """Checks if a given point (or snake head) causes a collision."""
        if point is None:
            point = self.head

        # Wall collision
        if not (0 <= point.x < self.width_cells and 0 <= point.y < self.height_cells):
            return True
        # Self-collision
        if point in self.snake_body[1:]:
            return True
        return False

    def draw(self, surface, block_size, x_offset, y_offset):
        """Draws the game state on the given surface."""
        surface.fill(BLACK) # Background for game area

        # Draw Grid (optional, can be performance intensive if redrawn every frame)
        # for x in range(self.width_cells):
        #     for y in range(self.height_cells):
        #         rect = pygame.Rect(x_offset + x * block_size, y_offset + y * block_size, block_size, block_size)
        #         pygame.draw.rect(surface, GRAY, rect, 1)


        # Draw snake
        for i, segment in enumerate(self.snake_body):
            rect = pygame.Rect(x_offset + segment.x * block_size,
                               y_offset + segment.y * block_size,
                               block_size, block_size)
            if i == 0: # Head
                pygame.draw.rect(surface, GREEN_DARK, rect)
                pygame.draw.rect(surface, GREEN_LIGHT, rect, 2) # Border for head
            else: # Body
                pygame.draw.rect(surface, GREEN_LIGHT, rect)
                pygame.draw.rect(surface, GREEN_DARK, rect, 1) # Border for body segments

        # Draw food
        if self.food:
            food_rect = pygame.Rect(x_offset + self.food.x * block_size,
                                    y_offset + self.food.y * block_size,
                                    block_size, block_size)
            pygame.draw.rect(surface, RED, food_rect)
            pygame.draw.ellipse(surface, ORANGE, food_rect.inflate(-block_size//4, -block_size//4))


        # Draw score in game panel
        score_text = f"Score: {self.score}"
        draw_text(surface, score_text, (x_offset, y_offset - block_size), self.font, WHITE)


# --- QNetwork Class (PyTorch nn.Module) ---
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- DQNAgent Class ---
class DQNAgent:
    def __init__(self, state_size, action_size, lr, gamma, eps_start, eps_end, eps_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.epsilon_end = eps_end
        self.epsilon_decay = eps_decay # For multiplicative decay
        # self.epsilon_decay_linear_rate = eps_decay_linear_rate # For linear decay

        self.policy_net = QNetwork(state_size, action_size).to(DEVICE)
        self.target_net = QNetwork(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.n_games = 0
        self.last_loss = 0.0 # For analytics

    def get_state(self, game):
        """
        Converts the game state into a binary vector for the DQN.
        State: [danger_straight, danger_right, danger_left,
                dir_l, dir_r, dir_u, dir_d,
                food_l, food_r, food_u, food_d]
        """
        head = game.head
        current_direction = game.direction # This is Point(dx, dy)

        # Directions relative to snake's current orientation
        # Straight is current_direction
        # Right turn: (current_direction.y, -current_direction.x)
        # Left turn: (-current_direction.y, current_direction.x)

        dir_straight_vec = current_direction
        dir_right_vec = Point(current_direction.y, -current_direction.x)
        dir_left_vec = Point(-current_direction.y, current_direction.x)

        # Points to check for danger
        pt_straight = Point(head.x + dir_straight_vec.x, head.y + dir_straight_vec.y)
        pt_right = Point(head.x + dir_right_vec.x, head.y + dir_right_vec.y)
        pt_left = Point(head.x + dir_left_vec.x, head.y + dir_left_vec.y)

        state = [
            # Danger ahead (straight, right, left)
            game._is_collision(pt_straight),
            game._is_collision(pt_right),
            game._is_collision(pt_left),

            # Current absolute direction of snake
            current_direction.x == -1,  # Moving Left
            current_direction.x == 1,   # Moving Right
            current_direction.y == -1,  # Moving Up
            current_direction.y == 1,   # Moving Down

            # Food location relative to head
            game.food.x < head.x,  # Food Left
            game.food.x > head.x,  # Food Right
            game.food.y < head.y,  # Food Up
            game.food.y > head.y   # Food Down
        ]
        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the replay memory."""
        self.memory.append(Experience(state, action, reward, next_state, done))

    def get_action(self, state, is_training=True):
        """
        Chooses an action using epsilon-greedy policy.
        Returns action index: 0 (straight), 1 (right turn), 2 (left turn).
        """
        if is_training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore: random action
        else:
            with torch.no_grad():
                # CORRECTED LINE: Changed torch.float3 to torch.float32
                state_tensor = torch.tensor(state, dtype=torch.float32).to(DEVICE).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item() # Exploit: best action from Q-network

    def replay(self, batch_size):
        """Trains the policy network using a batch of experiences from memory."""
        if len(self.memory) < batch_size:
            return # Not enough experiences to sample a batch

        mini_batch = random.sample(self.memory, batch_size)
        
        # Convert batch of Experiences to separate Tensors
        states = torch.tensor(np.array([e.state for e in mini_batch]), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor([e.action for e in mini_batch], dtype=torch.long).to(DEVICE).unsqueeze(1) # Add dimension for gather
        rewards = torch.tensor([e.reward for e in mini_batch], dtype=torch.float32).to(DEVICE).unsqueeze(1)
        next_states = torch.tensor(np.array([e.next_state for e in mini_batch]), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor([e.done for e in mini_batch], dtype=torch.bool).to(DEVICE).unsqueeze(1)

        # Get Q-values for current states from policy_net
        # Q(s,a)
        current_q_values = self.policy_net(states).gather(1, actions)

        # Get max Q-values for next states from target_net
        # max_a' Q_target(s', a')
        with torch.no_grad(): # No gradient calculation for target network part
            next_q_values_target = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # Compute target Q-values: R + gamma * max_a' Q_target(s', a') if not done, else R
        target_q_values = rewards + (self.gamma * next_q_values_target * (~dones))
        
        # Compute loss (MSE)
        loss = F.mse_loss(current_q_values, target_q_values)
        self.last_loss = loss.item() # Store loss for analytics

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional, but can help stabilize)
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()


    def update_target_network(self):
        """Copies weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Target network updated at game {self.n_games}")

    def decay_epsilon(self):
        """Decays epsilon for exploration-exploitation balance."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay # Multiplicative decay
            # self.epsilon -= self.epsilon_decay_linear_rate # Linear decay
            self.epsilon = max(self.epsilon_end, self.epsilon)


    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.n_games = checkpoint['n_games']
            self.memory = checkpoint.get('memory', deque(maxlen=REPLAY_MEMORY_SIZE)) # Load memory if saved
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"No model found at {path}, starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")


    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'n_games': self.n_games,
            # 'memory': self.memory # Saving memory can make files very large
        }, path)
        print(f"Model saved to {path} at game {self.n_games}")


# --- Main Training Loop ---
def main():
    pygame.init()
    pygame.font.init() # Explicitly initialize font module

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("DQN Snake Agent")
    clock = pygame.time.Clock()

    # Create separate surfaces for game and info panels
    game_surface = screen.subsurface(pygame.Rect(0, 0, GAME_PANEL_WIDTH, SCREEN_HEIGHT))
    info_surface = screen.subsurface(pygame.Rect(GAME_PANEL_WIDTH, 0, INFO_PANEL_WIDTH, SCREEN_HEIGHT))

    # Fonts for analytics
    title_font = pygame.font.Font(None, 48)
    stats_font = pygame.font.Font(None, 28)
    plot_title_font = pygame.font.Font(None, 24)


    game = SnakeGame(GRID_WIDTH, GRID_HEIGHT)
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, GAMMA,
                     EPSILON_START, EPSILON_END, EPSILON_DECAY)

    # Attempt to load a pre-trained model
    # agent.load_model("snake_dqn_model.pth")

    all_scores = []
    avg_scores = [] # Moving average scores
    losses = [] # Store losses for plotting

    max_episodes = 2000 # Number of games to play for training
    running = True
    print(f"Starting training for {max_episodes} episodes...")
    print(f"Game grid: {GRID_WIDTH}x{GRID_HEIGHT} cells. Block size: {BLOCK_SIZE}px.")
    print(f"Game panel: {GAME_PANEL_WIDTH}x{SCREEN_HEIGHT}px. Info panel: {INFO_PANEL_WIDTH}x{SCREEN_HEIGHT}px.")
    print(f"Game area offset: ({GAME_AREA_X_OFFSET}, {GAME_AREA_Y_OFFSET})")


    for episode in range(1, max_episodes + 1):
        if not running:
            break

        game.reset()
        current_state = agent.get_state(game)
        done = False
        episode_score = 0
        episode_steps = 0

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_s: # Manual save
                        agent.save_model("snake_dqn_model_manual.pth")


            action = agent.get_action(current_state) # Agent chooses action [0,1,2]
            reward, done, score_increment = game.step(action)
            next_state = agent.get_state(game)

            agent.remember(current_state, action, reward, next_state, done)
            
            # Only start replay once memory has enough samples
            if len(agent.memory) > BATCH_SIZE:
                 agent.replay(BATCH_SIZE)
                 if agent.last_loss is not None: # Ensure loss is not None before appending
                    losses.append(agent.last_loss) # Store loss after replay

            current_state = next_state
            episode_score += score_increment
            episode_steps += 1
            
            # --- Drawing ---
            # Draw Game on Game Panel
            game_surface.fill(BLACK) # Clear game surface
            pygame.draw.rect(game_surface, GRAY, (0,0, GAME_PANEL_WIDTH, SCREEN_HEIGHT), BORDER_THICKNESS) # Border for game panel
            game.draw(game_surface, BLOCK_SIZE, GAME_AREA_X_OFFSET, GAME_AREA_Y_OFFSET)


            # Draw Analytics on Info Panel
            info_surface.fill(BLUE_DARK) # Clear info surface
            pygame.draw.rect(info_surface, GRAY, (0,0, INFO_PANEL_WIDTH, SCREEN_HEIGHT), BORDER_THICKNESS) # Border for info panel

            y_offset_info = 20
            draw_text(info_surface, "DQN Training Analytics", (20, y_offset_info), title_font, YELLOW)
            y_offset_info += 60
            draw_text(info_surface, f"Episode: {episode}/{max_episodes}", (20, y_offset_info), stats_font, WHITE)
            y_offset_info += 30
            draw_text(info_surface, f"Agent Games: {agent.n_games}", (20, y_offset_info), stats_font, WHITE)
            y_offset_info += 30
            draw_text(info_surface, f"Epsilon: {agent.epsilon:.4f}", (20, y_offset_info), stats_font, WHITE)
            y_offset_info += 30
            draw_text(info_surface, f"Current Score: {episode_score}", (20, y_offset_info), stats_font, WHITE)
            y_offset_info += 30
            high_score = max(all_scores) if all_scores else 0
            draw_text(info_surface, f"High Score: {high_score}", (20, y_offset_info), stats_font, WHITE)
            y_offset_info += 30
            avg_score_val = avg_scores[-1] if avg_scores else 0
            draw_text(info_surface, f"Avg Score (last 100): {avg_score_val:.2f}", (20, y_offset_info), stats_font, WHITE)
            y_offset_info += 30
            draw_text(info_surface, f"Steps this episode: {episode_steps}", (20, y_offset_info), stats_font, WHITE)
            y_offset_info += 30
            draw_text(info_surface, f"Replay Memory: {len(agent.memory)}/{REPLAY_MEMORY_SIZE}", (20, y_offset_info), stats_font, WHITE)
            y_offset_info += 30
            draw_text(info_surface, f"Last Loss: {agent.last_loss:.4f}", (20, y_offset_info), stats_font, WHITE)
            y_offset_info += 40


            # Plot Scores
            plot_height = 150
            score_plot_rect = pygame.Rect(20, y_offset_info, INFO_PANEL_WIDTH - 40, plot_height)
            draw_plot(info_surface, all_scores, score_plot_rect, GREEN_LIGHT, title="Scores per Episode")
            if avg_scores:
                 draw_plot(info_surface, avg_scores, score_plot_rect, YELLOW, title="") # Overlay avg scores

            y_offset_info += plot_height + 10

            # Plot Losses (optional) - Uncomment to display
            # loss_plot_rect = pygame.Rect(20, y_offset_info, INFO_PANEL_WIDTH - 40, plot_height)
            # if losses: # Check if losses list is not empty
            #    # Filter out None values from losses if any were appended before replay started
            #    valid_losses = [l for l in losses if l is not None]
            #    if valid_losses:
            #        draw_plot(info_surface, valid_losses, loss_plot_rect, RED, title="Training Loss (MSE)", max_val=max(valid_losses) if valid_losses else 1)


            pygame.display.flip() # Update the full screen
            clock.tick(FPS_AGENT_PLAYING)


        # End of episode
        agent.n_games += 1
        agent.decay_epsilon()
        all_scores.append(episode_score)

        # Calculate moving average score (e.g., over last 100 episodes)
        avg_score = np.mean(all_scores[-100:])
        avg_scores.append(avg_score)

        if episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()

        print(f"Episode {episode}: Score={episode_score}, Avg Score={avg_score:.2f}, Epsilon={agent.epsilon:.4f}, Steps={episode_steps}, Loss={agent.last_loss:.4f}")

        if episode % 100 == 0: # Save model periodically
            agent.save_model(f"snake_dqn_model_ep{episode}.pth")

    # End of training
    agent.save_model("snake_dqn_model_final.pth")
    print("Training finished.")
    pygame.quit()

if __name__ == '__main__':
    main()

