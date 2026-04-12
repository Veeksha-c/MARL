# Import necessary libraries for deep learning and environment
import torch  # PyTorch for neural networks
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import numpy as np  # Numerical computations
import random  # Random number generation
import gymnasium as gym  # Environment for reinforcement learning
from collections import deque  # Efficient data structure for experience replay
import matplotlib.pyplot as plt  # For plotting results

# Set random seeds for reproducibility
torch.manual_seed(42)  # Set PyTorch random seed
np.random.seed(42)  # Set NumPy random seed
random.seed(42)  # Set Python random seed

# Define the neural network architecture for Q-value approximation
class QNetwork(nn.Module):  # Inherit from PyTorch's Module class
    def __init__(self, state_size, action_size):  # Constructor with input dimensions
        super(QNetwork, self).__init__()  # Call parent constructor
        # Define first fully connected layer (input to hidden)
        self.fc1 = nn.Linear(state_size, 64)  # 64 neurons in first hidden layer
        # Define second fully connected layer (hidden to hidden)
        self.fc2 = nn.Linear(64, 64)  # 64 neurons in second hidden layer
        # Define output layer (hidden to Q-values)
        self.fc3 = nn.Linear(64, action_size)  # Output Q-values for each action
        
    def forward(self, state):  # Forward pass through network
        # Apply ReLU activation to first layer output
        x = torch.relu(self.fc1(state))  # Non-linear transformation
        # Apply ReLU activation to second layer output
        x = torch.relu(self.fc2(x))  # Another non-linear transformation
        # Return raw Q-values (no activation on output layer)
        return self.fc3(x)  # Linear output for Q-values

# Define the Experience Replay buffer for storing and sampling transitions
class ReplayBuffer:  # Class to handle experience replay
    def __init__(self, capacity):  # Constructor with buffer size
        self.buffer = deque(maxlen=capacity)  # Use deque for efficient operations
        
    def push(self, state, action, reward, next_state, done):  # Store a transition
        # Add experience tuple to buffer
        self.buffer.append((state, action, reward, next_state, done))  # Store complete transition
        
    def sample(self, batch_size):  # Sample random batch of experiences
        # Randomly sample batch_size experiences from buffer
        batch = random.sample(self.buffer, batch_size)  # Random sampling
        # Unpack batch into separate arrays for each component
        states, actions, rewards, next_states, dones = zip(*batch)  # Decompress batch
        return states, actions, rewards, next_states, dones  # Return unpacked components
        
    def __len__(self):  # Get current buffer size
        return len(self.buffer)  # Return number of stored experiences

# Define the DQN Agent class that contains all learning logic
class DQNAgent:  # Main agent class
    def __init__(self, state_size, action_size):  # Constructor with environment dimensions
        self.state_size = state_size  # Dimension of state space
        self.action_size = action_size  # Number of possible actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        
        # Hyperparameters for the agent
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate (100% exploration)
        self.epsilon_min = 0.01  # Minimum exploration rate (1% exploration)
        self.epsilon_decay = 0.995  # Rate at which exploration decreases
        self.learning_rate = 0.001  # Learning rate for neural network
        self.batch_size = 64  # Number of experiences per training batch
        self.target_update_freq = 10  # How often to update target network
        
        # Initialize neural networks
        self.q_network = QNetwork(state_size, action_size).to(self.device)  # Main Q-network
        self.target_network = QNetwork(state_size, action_size).to(self.device)  # Target Q-network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)  # Adam optimizer
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(10000)  # Store up to 10,000 experiences
        
        # Initialize target network with same weights as main network
        self.update_target_network()  # Copy weights to target network
        
    def update_target_network(self):  # Copy weights from main to target network
        # Load state dict from main network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())  # Synchronize networks
        
    def remember(self, state, action, reward, next_state, done):  # Store experience in replay buffer
        # Convert numpy arrays to PyTorch tensors and store
        self.memory.push(state, action, reward, next_state, done)  # Add transition to buffer
        
    def act(self, state):  # Choose action using epsilon-greedy policy
        # With probability epsilon, choose random action (exploration)
        if np.random.random() <= self.epsilon:  # Check if should explore
            return random.randrange(self.action_size)  # Return random action
        # Otherwise choose best action according to Q-network (exploitation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Convert state to tensor
        q_values = self.q_network(state_tensor)  # Get Q-values from network
        return q_values.argmax().item()  # Return action with highest Q-value
        
    def replay(self):  # Train the neural network using experience replay
        # Only train if we have enough experiences in buffer
        if len(self.memory) < self.batch_size:  # Check if buffer has enough samples
            return  # Skip training if insufficient data
            
        # Sample random batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)  # Get batch
        
        # Convert all arrays to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)  # Convert states to tensor
        actions = torch.LongTensor(actions).to(self.device)  # Convert actions to tensor
        rewards = torch.FloatTensor(rewards).to(self.device)  # Convert rewards to tensor
        next_states = torch.FloatTensor(next_states).to(self.device)  # Convert next states to tensor
        dones = torch.BoolTensor(dones).to(self.device)  # Convert done flags to tensor
        
        # Get current Q-values for the actions that were taken
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))  # Select Q-values for taken actions
        
        # Get next Q-values from target network for Bellman equation
        next_q_values = self.target_network(next_states).max(1)[0].detach()  # Best Q-values for next states
        # Calculate target Q-values using Bellman equation
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)  # Q-learning target
        
        # Calculate loss between predicted and target Q-values
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)  # Mean squared error loss
        
        # Perform backpropagation to update network weights
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Calculate gradients
        self.optimizer.step()  # Update weights
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:  # Check if above minimum
            self.epsilon *= self.epsilon_decay  # Decay epsilon
            
    def train(self, env, episodes):  # Main training loop
        scores = []  # List to store episode scores
        for episode in range(episodes):  # Loop through all episodes
            state, _ = env.reset()  # Reset environment to starting state
            total_reward = 0  # Initialize episode reward
            done = False  # Episode completion flag
            while not done:  # Loop until episode ends
                action = self.act(state)  # Choose action using policy
                next_state, reward, done, _, _ = env.step(action)  # Take action in environment
                self.remember(state, action, reward, next_state, done)  # Store experience
                state = next_state  # Update current state
                total_reward += reward  # Accumulate reward
                self.replay()  # Train neural network
                
            scores.append(total_reward)  # Store episode score
            
            # Update target network every few episodes
            if episode % self.target_update_freq == 0:  # Check if time to update
                self.update_target_network()  # Synchronize target network
                
            # Print progress
            print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {self.epsilon:.2f}")  # Show stats
            
        return scores  # Return all episode scores

# Main execution block
if __name__ == "__main__":  # Run when script is executed directly
    # Create CartPole environment
    env = gym.make('CartPole-v1')  # Initialize CartPole environment
    state_size = env.observation_space.shape[0]  # Get state dimension (4 for CartPole)
    action_size = env.action_space.n  # Get number of actions (2 for CartPole)
    
    # Create DQN agent
    agent = DQNAgent(state_size, action_size)  # Initialize agent with correct dimensions
    
    # Train the agent
    episodes = 500  # Number of training episodes
    scores = agent.train(env, episodes)  # Train agent and get scores
    
    # Plot learning progress
    plt.figure(figsize=(12, 4))  # Create figure for plotting
    plt.subplot(1, 2, 1)  # Create first subplot
    plt.plot(scores)  # Plot scores over episodes
    plt.title('Episode Scores')  # Add title
    plt.xlabel('Episode')  # Add x-axis label
    plt.ylabel('Score')  # Add y-axis label
    
    plt.subplot(1, 2, 2)  # Create second subplot
    # Plot moving average for smoother visualization
    window = 20  # Window size for moving average
    moving_avg = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]  # Calculate moving average
    plt.plot(moving_avg)  # Plot moving average
    plt.title(f'Moving Average (window={window})')  # Add title
    plt.xlabel('Episode')  # Add x-axis label
    plt.ylabel('Average Score')  # Add y-axis label
    
    plt.tight_layout()  # Adjust subplot spacing
    plt.show()  # Display the plot
    
    # Test the trained agent
    print("\nTesting trained agent...")  # Print test message
    state, _ = env.reset()  # Reset environment
    done = False  # Episode completion flag
    test_score = 0  # Initialize test score
    while not done:  # Loop until episode ends
        action = agent.act(state)  # Choose action (no exploration during testing)
        state, reward, done, _, _ = env.step(action)  # Take action
        test_score += reward  # Accumulate reward
        env.render()  # Display environment (optional)
        
    print(f"Test Score: {test_score}")  # Print final test score
    env.close()  # Close environment
