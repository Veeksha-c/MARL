# Import necessary libraries for deep learning and multi-agent systems
import torch  # PyTorch for neural networks
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import numpy as np  # Numerical computations
import random  # Random number generation
from collections import deque  # Efficient data structure for experience replay
import matplotlib.pyplot as plt  # For plotting results

# Set random seeds for reproducibility
torch.manual_seed(42)  # Set PyTorch random seed
np.random.seed(42)  # Set NumPy random seed
random.seed(42)  # Set Python random seed

# Define the DQN network for each independent agent
class DQNNetwork(nn.Module):  # Inherit from PyTorch's Module class
    def __init__(self, state_dim=6, action_dim=4, hidden_dim=64):  # Constructor with network dimensions
        super(DQNNetwork, self).__init__()  # Call parent constructor
        
        # Define the first hidden layer (input to hidden)
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 6 inputs to 64 hidden neurons
        # Define the second hidden layer (hidden to hidden)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 64 to 64 hidden neurons
        # Define the output layer (hidden to Q-values)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 64 to 4 Q-value outputs
        
        # Initialize weights using Xavier initialization
        self.init_weights()  # Call weight initialization method
        
    def init_weights(self):  # Method to initialize network weights
        # Initialize first layer weights
        nn.init.xavier_uniform_(self.fc1.weight)  # Xavier uniform initialization
        nn.init.constant_(self.fc1.bias, 0.0)  # Initialize biases to zero
        # Initialize second layer weights
        nn.init.xavier_uniform_(self.fc2.weight)  # Xavier uniform initialization
        nn.init.constant_(self.fc2.bias, 0.0)  # Initialize biases to zero
        # Initialize output layer weights
        nn.init.xavier_uniform_(self.fc3.weight)  # Xavier uniform initialization
        nn.init.constant_(self.fc3.bias, 0.0)  # Initialize biases to zero
        
    def forward(self, state):  # Forward pass through DQN network
        # Apply ReLU activation to first layer output
        x = torch.relu(self.fc1(state))  # Non-linear transformation
        # Apply ReLU activation to second layer output
        x = torch.relu(self.fc2(x))  # Another non-linear transformation
        # Return Q-values for all actions
        q_values = self.fc3(x)  # Linear output for Q-values
        return q_values  # Return Q-value estimates

# Define the Experience Replay buffer for each independent agent
class ReplayBuffer:  # Class to handle experience replay for individual agents
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

# Define the Independent Q-Learning Agent
class IQLAgent:  # Individual agent class for IQL system
    def __init__(self, agent_id, state_dim=6, action_dim=4, lr=0.001):  # Constructor with agent parameters
        self.agent_id = agent_id  # Store unique agent identifier
        self.state_dim = state_dim  # Store state dimension
        self.action_dim = action_dim  # Store action dimension
        
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device selection
        
        # Hyperparameters for the agent
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate (100% exploration)
        self.epsilon_min = 0.01  # Minimum exploration rate (1% exploration)
        self.epsilon_decay = 0.995  # Rate at which exploration decreases
        self.learning_rate = lr  # Learning rate for neural network
        self.batch_size = 32  # Number of experiences per training batch
        self.target_update_freq = 10  # How often to update target network
        
        # Initialize neural networks (main and target)
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)  # Main Q-network
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)  # Target Q-network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)  # Adam optimizer
        
        # Initialize replay buffer for this agent
        self.memory = ReplayBuffer(5000)  # Store up to 5,000 experiences
        
        # Initialize target network with same weights as main network
        self.update_target_network()  # Copy weights to target network
        
    def update_target_network(self):  # Copy weights from main to target network
        # Load state dict from main network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())  # Synchronize networks
        
    def remember(self, state, action, reward, next_state, done):  # Store experience in replay buffer
        # Add transition to agent's personal replay buffer
        self.memory.push(state, action, reward, next_state, done)  # Store experience
        
    def act(self, state):  # Choose action using epsilon-greedy policy
        # With probability epsilon, choose random action (exploration)
        if np.random.random() <= self.epsilon:  # Check if should explore
            return random.randrange(self.action_dim)  # Return random action
        # Otherwise choose best action according to Q-network (exploitation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Convert state to tensor
        q_values = self.q_network(state_tensor)  # Get Q-values from network
        return q_values.argmax().item()  # Return action with highest Q-value
        
    def replay(self):  # Train the neural network using experience replay
        # Only train if we have enough experiences in buffer
        if len(self.memory) < self.batch_size:  # Check if buffer has enough samples
            return  # Skip training if insufficient data
            
        # Sample random batch from agent's replay buffer
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
            
    def save_model(self, filepath):  # Save agent's model weights
        torch.save({  # Save dictionary with model state and hyperparameters
            'q_network_state_dict': self.q_network.state_dict(),  # Main network weights
            'target_network_state_dict': self.target_network.state_dict(),  # Target network weights
            'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state
            'epsilon': self.epsilon,  # Current exploration rate
        }, filepath)  # Save to specified file
        
    def load_model(self, filepath):  # Load agent's model weights
        checkpoint = torch.load(filepath)  # Load saved checkpoint
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])  # Load main network
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])  # Load target network
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
        self.epsilon = checkpoint['epsilon']  # Load exploration rate

# Define the Smart Grid Environment Simulator
class SmartGridEnvironment:  # Environment class for simulating smart grid dynamics
    def __init__(self, num_agents=5):  # Constructor with number of agents
        self.num_agents = num_agents  # Store number of agents
        self.agent_names = ['Solar', 'Wind', 'Battery', 'Grid', 'Load']  # Agent names
        self.state_dim = 6  # State dimension per agent
        self.action_dim = 4  # Action dimension per agent
        self.current_step = 0  # Current time step
        self.max_steps = 1000  # Maximum steps per episode
        
    def reset(self):  # Reset environment to initial state
        self.current_step = 0  # Reset step counter
        # Generate random initial states for all agents
        self.states = []  # List to store agent states
        for i in range(self.num_agents):  # Loop through all agents
            # Generate random state: [soc, generation, demand, price, time, weather]
            state = np.random.rand(self.state_dim)  # Random state values [0,1]
            self.states.append(state)  # Add state to list
        return self.states  # Return initial states
        
    def step(self, actions):  # Execute one step in the environment
        self.current_step += 1  # Increment step counter
        
        # Generate next states based on current states and actions
        next_states = []  # List to store next states
        rewards = []  # List to store rewards
        dones = []  # List to store done flags
        
        for i in range(self.num_agents):  # Loop through all agents
            current_state = self.states[i]  # Get current state for agent i
            action = actions[i]  # Get action for agent i
            
            # Simulate state transition (dummy random dynamics)
            next_state = current_state + np.random.randn(self.state_dim) * 0.1  # Add noise to state
            next_state = np.clip(next_state, 0, 1)  # Clip to valid range [0,1]
            
            # Calculate reward based on action and state (dummy reward function)
            if self.agent_names[i] == 'Solar':  # Solar agent reward
                reward = next_state[1] * (1 - action * 0.1)  # Reward based on generation
            elif self.agent_names[i] == 'Wind':  # Wind agent reward
                reward = next_state[1] * (1 - action * 0.15)  # Reward based on generation
            elif self.agent_names[i] == 'Battery':  # Battery agent reward
                reward = abs(next_state[0] - 0.5) * (1 - action * 0.05)  # Reward based on SOC
            elif self.agent_names[i] == 'Grid':  # Grid agent reward
                reward = next_state[3] * (1 - action * 0.2)  # Reward based on price
            else:  # Load agent reward
                reward = next_state[2] * (1 - action * 0.1)  # Reward based on demand
                
            # Check if episode is done
            done = self.current_step >= self.max_steps  # Episode ends after max steps
            
            next_states.append(next_state)  # Add next state to list
            rewards.append(reward)  # Add reward to list
            dones.append(done)  # Add done flag to list
            
        self.states = next_states  # Update current states
        return next_states, rewards, dones  # Return next states, rewards, dones

# Define the IQL System Manager
class IQLSmartGridSystem:  # Main class managing all independent agents
    def __init__(self, num_agents=5, state_dim=6, action_dim=4):  # Constructor with system parameters
        self.num_agents = num_agents  # Store number of agents
        self.state_dim = state_dim  # Store state dimension
        self.action_dim = action_dim  # Store action dimension
        self.agent_names = ['Solar', 'Wind', 'Battery', 'Grid', 'Load']  # Agent names
        
        # Create independent agents
        self.agents = []  # List to store agents
        for i in range(num_agents):  # Loop through all agents
            agent = IQLAgent(i, state_dim, action_dim)  # Create agent i
            self.agents.append(agent)  # Add agent to list
            
        # Create environment
        self.env = SmartGridEnvironment(num_agents)  # Initialize environment
        
        # Training metrics
        self.episode_rewards = []  # List to store episode rewards
        self.agent_rewards = [[] for _ in range(num_agents)]  # Rewards per agent
        
    def train(self, num_episodes=1000):  # Main training loop
        print(f"Starting IQL training for {num_episodes} episodes...")  # Print training start
        
        for episode in range(num_episodes):  # Loop through all episodes
            states = self.env.reset()  # Reset environment
            episode_reward = 0  # Initialize episode reward
            agent_episode_rewards = [0] * self.num_agents  # Initialize agent rewards
            
            while True:  # Loop until episode ends
                # Get actions from all agents
                actions = []  # List to store actions
                for i, agent in enumerate(self.agents):  # Loop through all agents
                    action = agent.act(states[i])  # Get action from agent i
                    actions.append(action)  # Add action to list
                    
                # Execute actions in environment
                next_states, rewards, dones = self.env.step(actions)  # Step environment
                
                # Store experiences and train each agent independently
                for i, agent in enumerate(self.agents):  # Loop through all agents
                    agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])  # Store experience
                    agent.replay()  # Train agent independently
                    
                # Update metrics
                episode_reward += sum(rewards)  # Add total reward
                for i in range(self.num_agents):  # Loop through agents
                    agent_episode_rewards[i] += rewards[i]  # Add agent reward
                    
                states = next_states  # Update states
                
                # Check if episode is done
                if all(dones):  # If all agents are done
                    break  # Exit episode loop
                    
            # Store episode metrics
            self.episode_rewards.append(episode_reward)  # Store total episode reward
            for i in range(self.num_agents):  # Loop through agents
                self.agent_rewards[i].append(agent_episode_rewards[i])  # Store agent reward
                
            # Update target networks
            if episode % 10 == 0:  # Every 10 episodes
                for agent in self.agents:  # Loop through agents
                    agent.update_target_network()  # Update target network
                    
            # Print progress
            if episode % 50 == 0:  # Every 50 episodes
                avg_reward = np.mean(self.episode_rewards[-50:])  # Calculate average reward
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")  # Print progress
                
        print("Training completed!")  # Print completion message
        
    def evaluate(self, num_episodes=10):  # Evaluate trained agents
        print(f"Evaluating agents for {num_episodes} episodes...")  # Print evaluation start
        
        eval_rewards = []  # List to store evaluation rewards
        
        for episode in range(num_episodes):  # Loop through evaluation episodes
            states = self.env.reset()  # Reset environment
            episode_reward = 0  # Initialize episode reward
            
            while True:  # Loop until episode ends
                # Get actions from all agents (no exploration during evaluation)
                actions = []  # List to store actions
                for i, agent in enumerate(self.agents):  # Loop through all agents
                    # Set epsilon to 0 for greedy action selection
                    old_epsilon = agent.epsilon  # Store current epsilon
                    agent.epsilon = 0.0  # Set epsilon to 0
                    action = agent.act(states[i])  # Get greedy action
                    agent.epsilon = old_epsilon  # Restore epsilon
                    actions.append(action)  # Add action to list
                    
                # Execute actions in environment
                next_states, rewards, dones = self.env.step(actions)  # Step environment
                
                episode_reward += sum(rewards)  # Add total reward
                states = next_states  # Update states
                
                # Check if episode is done
                if all(dones):  # If all agents are done
                    break  # Exit episode loop
                    
            eval_rewards.append(episode_reward)  # Store evaluation reward
            
        avg_eval_reward = np.mean(eval_rewards)  # Calculate average evaluation reward
        print(f"Average evaluation reward: {avg_eval_reward:.2f}")  # Print evaluation result
        return eval_rewards  # Return evaluation rewards
        
    def plot_results(self):  # Plot training results
        plt.figure(figsize=(15, 10))  # Create figure for plotting
        
        # Plot total episode rewards
        plt.subplot(2, 3, 1)  # Create first subplot
        plt.plot(self.episode_rewards)  # Plot total rewards
        plt.title('Total Episode Rewards')  # Add title
        plt.xlabel('Episode')  # Add x-axis label
        plt.ylabel('Reward')  # Add y-axis label
        
        # Plot individual agent rewards
        for i in range(self.num_agents):  # Loop through agents
            plt.subplot(2, 3, i+2)  # Create subplot for agent i
            plt.plot(self.agent_rewards[i])  # Plot agent rewards
            plt.title(f'{self.agent_names[i]} Agent Rewards')  # Add title
            plt.xlabel('Episode')  # Add x-axis label
            plt.ylabel('Reward')  # Add y-axis label
            
        plt.tight_layout()  # Adjust subplot spacing
        plt.show()  # Display the plot
        
    def save_models(self, filepath_prefix):  # Save all agent models
        for i, agent in enumerate(self.agents):  # Loop through agents
            agent.save_model(f"{filepath_prefix}_agent_{i}.pth")  # Save agent model
        print(f"All models saved with prefix: {filepath_prefix}")  # Print save message
        
    def load_models(self, filepath_prefix):  # Load all agent models
        for i, agent in enumerate(self.agents):  # Loop through agents
            agent.load_model(f"{filepath_prefix}_agent_{i}.pth")  # Load agent model
        print(f"All models loaded with prefix: {filepath_prefix}")  # Print load message

# Main execution block
if __name__ == "__main__":  # Run when script is executed directly
    # Create IQL Smart Grid System
    iql_system = IQLSmartGridSystem(num_agents=5, state_dim=6, action_dim=4)  # Initialize system
    
    # Train the system
    iql_system.train(num_episodes=500)  # Train for 500 episodes
    
    # Evaluate the trained system
    eval_rewards = iql_system.evaluate(num_episodes=20)  # Evaluate for 20 episodes
    
    # Plot training results
    iql_system.plot_results()  # Display training plots
    
    # Save trained models
    iql_system.save_models("iql_models")  # Save models
    
    print("\nIQL Smart Grid System training completed!")  # Print completion message
