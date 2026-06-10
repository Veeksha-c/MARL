import torch  # Import PyTorch library for neural networks and tensors
import torch.nn as nn  # Import neural network module from PyTorch
import torch.nn.functional as F  # Import functional operations for neural networks
import numpy as np  # Import NumPy for numerical operations
import random  # Import random module for generating random numbers
from collections import deque  # Import deque for experience replay buffer
import matplotlib.pyplot as plt  # Import matplotlib for plotting results


# Define the individual DQN network for each agent
class DQNNetwork(nn.Module):  # Define DQN network class inheriting from nn.Module
    def __init__(self, state_dim=6, action_dim=4, hidden_dim=64):  # Initialize network with dimensions
        super(DQNNetwork, self).__init__()  # Call parent constructor
        # Define first fully connected layer: state_dim -> hidden_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # First linear transformation
        # Define second fully connected layer: hidden_dim -> hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second linear transformation
        # Define third fully connected layer: hidden_dim -> hidden_dim
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Third linear transformation
        # Define output layer: hidden_dim -> action_dim (Q-values for each action)
        self.fc4 = nn.Linear(hidden_dim, action_dim)  # Output layer for Q-values

    def forward(self, state):  # Define forward pass through network
        # Apply ReLU activation to first layer output
        x = F.relu(self.fc1(state))  # First activation function
        # Apply ReLU activation to second layer output
        x = F.relu(self.fc2(x))  # Second activation function
        # Apply ReLU activation to third layer output
        x = F.relu(self.fc3(x))  # Third activation function
        # Return Q-values for all actions (no activation on output)
        return self.fc4(x)  # Final Q-values output


# Define the QMIX mixing network with non-negative weights
class QMIXMixingNetwork(nn.Module):  # Define mixing network class
    def __init__(self, num_agents=5, state_dim=30, mixing_embed_dim=32, hypernet_embed=64):  # Initialize mixing network
        super(QMIXMixingNetwork, self).__init__()  # Call parent constructor
        self.num_agents = num_agents  # Store number of agents
        self.state_dim = state_dim  # Store global state dimension
        self.mixing_embed_dim = mixing_embed_dim  # Store mixing embedding dimension
        
        # Hypernetwork to generate weights for first mixing layer
        self.hyper_w1 = nn.Sequential(  # Define hypernetwork for first layer weights
            nn.Linear(state_dim, hypernet_embed),  # First linear layer of hypernetwork
            nn.ReLU(),  # ReLU activation
            nn.Linear(hypernet_embed, mixing_embed_dim * num_agents)  # Output layer for weights
        )
        
        # Hypernetwork to generate bias for first mixing layer
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)  # Hypernetwork for first layer bias
        
        # Hypernetwork to generate weights for second mixing layer
        self.hyper_w2 = nn.Sequential(  # Define hypernetwork for second layer weights
            nn.Linear(state_dim, hypernet_embed),  # First linear layer of hypernetwork
            nn.ReLU(),  # ReLU activation
            nn.Linear(hypernet_embed, mixing_embed_dim)  # Output layer for weights
        )
        
        # Hypernetwork to generate bias for second mixing layer
        self.hyper_b2 = nn.Sequential(  # Define hypernetwork for second layer bias
            nn.Linear(state_dim, mixing_embed_dim),  # First linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(mixing_embed_dim, 1)  # Output scalar bias
        )

    def forward(self, agent_qs, global_state):  # Define forward pass
        # agent_qs: tensor of shape (batch_size, num_agents) - Q-values from each agent
        # global_state: tensor of shape (batch_size, state_dim) - global state information
        
        # Generate weights for first mixing layer using hypernetwork
        w1 = torch.abs(self.hyper_w1(global_state))  # Apply absolute value to ensure non-negative weights
        # Reshape weights to (batch_size, mixing_embed_dim, num_agents)
        w1 = w1.view(-1, self.mixing_embed_dim, self.num_agents)  # Reshape for matrix multiplication
        
        # Generate bias for first mixing layer
        b1 = self.hyper_b1(global_state)  # Generate bias from global state
        # Reshape bias to (batch_size, mixing_embed_dim, 1)
        b1 = b1.view(-1, 1, self.mixing_embed_dim)  # Reshape for broadcasting
        
        # Generate weights for second mixing layer using hypernetwork
        w2 = torch.abs(self.hyper_w2(global_state))  # Apply absolute value to ensure non-negative weights
        # Reshape weights to (batch_size, 1, mixing_embed_dim)
        w2 = w2.view(-1, 1, self.mixing_embed_dim)  # Reshape for matrix multiplication
        
        # Generate bias for second mixing layer
        b2 = self.hyper_b2(global_state)  # Generate final bias from global state
        # Reshape bias to (batch_size, 1, 1)
        b2 = b2.view(-1, 1, 1)  # Reshape for broadcasting
        
        # Reshape agent Q-values to (batch_size, num_agents, 1)
        agent_qs = agent_qs.view(-1, self.num_agents, 1)  # Prepare for matrix multiplication
        
        # First mixing layer: combine agent Q-values
        hidden = F.elu(torch.bmm(w1, agent_qs) + b1)  # Matrix multiplication plus bias, with ELU activation
        
        # Second mixing layer: produce final total Q-value
        q_tot = torch.bmm(w2, hidden) + b2  # Final matrix multiplication plus bias
        
        # Squeeze to remove extra dimension and return total Q-value
        return q_tot.squeeze(-1)  # Return shape (batch_size,)


# Define the QMIX agent class that manages the entire system
class QMIXAgent:  # Define main QMIX agent class
    def __init__(self, num_agents=5, state_dim=6, action_dim=4, global_state_dim=30,  # Initialize QMIX agent
                 lr=0.001, gamma=0.99, buffer_size=10000, batch_size=32):  # Set hyperparameters
        self.num_agents = num_agents  # Store number of agents
        self.state_dim = state_dim  # Store individual agent state dimension
        self.action_dim = action_dim  # Store action dimension
        self.global_state_dim = global_state_dim  # Store global state dimension
        self.gamma = gamma  # Store discount factor
        self.batch_size = batch_size  # Store batch size
        
        # Create agent names for the smart grid
        self.agent_names = ['Solar', 'Wind', 'Battery', 'Grid', 'Load']  # Names of 5 agents
        
        # Initialize DQN networks for each agent (both main and target networks)
        self.agent_networks = []  # List to store main DQN networks
        self.target_networks = []  # List to store target DQN networks
        for i in range(num_agents):  # Loop through all agents
            # Create main DQN network for agent i
            agent_net = DQNNetwork(state_dim, action_dim)  # Initialize main network
            # Create target DQN network for agent i
            target_net = DQNNetwork(state_dim, action_dim)  # Initialize target network
            # Copy weights from main to target network
            target_net.load_state_dict(agent_net.state_dict())  # Initialize target with main weights
            # Add networks to respective lists
            self.agent_networks.append(agent_net)  # Add main network
            self.target_networks.append(target_net)  # Add target network
        
        # Initialize QMIX mixing networks (both main and target)
        self.mixing_network = QMIXMixingNetwork(num_agents, global_state_dim)  # Main mixing network
        self.target_mixing_network = QMIXMixingNetwork(num_agents, global_state_dim)  # Target mixing network
        # Copy weights from main to target mixing network
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())  # Initialize target mixing
        
        # Initialize optimizers for all networks
        self.agent_optimizers = []  # List to store optimizers for agent networks
        for agent_net in self.agent_networks:  # Loop through agent networks
            # Create Adam optimizer for each agent network
            optimizer = torch.optim.Adam(agent_net.parameters(), lr=lr)  # Initialize optimizer
            self.agent_optimizers.append(optimizer)  # Add optimizer to list
        
        # Create optimizer for mixing network
        self.mixer_optimizer = torch.optim.Adam(self.mixing_network.parameters(), lr=lr)  # Mixing network optimizer
        
        # Initialize experience replay buffer
        self.buffer = deque(maxlen=buffer_size)  # Create replay buffer with fixed size
        
        # Set networks to training mode
        for agent_net in self.agent_networks:  # Loop through agent networks
            agent_net.train()  # Set to training mode
        self.mixing_network.train()  # Set mixing network to training mode

    def select_action(self, states, epsilon=0.0):  # Select actions for all agents using epsilon-greedy
        actions = []  # List to store selected actions
        for i, state in enumerate(states):  # Loop through each agent's state
            # Convert state to PyTorch tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            
            # With probability epsilon, select random action (exploration)
            if random.random() < epsilon:  # Check if random action should be taken
                action = random.randint(0, self.action_dim - 1)  # Select random action
            else:  # Otherwise, select greedy action (exploitation)
                with torch.no_grad():  # Disable gradient computation
                    q_values = self.agent_networks[i](state_tensor)  # Get Q-values from network
                    action = q_values.argmax().item()  # Select action with highest Q-value
            
            actions.append(action)  # Add selected action to list
        
        return actions  # Return list of actions for all agents

    def store_experience(self, states, actions, rewards, next_states, global_state, next_global_state, done):  # Store transition in replay buffer
        # Create experience tuple
        experience = (states, actions, rewards, next_states, global_state, next_global_state, done)
        # Add experience to replay buffer
        self.buffer.append(experience)  # Store experience

    def sample_experience(self):  # Sample batch of experiences from replay buffer
        # Check if buffer has enough experiences
        if len(self.buffer) < self.batch_size:  # If not enough experiences
            return None  # Return None
        
        # Sample random batch of experiences
        experiences = random.sample(self.buffer, self.batch_size)  # Random sampling
        
        # Unpack experiences into separate lists
        states, actions, rewards, next_states, global_states, next_global_states, dones = zip(*experiences)
        
        # Convert lists to numpy arrays for easier manipulation
        states = np.array(states)  # Convert states to array
        actions = np.array(actions)  # Convert actions to array
        rewards = np.array(rewards)  # Convert rewards to array
        next_states = np.array(next_states)  # Convert next states to array
        global_states = np.array(global_states)  # Convert global states to array
        next_global_states = np.array(next_global_states)  # Convert next global states to array
        dones = np.array(dones)  # Convert dones to array
        
        # Return batch of experiences
        return states, actions, rewards, next_states, global_states, next_global_states, dones

    def train_step(self):  # Perform one training step
        # Sample batch of experiences
        batch = self.sample_experience()  # Get training batch
        if batch is None:  # If no batch available
            return 0.0  # Return zero loss
        
        # Unpack batch
        states, actions, rewards, next_states, global_states, next_global_states, dones = batch
        
        # Convert numpy arrays to PyTorch tensors
        states = torch.FloatTensor(states)  # Convert states to tensor
        actions = torch.LongTensor(actions)  # Convert actions to tensor
        rewards = torch.FloatTensor(rewards)  # Convert rewards to tensor
        next_states = torch.FloatTensor(next_states)  # Convert next states to tensor
        global_states = torch.FloatTensor(global_states)  # Convert global states to tensor
        next_global_states = torch.FloatTensor(next_global_states)  # Convert next global states to tensor
        dones = torch.FloatTensor(dones)  # Convert dones to tensor
        
        # Calculate current Q-values for each agent
        current_q_values = []  # List to store current Q-values
        for i in range(self.num_agents):  # Loop through agents
            # Get Q-values for agent i
            q_vals = self.agent_networks[i](states[:, i, :])  # Forward pass through agent network
            # Select Q-values for taken actions
            q_vals = q_vals.gather(1, actions[:, i].unsqueeze(1)).squeeze(1)  # Gather action Q-values
            current_q_values.append(q_vals)  # Add to list
        
        # Stack current Q-values into tensor
        current_q_values = torch.stack(current_q_values, dim=1)  # Shape: (batch_size, num_agents)
        
        # Calculate next Q-values for each agent using target networks
        next_q_values = []  # List to store next Q-values
        for i in range(self.num_agents):  # Loop through agents
            # Get next Q-values from target network
            next_q_vals = self.target_networks[i](next_states[:, i, :])  # Forward pass through target
            # Get maximum Q-value for next state
            next_q_vals = next_q_vals.max(dim=1)[0]  # Max Q-value
            next_q_values.append(next_q_vals)  # Add to list
        
        # Stack next Q-values into tensor
        next_q_values = torch.stack(next_q_values, dim=1)  # Shape: (batch_size, num_agents)
        
        # Calculate target total Q-value
        with torch.no_grad():  # Disable gradient computation for target calculation
            # Get next total Q-value from target mixing network
            next_q_tot = self.target_mixing_network(next_q_values, next_global_states)  # Forward pass through target mixing
            # Calculate target total Q-value using Bellman equation
            target_q_tot = rewards + (1.0 - dones) * self.gamma * next_q_tot  # Bellman equation
        
        # Calculate current total Q-value from mixing network
        current_q_tot = self.mixing_network(current_q_values, global_states)  # Forward pass through mixing network
        
        # Calculate loss (MSE between current and target total Q-values)
        mixer_loss = F.mse_loss(current_q_tot, target_q_tot)  # Calculate mixing network loss
        
        # Backpropagate loss for mixing network
        self.mixer_optimizer.zero_grad()  # Reset gradients
        mixer_loss.backward()  # Backpropagate loss
        self.mixer_optimizer.step()  # Update mixing network weights
        
        # Calculate individual agent losses (optional, for monitoring)
        agent_losses = []  # List to store individual agent losses
        for i in range(self.num_agents):  # Loop through agents
            # Calculate individual agent loss
            agent_loss = F.mse_loss(current_q_values[:, i], target_q_tot)  # Individual agent loss
            agent_losses.append(agent_loss.item())  # Add loss value to list
        
        # Return average loss for monitoring
        return mixer_loss.item()  # Return mixing network loss

    def update_target_networks(self):  # Update target networks with main network weights
        # Update target networks for each agent
        for i in range(self.num_agents):  # Loop through agents
            # Copy weights from main to target network
            self.target_networks[i].load_state_dict(self.agent_networks[i].state_dict())  # Update target
        
        # Update target mixing network
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())  # Update target mixing

    def save_models(self, filepath):  # Save all models to file
        # Create checkpoint dictionary
        checkpoint = {  # Dictionary to store all model states
            'agent_networks': [net.state_dict() for net in self.agent_networks],  # Agent network states
            'target_networks': [net.state_dict() for net in self.target_networks],  # Target network states
            'mixing_network': self.mixing_network.state_dict(),  # Mixing network state
            'target_mixing_network': self.target_mixing_network.state_dict(),  # Target mixing network state
        }
        # Save checkpoint to file
        torch.save(checkpoint, filepath)  # Save models

    def load_models(self, filepath):  # Load all models from file
        # Load checkpoint from file
        checkpoint = torch.load(filepath)  # Load saved models
        
        # Load agent network states
        for i, net in enumerate(self.agent_networks):  # Loop through agent networks
            net.load_state_dict(checkpoint['agent_networks'][i])  # Load agent network state
        
        # Load target network states
        for i, net in enumerate(self.target_networks):  # Loop through target networks
            net.load_state_dict(checkpoint['target_networks'][i])  # Load target network state
        
        # Load mixing network states
        self.mixing_network.load_state_dict(checkpoint['mixing_network'])  # Load mixing network
        self.target_mixing_network.load_state_dict(checkpoint['target_mixing_network'])  # Load target mixing


# Define dummy smart grid environment for testing
class DummySmartGridEnvironment:  # Define dummy environment class
    def __init__(self, num_agents=5, state_dim=6, action_dim=4, max_steps=100):  # Initialize environment
        self.num_agents = num_agents  # Store number of agents
        self.state_dim = state_dim  # Store state dimension
        self.action_dim = action_dim  # Store action dimension
        self.max_steps = max_steps  # Store maximum episode length
        self.current_step = 0  # Initialize current step counter
        
        # Initialize random states for all agents
        self.states = np.random.randn(num_agents, state_dim)  # Random initial states
        # Initialize random global state (concatenation of all agent states)
        self.global_state = self.states.flatten()  # Flatten agent states for global state
        # Pad global state to desired dimension if needed
        if len(self.global_state) < 30:  # If global state is too small
            padding = np.random.randn(30 - len(self.global_state))  # Generate random padding
            self.global_state = np.concatenate([self.global_state, padding])  # Add padding

    def reset(self):  # Reset environment to initial state
        self.current_step = 0  # Reset step counter
        # Generate new random states
        self.states = np.random.randn(self.num_agents, self.state_dim)  # New random states
        # Update global state
        self.global_state = self.states.flatten()  # Flatten states
        # Pad global state if needed
        if len(self.global_state) < 30:  # If padding needed
            padding = np.random.randn(30 - len(self.global_state))  # Generate padding
            self.global_state = np.concatenate([self.global_state, padding])  # Add padding
        return self.states.copy(), self.global_state.copy()  # Return initial states

    def step(self, actions):  # Take step in environment
        # Generate random rewards based on actions
        rewards = np.random.randn(self.num_agents)  # Random rewards
        # Add some action-dependent component to rewards
        for i, action in enumerate(actions):  # Loop through actions
            rewards[i] += 0.1 * action  # Add action influence to reward
        
        # Generate next states (random walk)
        next_states = self.states + 0.1 * np.random.randn(self.num_agents, self.state_dim)  # Random walk
        # Clip states to reasonable range
        next_states = np.clip(next_states, -2, 2)  # Clip states
        
        # Update global state
        next_global_state = next_states.flatten()  # Flatten next states
        # Pad global state if needed
        if len(next_global_state) < 30:  # If padding needed
            padding = np.random.randn(30 - len(next_global_state))  # Generate padding
            next_global_state = np.concatenate([next_global_state, padding])  # Add padding
        
        # Check if episode is done
        self.current_step += 1  # Increment step counter
        done = self.current_step >= self.max_steps  # Check if max steps reached
        
        # Update current states
        self.states = next_states  # Update to next states
        self.global_state = next_global_state  # Update global state
        
        return next_states.copy(), rewards, done, next_global_state.copy()  # Return step results


# Define training function
def train_qmix(num_episodes=1000, max_steps_per_episode=100, update_target_freq=100,  # Define training parameters
               epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):  # Epsilon parameters
    # Initialize QMIX agent
    agent = QMIXAgent(num_agents=5, state_dim=6, action_dim=4, global_state_dim=30)  # Create agent
    
    # Initialize environment
    env = DummySmartGridEnvironment(num_agents=5, state_dim=6, action_dim=4, max_steps=max_steps_per_episode)  # Create environment
    
    # Training metrics
    episode_rewards = []  # List to store episode rewards
    losses = []  # List to store training losses
    
    # Training loop
    epsilon = epsilon_start  # Initialize epsilon for exploration
    for episode in range(num_episodes):  # Loop through episodes
        # Reset environment
        states, global_state = env.reset()  # Get initial states
        episode_reward = 0  # Initialize episode reward
        episode_loss = 0  # Initialize episode loss
        num_updates = 0  # Initialize update counter
        
        # Episode loop
        for step in range(max_steps_per_episode):  # Loop through steps
            # Select actions using epsilon-greedy policy
            actions = agent.select_action(states, epsilon)  # Get actions from agent
            
            # Take step in environment
            next_states, rewards, done, next_global_state = env.step(actions)  # Environment step
            
            # Store experience in replay buffer
            agent.store_experience(states, actions, rewards, next_states, global_state, next_global_state, done)  # Store transition
            
            # Update states
            states = next_states  # Update to next states
            global_state = next_global_state  # Update global state
            
            # Accumulate episode reward
            episode_reward += sum(rewards)  # Add step rewards
            
            # Train agent if buffer has enough experiences
            if len(agent.buffer) > agent.batch_size:  # Check if training possible
                loss = agent.train_step()  # Perform training step
                episode_loss += loss  # Accumulate loss
                num_updates += 1  # Increment update counter
            
            # Check if episode is done
            if done:  # If episode finished
                break  # Exit episode loop
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)  # Update epsilon
        
        # Calculate average loss for episode
        avg_loss = episode_loss / max(1, num_updates)  # Average loss
        losses.append(avg_loss)  # Store average loss
        
        # Store episode reward
        episode_rewards.append(episode_reward)  # Store total episode reward
        
        # Update target networks periodically
        if (episode + 1) % update_target_freq == 0:  # Check if it's time to update
            agent.update_target_networks()  # Update target networks
        
        # Print progress
        if (episode + 1) % 100 == 0:  # Print every 100 episodes
            print(f"Episode {episode + 1}/{num_episodes}, "  # Print episode number
                  f"Reward: {episode_reward:.2f}, "  # Print episode reward
                  f"Loss: {avg_loss:.4f}, "  # Print average loss
                  f"Epsilon: {epsilon:.3f}")  # Print current epsilon
    
    return agent, episode_rewards, losses  # Return trained agent and metrics


# Define testing function
def test_qmix(agent, num_episodes=10, max_steps_per_episode=100):  # Define testing parameters
    # Initialize environment
    env = DummySmartGridEnvironment(num_agents=5, state_dim=6, action_dim=4, max_steps=max_steps_per_episode)  # Create environment
    
    # Testing metrics
    episode_rewards = []  # List to store test episode rewards
    
    # Testing loop
    for episode in range(num_episodes):  # Loop through test episodes
        # Reset environment
        states, global_state = env.reset()  # Get initial states
        episode_reward = 0  # Initialize episode reward
        
        # Episode loop (no exploration during testing)
        for step in range(max_steps_per_episode):  # Loop through steps
            # Select greedy actions (no exploration)
            actions = agent.select_action(states, epsilon=0.0)  # Get greedy actions
            
            # Take step in environment
            next_states, rewards, done, next_global_state = env.step(actions)  # Environment step
            
            # Update states
            states = next_states  # Update to next states
            global_state = next_global_state  # Update global state
            
            # Accumulate episode reward
            episode_reward += sum(rewards)  # Add step rewards
            
            # Check if episode is done
            if done:  # If episode finished
                break  # Exit episode loop
        
        # Store episode reward
        episode_rewards.append(episode_reward)  # Store test episode reward
        
        # Print test progress
        print(f"Test Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")  # Print test result
    
    return episode_rewards  # Return test rewards


# Define plotting function
def plot_results(episode_rewards, losses):  # Define plotting function
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Create subplots
    
    # Plot episode rewards
    ax1.plot(episode_rewards)  # Plot rewards over episodes
    ax1.set_title('Episode Rewards')  # Set title
    ax1.set_xlabel('Episode')  # Set x-axis label
    ax1.set_ylabel('Total Reward')  # Set y-axis label
    ax1.grid(True)  # Add grid
    
    # Plot training losses
    ax2.plot(losses)  # Plot losses over episodes
    ax2.set_title('Training Loss')  # Set title
    ax2.set_xlabel('Episode')  # Set x-axis label
    ax2.set_ylabel('Loss')  # Set y-axis label
    ax2.grid(True)  # Add grid
    
    # Adjust layout and show plot
    plt.tight_layout()  # Adjust subplot layout
    plt.show()  # Display plot


# Main execution block
if __name__ == "__main__":  # Check if script is run directly
    print("Starting QMIX training for Smart Grid environment...")  # Print start message
    
    # Train QMIX agent
    trained_agent, train_rewards, train_losses = train_qmix(num_episodes=500)  # Train agent
    
    print("\nTraining completed. Starting testing...")  # Print completion message
    
    # Test trained agent
    test_rewards = test_qmix(trained_agent, num_episodes=10)  # Test agent
    
    print(f"\nAverage test reward: {np.mean(test_rewards):.2f}")  # Print average test reward
    
    # Plot results
    plot_results(train_rewards, train_losses)  # Plot training metrics
    
    print("QMIX Smart Grid simulation completed!")  # Print final message
