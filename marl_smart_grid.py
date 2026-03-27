# Import necessary libraries for deep learning and multi-agent systems
import torch  # PyTorch for neural networks
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Activation functions and utilities
import numpy as np  # Numerical computations
import torch.optim as optim  # Optimization algorithms

# Set random seeds for reproducibility
torch.manual_seed(42)  # Set PyTorch random seed
np.random.seed(42)  # Set NumPy random seed

# Define the Actor network for policy approximation in MARL
class ActorNetwork(nn.Module):  # Inherit from PyTorch's Module class
    def __init__(self, state_dim=6, action_dim=4, hidden_dim=128):  # Constructor with network dimensions
        super(ActorNetwork, self).__init__()  # Call parent constructor
        
        # Define the first hidden layer (input to hidden)
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 6 inputs to 128 hidden neurons
        # Define the second hidden layer (hidden to hidden)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 128 to 128 hidden neurons
        # Define the third hidden layer (hidden to hidden)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)  # 128 to 64 hidden neurons
        # Define the output layer (hidden to action logits)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)  # 64 to 4 action outputs
        
        # Initialize weights using Xavier initialization for better training
        self.init_weights()  # Call weight initialization method
        
    def init_weights(self):  # Method to initialize network weights
        # Initialize first layer weights
        nn.init.xavier_uniform_(self.fc1.weight)  # Xavier uniform initialization
        nn.init.constant_(self.fc1.bias, 0.0)  # Initialize biases to zero
        # Initialize second layer weights
        nn.init.xavier_uniform_(self.fc2.weight)  # Xavier uniform initialization
        nn.init.constant_(self.fc2.bias, 0.0)  # Initialize biases to zero
        # Initialize third layer weights
        nn.init.xavier_uniform_(self.fc3.weight)  # Xavier uniform initialization
        nn.init.constant_(self.fc3.bias, 0.0)  # Initialize biases to zero
        # Initialize output layer weights with smaller values
        nn.init.xavier_uniform_(self.fc4.weight, gain=0.01)  # Small gain for policy network
        nn.init.constant_(self.fc4.bias, 0.0)  # Initialize biases to zero
        
    def forward(self, state):  # Forward pass through actor network
        # Apply ReLU activation to first layer output
        x = F.relu(self.fc1(state))  # Non-linear transformation
        # Apply ReLU activation to second layer output
        x = F.relu(self.fc2(x))  # Another non-linear transformation
        # Apply ReLU activation to third layer output
        x = F.relu(self.fc3(x))  # Final hidden layer transformation
        # Return action logits (raw scores before softmax)
        action_logits = self.fc4(x)  # Linear output for action selection
        return action_logits  # Return unnormalized action probabilities
        
    def get_action(self, state, exploration_noise=0.1):  # Method to select actions with exploration
        # Convert state to tensor if it's not already
        if isinstance(state, np.ndarray):  # Check if input is numpy array
            state = torch.FloatTensor(state)  # Convert to PyTorch tensor
            
        # Get action logits from forward pass
        with torch.no_grad():  # Disable gradient computation for action selection
            action_logits = self.forward(state)  # Get raw action scores
            
        # Apply softmax to get probability distribution
        action_probs = F.softmax(action_logits, dim=-1)  # Convert logits to probabilities
        
        # Add exploration noise to encourage exploration
        if exploration_noise > 0:  # Check if noise should be added
            # Add Gaussian noise to logits
            noise = torch.randn_like(action_logits) * exploration_noise  # Generate noise
            action_logits = action_logits + noise  # Add noise to logits
            action_probs = F.softmax(action_logits, dim=-1)  # Recompute probabilities
            
        # Sample action from probability distribution
        action = torch.multinomial(action_probs, num_samples=1)  # Sample action
        return action.item()  # Return action as integer
        
    def get_action_prob(self, state):  # Method to get action probabilities
        # Get action logits from forward pass
        action_logits = self.forward(state)  # Get raw action scores
        # Apply softmax to get probability distribution
        action_probs = F.softmax(action_logits, dim=-1)  # Convert to probabilities
        return action_probs  # Return action probability distribution

# Define the Critic network for value function approximation in MARL
class CriticNetwork(nn.Module):  # Inherit from PyTorch's Module class
    def __init__(self, state_dim=6, action_dim=4, num_agents=5, hidden_dim=128):  # Constructor with MARL parameters
        super(CriticNetwork, self).__init__()  # Call parent constructor
        
        self.num_agents = num_agents  # Store number of agents
        self.state_dim = state_dim  # Store state dimension
        self.action_dim = action_dim  # Store action dimension
        
        # Calculate total input dimension (all agents' states and actions)
        total_input_dim = num_agents * (state_dim + action_dim)  # 5 * (6 + 4) = 50
        
        # Define the first hidden layer (joint state-action to hidden)
        self.fc1 = nn.Linear(total_input_dim, hidden_dim * 2)  # 50 inputs to 256 hidden neurons
        # Define the second hidden layer (hidden to hidden)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)  # 256 to 128 hidden neurons
        # Define the third hidden layer (hidden to hidden)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)  # 128 to 64 hidden neurons
        # Define the output layer (hidden to Q-value)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)  # 64 to 1 Q-value output
        
        # Initialize weights using Xavier initialization
        self.init_weights()  # Call weight initialization method
        
    def init_weights(self):  # Method to initialize network weights
        # Initialize first layer weights
        nn.init.xavier_uniform_(self.fc1.weight)  # Xavier uniform initialization
        nn.init.constant_(self.fc1.bias, 0.0)  # Initialize biases to zero
        # Initialize second layer weights
        nn.init.xavier_uniform_(self.fc2.weight)  # Xavier uniform initialization
        nn.init.constant_(self.fc2.bias, 0.0)  # Initialize biases to zero
        # Initialize third layer weights
        nn.init.xavier_uniform_(self.fc3.weight)  # Xavier uniform initialization
        nn.init.constant_(self.fc3.bias, 0.0)  # Initialize biases to zero
        # Initialize output layer weights
        nn.init.xavier_uniform_(self.fc4.weight)  # Xavier uniform initialization
        nn.init.constant_(self.fc4.bias, 0.0)  # Initialize biases to zero
        
    def forward(self, states, actions):  # Forward pass through critic network
        # Flatten states and actions if they're not already flat
        if len(states.shape) > 2:  # Check if states need flattening
            batch_size = states.shape[0]  # Get batch size
            states = states.view(batch_size, -1)  # Flatten states
        else:  # If single sample
            states = states.view(1, -1)  # Reshape for single sample
            
        if len(actions.shape) > 2:  # Check if actions need flattening
            actions = actions.view(batch_size, -1)  # Flatten actions
        else:  # If single sample
            actions = actions.view(1, -1)  # Reshape for single sample
            
        # Concatenate all states and actions for centralized critic
        joint_input = torch.cat([states, actions], dim=-1)  # Combine states and actions
        
        # Apply ReLU activation to first layer output
        x = F.relu(self.fc1(joint_input))  # Non-linear transformation
        # Apply ReLU activation to second layer output
        x = F.relu(self.fc2(x))  # Another non-linear transformation
        # Apply ReLU activation to third layer output
        x = F.relu(self.fc3(x))  # Final hidden layer transformation
        # Return Q-value (single scalar output)
        q_value = self.fc4(x)  # Linear output for Q-value
        return q_value  # Return Q-value estimate

# Define the Multi-Agent Actor-Critic system for Smart Grid
class MARLSmartGrid:  # Main class containing all agents and networks
    def __init__(self, num_agents=5, state_dim=6, action_dim=4, hidden_dim=128):  # Constructor with system parameters
        self.num_agents = num_agents  # Store number of agents
        self.state_dim = state_dim  # Store state dimension
        self.action_dim = action_dim  # Store action dimension
        
        # Create individual actor networks for each agent
        self.actors = []  # List to store actor networks
        for i in range(num_agents):  # Loop through all agents
            actor = ActorNetwork(state_dim, action_dim, hidden_dim)  # Create actor for agent i
            self.actors.append(actor)  # Add actor to list
            
        # Create centralized critic network
        self.critic = CriticNetwork(state_dim, action_dim, num_agents, hidden_dim)  # Create critic
        
        # Create optimizers for all networks
        self.actor_optimizers = []  # List to store actor optimizers
        for actor in self.actors:  # Loop through all actors
            optimizer = optim.Adam(actor.parameters(), lr=0.001)  # Create Adam optimizer
            self.actor_optimizers.append(optimizer)  # Add optimizer to list
            
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)  # Critic optimizer
        
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device selection
        
        # Move all networks to the selected device
        for actor in self.actors:  # Loop through all actors
            actor.to(self.device)  # Move actor to device
        self.critic.to(self.device)  # Move critic to device
        
    def get_actions(self, states, exploration_noise=0.1):  # Method to get actions for all agents
        actions = []  # List to store actions
        for i, actor in enumerate(self.actors):  # Loop through all agents
            state = states[i]  # Get state for agent i
            # Convert state to tensor and move to device
            state_tensor = torch.FloatTensor(state).to(self.device)  # Convert to tensor
            action = actor.get_action(state_tensor, exploration_noise)  # Get action from actor
            actions.append(action)  # Add action to list
        return actions  # Return all actions
        
    def get_joint_actions_prob(self, states):  # Method to get joint action probabilities
        action_probs = []  # List to store action probabilities
        for i, actor in enumerate(self.actors):  # Loop through all agents
            state = states[i]  # Get state for agent i
            # Convert state to tensor and move to device
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Convert to tensor
            probs = actor.get_action_prob(state_tensor)  # Get action probabilities
            action_probs.append(probs)  # Add probabilities to list
        return action_probs  # Return all action probabilities
        
    def evaluate_joint_actions(self, states, actions):  # Method to evaluate joint actions using critic
        # Convert states and actions to tensors
        states_array = np.array(states)  # Convert list to numpy array first
        actions_array = np.array(actions)  # Convert list to numpy array first
        states_tensor = torch.FloatTensor(states_array).unsqueeze(0).to(self.device)  # Convert states
        
        # Convert actions to one-hot encoding
        actions_onehot = np.zeros((len(actions), self.action_dim))  # Create zero array
        for i, action in enumerate(actions):  # Loop through actions
            actions_onehot[i, action] = 1.0  # Set one-hot value
        actions_tensor = torch.FloatTensor(actions_onehot).unsqueeze(0).to(self.device)  # Convert actions
        
        # Get Q-value from critic
        q_value = self.critic(states_tensor, actions_tensor)  # Evaluate joint state-action
        return q_value.item()  # Return Q-value as scalar
        
    def save_models(self, filepath_prefix):  # Method to save all models
        # Save each actor network
        for i, actor in enumerate(self.actors):  # Loop through all actors
            torch.save(actor.state_dict(), f"{filepath_prefix}_actor_{i}.pth")  # Save actor
            
        # Save critic network
        torch.save(self.critic.state_dict(), f"{filepath_prefix}_critic.pth")  # Save critic
        
    def load_models(self, filepath_prefix):  # Method to load all models
        # Load each actor network
        for i, actor in enumerate(self.actors):  # Loop through all actors
            actor.load_state_dict(torch.load(f"{filepath_prefix}_actor_{i}.pth"))  # Load actor
            
        # Load critic network
        self.critic.load_state_dict(torch.load(f"{filepath_prefix}_critic.pth"))  # Load critic

# Example usage and testing function
def test_marl_system():  # Function to test the MARL system
    # Create MARL smart grid system
    marl_system = MARLSmartGrid(num_agents=5, state_dim=6, action_dim=4)  # Initialize system
    
    # Create sample states for all agents (battery SOC, solar, wind, price, demand, time)
    sample_states = []  # List to store sample states
    for i in range(5):  # Loop through all agents
        # Generate random state values for each agent
        state = np.random.rand(6)  # Random state with 6 dimensions
        sample_states.append(state)  # Add state to list
        
    print("Testing MARL Smart Grid System")  # Print test header
    print(f"Number of agents: {marl_system.num_agents}")  # Print number of agents
    print(f"State dimension per agent: {marl_system.state_dim}")  # Print state dimension
    print(f"Action dimension per agent: {marl_system.action_dim}")  # Print action dimension
    
    # Test action selection
    print("\nTesting action selection:")  # Print section header
    actions = marl_system.get_actions(sample_states, exploration_noise=0.1)  # Get actions
    for i, action in enumerate(actions):  # Loop through all agents
        print(f"Agent {i} action: {action}")  # Print action for each agent
        
    # Test action probabilities
    print("\nTesting action probabilities:")  # Print section header
    action_probs = marl_system.get_joint_actions_prob(sample_states)  # Get probabilities
    for i, probs in enumerate(action_probs):  # Loop through all agents
        print(f"Agent {i} action probabilities: {probs.squeeze().detach().cpu().numpy()}")  # Print probabilities
        
    # Test critic evaluation
    print("\nTesting critic evaluation:")  # Print section header
    q_value = marl_system.evaluate_joint_actions(sample_states, actions)  # Get Q-value
    print(f"Joint action Q-value: {q_value:.4f}")  # Print Q-value
    
    # Test model saving
    print("\nTesting model saving:")  # Print section header
    marl_system.save_models("test_models")  # Save models
    print("Models saved successfully!")  # Print success message
    
    # Test model loading
    print("\nTesting model loading:")  # Print section header
    marl_system.load_models("test_models")  # Load models
    print("Models loaded successfully!")  # Print success message
    
    print("\nMARL Smart Grid System test completed!")  # Print completion message

# Main execution block
if __name__ == "__main__":  # Run when script is executed directly
    test_marl_system()  # Run the test function
