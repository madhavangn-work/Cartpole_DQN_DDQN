import numpy as np
np.random.seed(0)
from collections import deque
import gym
from keras.models import load_model
from src.Utils.EpsilonGreedy import EpsilonGreedy
from src.Utils.Network import Network
import keras
keras.utils.set_random_seed(0)

class DQNAgent():
    def __init__(self, strat=EpsilonGreedy()):
        # Initialize the DQNAgent class with an optional strategy (default: EpsilonGreedy)
        self.env = gym.make("CartPole-v1")  # Create the CartPole environment
        self.env.seed(0)  # Set a seed for reproducibility
        self.state_size = self.env.observation_space.shape[0]  # Get the size of the state space
        self.action_space = self.env.action_space.n  # Get the number of possible actions

        self.gamma = 0.99  # Discount factor for future rewards
        self.batch_size = 32  # Number of samples in each training batch
        self.memory = deque(maxlen=2_000)  # Replay memory to store past experiences

        self.online_network = Network().build_model()  # Build the neural network model for the online network

        self.EGS = strat  # Set the exploration strategy

        self.q_min = []  # List to store the minimum Q-values during training
        self.q_mean = []  # List to store the mean Q-values during training
        self.q_max = []  # List to store the maximum Q-values during training
        self.rewards = []  # List to store the rewards obtained during training
        self.epsilons = []  # List to store the epsilon values used during training
        self.test_scores = []  # List to store the test scores used during testing


    def load(self, name="dqn"):
        """
        Load a trained model from disk.

        Arguments:
        - name: The name of the model to load.
        """
        self.online_network = load_model(f"./saved_models/{name}/trained_{name}_model")  # Load the trained model from disk and assign it to the online_network attribute

    def save(self, name="dqn"):
        """
        Save the current model to disk.

        Arguments:
        - name: The name of the model to save.
        """
        self.online_network.save(f"./saved_models/{name}/trained_{name}_model")  # Save the current model to disk at the specified path and filename

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in the agent's memory.

        Arguments:
        - state: The current state.
        - action: The action taken.
        - reward: The reward received.
        - next_state: The next state.
        - done: A flag indicating whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))  # Store the experience as a tuple in the agent's memory
        self.EGS.update_epsilon()  # Update the exploration strategy's epsilon value

    def replay(self):
        """
        Train the online network using a batch of experiences from the memory.
        """
        if len(self.memory) < self.batch_size:
            return

        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)  # Randomly select batch indices from the memory
        batch = [self.memory[index] for index in batch_indices]  # Get the batch of experiences corresponding to the selected indices
        states, actions, rewards, next_states, dones = zip(*batch)  # Unpack the batch into separate lists of states, actions, rewards, next states, and done flags

        q_values = self.online_network.predict(np.squeeze(states), verbose=0)  # Get Q-values for the current states using the online network
        next_q_values = self.online_network.predict(np.array(np.squeeze(next_states)), verbose=0)  # Get Q-values for the next states using the online network

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]  # If the episode is done, set the Q-value of the action to the received reward
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])  # Otherwise, update the Q-value using the Bellman equation
        
        self.online_network.fit(np.squeeze(states), q_values, batch_size=self.batch_size, verbose=0)  # Train the online network using the states and updated Q-values

    def train(self, name):
        """
        Train the DQN agent.

        Arguments:
        - name: The name to use when saving the trained model.
        """
        for epoch in range(1, 1001):
            state = self.env.reset()  # Reset the environment and get the initial state
            state = np.reshape(state, [1, self.state_size])  # Reshape the state to match the input shape of the network
            done = False  # Flag to indicate if the episode is done
            time_steps = 0  # Counter for the number of time steps in the episode
            qlist = []  # List to store the maximum Q-values during the episode

            while not done:
                q_val = self.online_network.predict(state, verbose=0)  # Get the Q-values for the current state
                action = self.EGS.act(q_val)  # Choose an action based on the exploration strategy
                qlist.append(np.amax(q_val))  # Store the maximum Q-value in qlist
                next_state, reward, done, _ = self.env.step(action)  # Take the chosen action in the environment
                next_state = np.reshape(next_state, [1, self.state_size])  # Reshape the next state

                # Adjust the reward for incomplete episodes
                reward = reward if not done or time_steps == self.env._max_episode_steps - 1 else -100

                self.remember(state, action, reward, next_state, done)  # Store the experience in the agent's memory

                state = next_state  # Update the current state

                time_steps += 1  # Increment the time steps counter

                if done:
                    self.epsilons.append(self.EGS.eps_start)  # Store the starting epsilon value for the episode
                    self.rewards.append(time_steps)  # Store the total number of time steps as the episode's score
                    self.q_min.append(np.min(qlist))  # Store the minimum Q-value during the episode
                    self.q_mean.append(np.mean(qlist))  # Store the mean Q-value during the episode
                    self.q_max.append(np.max(qlist))  # Store the maximum Q-value during the episode
                    print(f'Epoch: {epoch} of {1000}  |  Score: {time_steps}  |  Exploration: {self.EGS.eps_start}')

                    if time_steps == self.env._max_episode_steps:
                        self.save(name)  # Save the trained model when the maximum time steps are reached
                        return
                self.replay()  # Train the online network using a batch of experiences from memory

    def test(self, name, epochs):
        """
        Test the DQN agent using a trained model.

        Arguments:
        - name: The name of the model to load for testing.
        - epochs: Number of epochs to run for testing
        """
        self.load(name)  # Load the trained model for testing

        for epoch in range(1, epochs+1):
            state = self.env.reset()  # Reset the environment and get the initial state
            state = np.reshape(state, [1, self.state_size])  # Reshape the state to match the input shape of the network
            done = False  # Flag to indicate if the episode is done
            time_steps = 0  # Counter for the number of time steps in the episode

            while not done:
                self.env.render()  # Render the environment
                action = np.argmax(self.online_network.predict(state, verbose=0))  # Choose the action with the highest Q-value
                next_state, _, done, _ = self.env.step(action)  # Take the chosen action in the environment
                next_state = np.reshape(next_state, [1, self.state_size])  # Reshape the next state
                state = next_state  # Update the current state
                time_steps += 1  # Increment the time steps counter

                if done:
                    self.test_scores.append(time_steps)
                    print(f'Epoch: {epoch} of {1000}  |  Score: {time_steps}')
                    break
        self.env.close()  # Close the environment after testing

    def get_plot_values(self):
        """
        Get the values for plotting during training.

        Returns:
        A dictionary containing the values for plotting:
        - 'q_min': List of minimum Q-values during training.
        - 'q_mean': List of mean Q-values during training.
        - 'q_max': List of maximum Q-values during training.
        - 'rewards': List of rewards obtained during training.
        - 'epsilons': List of epsilon values used during training.
        """
        return {
            'q_min': self.q_min,  # List of minimum Q-values during training
            'q_mean': self.q_mean,  # List of mean Q-values during training
            'q_max': self.q_max,  # List of maximum Q-values during training
            'rewards': self.rewards,  # List of rewards obtained during training
            'epsilons': self.epsilons  # List of epsilon values used during training
        }