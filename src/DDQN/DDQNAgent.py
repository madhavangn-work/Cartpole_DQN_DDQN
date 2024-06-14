from src.Utils.EpsilonGreedy import EpsilonGreedy
from src.DQN.DQNAgent import DQNAgent
from src.Utils.Network import Network
import numpy as np
np.random.seed(0)

class DDQNAgent(DQNAgent):
    def __init__(self, strat=EpsilonGreedy()):
        super().__init__(strat)
        self.target_network = Network().build_model()  # Create the target network for Double DQN

    def update_target_network(self):
        """
        Update the target network by setting its weights to the weights of the online network.
        """
        self.target_network.set_weights(self.online_network.get_weights())  # Copy weights from the online network to the target network

    def replay(self):
        """
        Train the online network using a batch of experiences from the memory with Double DQN.
        """
        if len(self.memory) < self.batch_size:
            return
        
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)  # Randomly select batch indices from the memory
        batch = [self.memory[index] for index in batch_indices]  # Get the batch of experiences corresponding to the selected indices
        states, actions, rewards, next_states, dones = zip(*batch)  # Unpack the batch into separate lists of states, actions, rewards, next states, and done flags

        q_values = self.online_network.predict(np.squeeze(states), verbose=0)  # Get Q-values for the current states using the online network
        next_q_values = self.online_network.predict(np.squeeze(next_states), verbose=0)  # Get Q-values for the next states using the online network
        next_q_values_target = self.target_network.predict(np.array(np.squeeze(next_states)), verbose=0)  # Get Q-values for the next states using the target network

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]  # If the episode is done, set the Q-value of the action to the received reward
            else:
                action = np.argmax(next_q_values[i])  # Choose the action with the highest Q-value according to the online network
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values_target[i][action])  # Update the Q-value using the Double DQN algorithm
        
        self.online_network.fit(np.squeeze(states), q_values, batch_size=self.batch_size, verbose=0)  # Train the online network using the states and updated Q-values

    def train(self, name):
        """
        Train the DDQN agent.

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

                reward = reward if not done or time_steps == self.env._max_episode_steps - 1 else -100  # Adjust the reward for incomplete episodes

                self.remember(state, action, reward, next_state, done)  # Store the experience in the agent's memory

                state = next_state  # Update the current state
                time_steps += 1  # Increment the time steps counter

                if done:
                    self.update_target_network()  # Update the target network
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