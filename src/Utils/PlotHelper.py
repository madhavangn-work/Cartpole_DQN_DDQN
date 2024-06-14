import matplotlib.pyplot as plt

class Plotter():
    def __init__(self, dqn_plot_values, ddqn_plot_values):
        # Initialize values for DQN plots
        self.dqn_q_min = dqn_plot_values['q_min']
        self.dqn_q_mean = dqn_plot_values['q_mean']
        self.dqn_q_max = dqn_plot_values['q_max']
        self.dqn_rewards = dqn_plot_values['rewards']
        self.dqn_epsilons = dqn_plot_values['epsilons']

        # Initialize values for DDQN plots
        self.ddqn_q_min = ddqn_plot_values['q_min']
        self.ddqn_q_mean = ddqn_plot_values['q_mean']
        self.ddqn_q_max = ddqn_plot_values['q_max']
        self.ddqn_rewards = ddqn_plot_values['rewards']
        self.ddqn_epsilons = ddqn_plot_values['epsilons']

    def save_plot(self, data1, data2, ylabel, title, filename, qval=False):
        # Determine the minimum length of data points for both sets
        min_length = min(len(data1), len(data2))

        # Create a new figure
        plt.figure()

        # Plot the data points for the first set (DQN)
        plt.plot(data1[:min_length], label='DQN', color='blue')

        # Plot the data points for the second set (DDQN)
        plt.plot(data2[:min_length], label='DDQN', color='orange')

        # Set the labels and title of the plot
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.title(title)

        if qval:
            # Fill the area between the two sets of data points
            plt.fill_between(range(min_length), data1[:min_length], data2[:min_length], color='blue', alpha=0.2)

            # Fill the area between the minimum and maximum Q-values for DQN
            plt.fill_between(range(min_length), self.dqn_q_min[:min_length], self.dqn_q_max[:min_length], color='blue', alpha=0.1)

            # Fill the area between the minimum and maximum Q-values for DDQN
            plt.fill_between(range(min_length), self.ddqn_q_min[:min_length], self.ddqn_q_max[:min_length], color='orange', alpha=0.1)

            # Plot the minimum Q-values for DQN with dashed line
            plt.plot(self.dqn_q_min[:min_length], '--', color='blue')

            # Plot the maximum Q-values for DQN with dashed line
            plt.plot(self.dqn_q_max[:min_length], '--', color='blue')

            # Plot the minimum Q-values for DDQN with dashed line
            plt.plot(self.ddqn_q_min[:min_length], '--', color='orange')

            # Plot the maximum Q-values for DDQN with dashed line
            plt.plot(self.ddqn_q_max[:min_length], '--', color='orange')

        # Add legend to the plot
        plt.legend()

        # Save the plot as an image
        plt.savefig(filename)

        # Close the plot
        plt.close()

    def save_plots(self):
        # Save the plot comparing Q-values for DQN and DDQN
        self.save_plot(self.dqn_q_mean, self.ddqn_q_mean, 'Q Values', 'Q Values Over Episodes', './plots/q_values_plot.png', qval=True)

        # Save the plot comparing rewards for DQN and DDQN
        self.save_plot(self.dqn_rewards, self.ddqn_rewards, 'Rewards', 'Rewards Over Episodes', './plots/rewards_plot.png')

        # Save the plot comparing epsilons for DQN and DDQN
        self.save_plot(self.dqn_epsilons, self.ddqn_epsilons, 'Epsilons', 'Epsilons Over Episodes', './plots/epsilons_plot.png')