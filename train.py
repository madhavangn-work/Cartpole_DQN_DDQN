from src.DQN.DQNAgent import DQNAgent
from src.DDQN.DDQNAgent import DDQNAgent
from src.Utils.PlotHelper import Plotter
import pandas as pd

######################################################################################################
###################################---DQNAgent and DDQNAgent Initialization---########################
######################################################################################################

dqn_agent = DQNAgent()  # Create DQNAgent instance
ddqn_agent = DDQNAgent()  # Create DDQNAgent instance

######################################################################################################
###################################---Training DQNAgent---############################################
######################################################################################################

dqn_agent.train("dqn")  # Train DQNAgent

######################################################################################################
###################################---Saving DQN Plot Values---#######################################
######################################################################################################

dqn_plot_values = dqn_agent.get_plot_values()  # Get plot values from DQNAgent

df_dqn = pd.DataFrame.from_dict(dqn_plot_values, orient="index")  # Convert plot values to DataFrame

df_dqn.to_csv("./data/dqn.csv")  # Save DQN plot values to CSV file

######################################################################################################
###################################---Training DDQNAgent---###########################################
######################################################################################################

ddqn_agent.train("ddqn")  # Train DDQNAgent

######################################################################################################
###################################---Saving DDQN Plot Values---######################################
######################################################################################################

ddqn_plot_values = ddqn_agent.get_plot_values()  # Get plot values from DDQNAgent

df_ddqn = pd.DataFrame.from_dict(ddqn_plot_values, orient="index")  # Convert plot values to DataFrame

df_ddqn.to_csv("./data/ddqn.csv")  # Save DDQN plot values to CSV file

######################################################################################################
#######################################---Plotting---#################################################
######################################################################################################

Plotter(dqn_plot_values, ddqn_plot_values).save_plots()  # Create Plotter instance and save plots