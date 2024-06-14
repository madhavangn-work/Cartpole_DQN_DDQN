# Enhancing Deep Reinforcement Learning: A Comparative Study of DQN and Double DQN Algorithms for Optimal Decision-Making

## Description:
This project aims to enhance deep reinforcement learning by conducting a comparative study between the Deep Q-Network (DQN) and Double Deep Q-Network (DDQN) algorithms. The evaluation of these algorithms will be performed using the CartPole-v1 environment from OpenAI Gym.

## Installation:
To install the required dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Usage:
To train the DQN and DDQN algorithms, execute the `train.py` script. The trained models will be saved to the `saved_models` directory. The resulting plots will be saved in the `plots` directory. Additionally, the plot values will be stored in a CSV file within the `data` directory.

Please note that the training process may take approximately 30 to 40 minutes. Once the models are trained, you can use the `test.py` script to evaluate these models by loading the weights from the saved models.

The mathematical analysis of these values is performed using the R programming language. The analysis code is available in the R Markdown file named `SCC.462.Rmd`.