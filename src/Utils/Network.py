from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import RMSprop
import keras
keras.utils.set_random_seed(0)

class Network():
    def __init__(self, learning_rate=0.00025, rho=0.95, epsilon=0.01):
        self.lr = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def build_model(self, observation_size=(4,), action_size=2):
        """
        Build the neural network model for the DQN agent.

        Arguments:
        - observation_size: The shape of the input observation.
        - action_size: The number of possible actions.

        Returns:
        - The compiled Keras model.
        """
        In = Input(observation_size)  # Input layer with the shape of the observation
        X = Dense(256, activation='relu', kernel_initializer='he_normal')(In)  # Dense layer with 256 units and ReLU activation
        X = Dense(128, activation='relu', kernel_initializer='he_normal')(X)  # Dense layer with 128 units and ReLU activation
        X = Dense(64, activation='relu', kernel_initializer='he_normal')(X)  # Dense layer with 64 units and ReLU activation
        X = Dense(32, activation='relu', kernel_initializer='he_normal')(X)  # Dense layer with 32 units and ReLU activation
        Out = Dense(action_size, kernel_initializer='he_normal')(X)  # Output layer with the number of units equal to the action size

        model = Model(inputs=In, outputs=Out)  # Create a model with the defined layers
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.lr, rho=self.rho, epsilon=self.epsilon), metrics=['acc'])  # Compile the model with MSE loss and RMSprop optimizer

        return model