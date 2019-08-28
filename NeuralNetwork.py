# importing some things here: numpy for the array stuff
import numpy as np


# Create a class for objectifying our Neural Network
class NeuralNetwork:
    def __init__(self):
        # Create a random seed
        np.random.seed(1)
        # Making a set of random weights to start off with
        self.synaptic_weights = 2 * np.random.random((3, 1)) / 654325181

    # Defining a sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Defining a sigmoid derivative function
    def sig_der(self, x):
        return x * (1 - x)

    # Defining out training function
    def train(self, training_inputs, training_outputs, training_iterations):
        # For loop that lasts the number of iterations
        for x in range(training_iterations):
            # A check to print the current weights twice during the iterations
            if x % (training_iterations / 2) == 0:
                print("\nAfter ", x, " training sessions:")
                print("Weights: \n", self.synaptic_weights)
            # Create a 'result' for this iteration
            output = self.think(training_inputs)
            # Calculate the error value from this iteration
            error = training_outputs - output
            # Make some adjustments using our error value and the sigmoid derivative
            adjustments = error * self.sig_der(output)
            # Modify the weights based on the inputs and the adjustments
            self.synaptic_weights += np.dot(training_inputs.T, adjustments)
            # On the last iteration, print out the final weights we will use
            if x == training_iterations - 1:
                print("\nFinal weights: \n ", self.synaptic_weights)

    # Create the think method
    def think(self, inputs):
        # Convert the inputs into floats
        inputs = inputs.astype(float)
        # Make an output that multiplies out inputs by the current weights
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        # Give the best guess based on our current weights
        return output
