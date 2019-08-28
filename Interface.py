# Importing a few things we need here: Tkinter for the interface,
# numpy for the array stuff, Neural Network for obvious reasons,
# askcolor to have a simple way to get a RGB value from the user.
import tkinter as tk
import numpy as np
import NeuralNetwork as nn
from tkcolorpicker import askcolor

# Create the Neural Network object
colorNN = nn.NeuralNetwork()
# Create an array of test data to train the network
NN_inputs = np.array([[0, 0, 0],
                      [0.33, 0, 0],
                      [0, 0.33, 0],
                      [0, 0, 0.33],
                      [0.66, 0, 0],
                      [0, 0, 0.66],
                      [0.66, 0.33, 0],
                      [1, 1, 1]])
# Create and transpose an array of outputs to train the network
NN_outputs = np.array([[0, 0, 0, 0, 1, 1, 1, 1]]).T
# Call the training method, passing out input and output with
# the number of times to train
colorNN.train(NN_inputs, NN_outputs, 100000)


# Create the command for the button that interacts with the user
def colorPickCommand():
    # Ask for the user to pick a color
    newColor = askcolor((0, 0, 0), root)
    # Create a variable with the Red, Blue, and Green values from
    # the color the user chose
    pickedColor = np.array([[(newColor[0][0] / 255),
                             (newColor[0][1] / 255),
                             (newColor[0][2] / 255)]])
    # Change the canvas background color to match the color the
    # user chose
    canvas.configure(bg=newColor[1])
    # Delete all the objects on the canvas to prepare for new
    # objects to be created
    canvas.delete("all")
    # Create a result variable for validation checking
    result = colorNN.think(pickedColor)

    # Check if our result is above or below our threshold for choosing
    # if we would want to use a light or dark colored text
    if result <= 0.7:
        canvas.create_text(100, 100, text="Use Light Text!", fill="White")
    else:
        canvas.create_text(100, 100, text="Use Dark Text!", fill="Black")


# Create a TKinter object
root = tk.Tk()
# Create a canvas to show some colors on
canvas = tk.Canvas(root, width=200, height=200)
# Create a simple text object
canvas.create_text(100, 100, text="Press the button!", tags="text")
# Create a button to fire the command so we can get input
colorButton = tk.Button(root, text="Pick A Color ..", command=colorPickCommand)
# Pack both of those to the interface
canvas.pack()
colorButton.pack()

# AND GO!
root.mainloop()
