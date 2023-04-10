import numpy as np
import matplotlib.pyplot as plt

# Define the dataset
X = np.array([[0.1, 0.5],
              [0.2, 0.6],
              [0.3, 0.7],
              [0.4, 0.8],
              [0.5, 0.9],
              [0.6, 0.1],
              [0.7, 0.2],
              [0.8, 0.3],
              [0.9, 0.4],
              [1.0, 0.5]])
y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Initialize weights and bias with random values
w = np.random.rand(2)
b = np.random.rand()

# Define the line equation for plotting
x_line = np.linspace(0, 1, 10)

# Define the training loop with specified number of epochs
epochs = 15
for epoch in range(epochs):

    # Loop over all data points and update weights and bias
    for i in range(X.shape[0]):
        s = np.dot(w, X[i]) + b
        prediction = 1 if s > 0 else 0
        error = y[i] - prediction
        w += error * X[i]
        b += error

    # Update the line equation for plotting
    y_line = (-w[0]*x_line - b) / w[1]

    # Plot the current state of the classifier
    plt.plot(x_line, y_line, color='blue')
    plt.scatter(X[:, 0][y==0], X[:, 1][y==0], color='green', label='0')
    plt.scatter(X[:, 0][y==1], X[:, 1][y==1], color='red', label='1')
    plt.legend()
    plt.show()
