# Perceptron Algorithm for Binary Classification

In this code, we are implementing the perceptron algorithm for binary classification using numpy and matplotlib libraries. The goal is to classify the data points into two classes (0 or 1) using a linear decision boundary.

## Dataset
We define a dataset X with 10 data points, each having two features, and a corresponding binary class label y. The data points are randomly generated using numpy.

```python 
import numpy as np

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
```

## Perceptron Algorithm
The perceptron algorithm is a binary classification algorithm that finds a linear decision boundary to separate the two classes. The algorithm takes the feature vectors X and their corresponding class labels y as inputs.

```python 
w = np.random.rand(2)  # initialize weights
b = np.random.rand()   # initialize bias
epochs = 15             # number of iterations over the entire dataset

for epoch in range(epochs):
    for i in range(X.shape[0]):
        s = w[0]*X[i][0] + w[1]*X[i][1] + b    # calculate weighted sum
        if s > 0:
            prediction = 1
        else:
            prediction = 0
        error = y[i] - prediction                # calculate prediction error
        w[0] += error*X[i][0]                    # update weight 1
        w[1] += error*X[i][1]                    # update weight 2
        b += error                               # update bias
```
In each iteration of the algorithm, the weights and bias are updated based on the prediction error. The algorithm tries to minimize the error between the predicted class and the actual class.

## Visualization
We visualize the decision boundary and the data points using matplotlib.


```python 
import matplotlib.pyplot as plt

x_line = np.linspace(0, 1, 10)                          # create x-axis values
y_line = (-w[0]*x_line - b) / w[1]                       # calculate y-axis values

plt.plot(x_line, y_line, color='blue')                   # plot decision boundary
plt.scatter(X[:,0][y==0], X[:,1][y==0], color='green', label='0')  # plot data points for class 0
plt.scatter(X[:,0][y==1], X[:,1][y==1], color='red', label='1')    # plot data points for class 1
plt.legend()
plt.show()
The green and red dots represent the two classes, and the blue line represents the decision boundary separating them.
```
## After each epoch
<div>
<style>
.carousel {
  overflow: hidden;
  white-space: nowrap;
}

.carousel img {
  display: inline-block;
  width: 100%;
  height: auto;
  transition: transform 0.3s ease-in-out;
}

.carousel img:not(:first-child) {
  transform: translateX(-100%);
}

.carousel:hover img:not(:hover) {
  transform: translateX(-200%);
}
</style>
<div class="carousel">
  <img src="/results/result_0-ep.png" alt="Image 1">
  <img src="/results/result_1-ep.png" alt="Image 2">
  <img src="/results/result_2-ep.png" alt="Image 3">
</div>
</div>
# Conclusion
The perceptron algorithm is a simple yet powerful algorithm for binary classification. It can be used as a building block for more complex algorithms and neural networks. The code we have implemented can be used as a starting point for further experimentation and learning.
