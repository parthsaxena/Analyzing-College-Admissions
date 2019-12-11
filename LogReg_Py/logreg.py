import numpy as np
import matplotlib.pyplot as plt

epochs = 10000000 # number of gradient descent iterations 
alpha = 0.05 # learning rate

input_file = './data/output_usc.csv';

# element-wise sigmoid function 
def sig(theta, x):
    return 1./(1+np.exp(theta @ x.T))

# cross-entropy loss-function
def cost(theta, x, y):
    return -1 * (1./y.size) * (np.log(sig(theta, x)+1e-9) @ y) + (np.log(1-sig(theta, x)+1e-9) @ (1-y))

# finds parameter set to minimize cost function 
def gradient_descent(x, y, epochs, alpha):
    theta = np.full((1, (int) (x.size/y.size)), 0.00001) # initializes arbritrary starting parameters
    print("Initial cost: " + cost(theta, x, y))
    for i in range(epochs):
        theta -= (alpha/y.size) * ((y.T - sig(theta, x)) @ x)
    print("Parameter vector: " + theta)
    return theta

# reads in college data
data = np.genfromtxt(input_file, delimiter=',')
x = data[:, :-1] # input matrix 
y = (data[:, 2:]) # output vector 
x = np.column_stack((x, np.ones((y.size, 1)))) # appends column of 1s to x for biases

theta = gradient_descent(x, y, epochs, alpha)

print("Final cost: " + cost(theta, x, y))

# evaluates accuracy of model by testing it on training data
# range of (0,1)
# lower values indicate a more accurate model
output = sig(theta, x)
misclassified = 0
for i in range(y.size-1):
    if (int(output[0][i]+0.5) != y[i]):
        misclassified += 1
print("Training acuracy: " + 1 - (misclassified/y.size))
