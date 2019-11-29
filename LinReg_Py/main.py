import pandas as panda
import matplotlib.pyplot as plot
from mpl_toolkits import mplot3d
import numpy as np

# parse data
data = panda.read_csv('salaries-by-college-type.csv')
X = data.loc[:, 'Acceptance Rate']
Y = data.loc[:, 'Yield Rate']
Z = data.loc[:, 'Starting Median Salary']

# configuration
learning_rate = 0.00001
epochs = 1000

def descend(iterations, learning_rate):
    m, n, b = 0, 0, 0
    count = float(len(X))
    print("Provided " + str(count) + " Samples")
    # steps
    for i in range(iterations):
        # get predicted SIP's to calculate error for this step
        pred = m*X + n*Y + b
        # take partial derivatives wrt m, n, b of least-squared error cost function
        d_wrt_m = (-2/count) * sum((X) * (Z - pred))
        d_wrt_n = (-2/count) * sum((Y) * (Z - pred))
        d_wrt_b = (-2/count) * sum(Z - pred)
        # update parameters to "descend" to lower error in next step
        m -= learning_rate * d_wrt_m
        n -= learning_rate * d_wrt_n
        b -= learning_rate * d_wrt_b
    return m, n, b

# calculate optimal parameters
m, n, b = descend(epochs, learning_rate)
print("-----------------", "\nAcceptance Rate Coefficient (I): ", m, "\Yield Rate Coefficient (U): ", n, "\nIntercept: ", b)
print("-----------------", "\nStarting Median Salary = " + str(round(m, 3)) + "I + " + str(round(n, 3)) + "U + " + str(round(b, 3)))

# setup plot
fig = plot.figure()
fig.suptitle("Starting Median Salary vs. [Acceptance Rate & Yield Rate]", fontsize=13, color="blue")
ax = fig.gca(projection='3d')
ax.set_xlabel("Acceptance Rate")
ax.set_ylabel("Yield Rate")
ax.set_zlabel("Starting Median Salary")

# lin reg plane
x_s = np.linspace(min(X),max(X),10)
y_s = np.linspace(min(Y),max(Y),10)
X_s,Y_s = np.meshgrid(x_s,y_s)
Z_s = m*X_s + n*Y_s + b
ax.plot_surface(X_s, Y_s, Z_s, alpha=0.5, color="red")

# plot real points
ax.scatter3D(X, Y, Z, marker='.', color="blue");
plot.show()
