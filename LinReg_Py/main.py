import pandas as panda
import matplotlib.pyplot as plot
from mpl_toolkits import mplot3d
import numpy as np

# parse data
Stock_Market = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]
                }
df = panda.DataFrame(Stock_Market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])
X = df['Interest_Rate']
Y = df['Unemployment_Rate']
Z = df['Stock_Index_Price']

# configuration
learning_rate = 0.01
epochs = 1000

def descend(iterations, learning_rate):
    m, n, b = 0, 0, 0
    count = float(len(X))
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
print("-----------------", "\nInterest Rate Coefficient (I): ", m, "\nUnemployment Rate Coefficient (U): ", n, "\nIntercept: ", b)
print("-----------------", "\nStock Index Price = " + str(round(m, 3)) + "I + " + str(round(n, 3)) + "U + " + str(round(b, 3)))

# setup plot
fig = plot.figure()
fig.suptitle("Stock Index Price vs. [Interest Rate & Unemployment Rate]", fontsize=13, color="blue")
ax = fig.gca(projection='3d')
ax.set_xlabel("Interest Rate")
ax.set_ylabel("Unemployment Rate")
ax.set_zlabel("Stock Index Price")

# lin reg plane
x_s = np.linspace(min(X),max(X),10)
y_s = np.linspace(min(Y),max(Y),10)
X_s,Y_s = np.meshgrid(x_s,y_s)
Z_s = m*X_s + n*Y_s + b
ax.plot_surface(X_s, Y_s, Z_s, alpha=0.5, color="red")

# plot real points
ax.scatter3D(X, Y, Z, marker='.', color="blue");
plot.show()
