import sys
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
import time
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from mpl_toolkits import mplot3d

N = 50

#create random points (50) with approximate slope of 1

X = np.array([np.linspace(0,10,num=N),np.linspace(0,10,num=N)])
X+= np.random.randn(2, N)

#place x and y in matrix/array
x,y = X


fig, ax = plt.subplots()
print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

#create array that multiplies by hypothesis - [x, 1]

bias_with_x = np.array([(1., a) for a in x]).astype(np.float32)

#define step count and learning rate
numsteps = 500
lrate = .02


input = bias_with_x
target = np.transpose([y]).astype(np.float32)

#randomize initial values of m and b to an array/matrix
hyp = np.random.rand(2,1) *0.5
#set b to -5 just to test
hyp[[0]] = -5

#draws the graph of the points
ax.scatter(x, y)


hyp0array = []
hyp1array = []
#------------
#repeats this loop numsteps times
for count in range (numsteps):
    temp0 = float(hyp[0])
    temp1 = float(hyp[1])
    hyp0array.append(temp0)
    hyp1array.append(temp1)

    #creates the prediction matrix/array by multiplying hypothesis by [x, 1] matrix/array

    prediction = np.matmul(bias_with_x,hyp)

    #predicted values of m and b separated, so I can plug them into the partial derivative formulas

    predb = [row[0] for row in bias_with_x]*hyp[0]
    predmx = [row[1] for row in bias_with_x]*hyp[1]


    error = np.subtract(prediction,target)
    #mean squared error of error
    loss = np.sum(error ** 2)/50

    #calculates partial derivatives
    dm = (2.0/N)*np.sum((-1*x)*(y-(predmx+predb)))
    db = (2.0/N)*(-1)*np.sum(y-(predmx+predb))


    #updates the weights
    hyp[0] = hyp[0] - db*lrate
    hyp[1] = hyp[1] - dm*lrate
#-----------------------------------
#creates the animation of the line's changing slope and y-intercept over 500 steps

ax.plot([0,10],[hyp[0],hyp[1]*10+hyp[0]],color='k', linestyle='-', linewidth=2)
def update(i):
        label = 'timestep {0}'.format(i)

        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        del ax.lines[0]
        line = ax.plot([0,10],[hyp0array[i],hyp1array[i]*10+hyp0array[i]],color='k', linestyle='-', linewidth=2)
        ax.set_xlabel(label)
        return line, ax
if __name__ == '__main__':

    anim = FuncAnimation(fig, update, frames=np.arange(0, 500), interval=25)


#prints out values of m and b
print(hyp[1])
print(hyp[0])
b_with_x = []
def f(m,b):
    hyp2 = [[m],[b]]
    return np.subtract(np.matmul(bias_with_x,hyp2),target)

'''
xplot = np.linspace(-6, 6, 30)
yplot = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(xplot, yplot)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('m')
ax.set_ylabel('b')
ax.set_zlabel('error');
'''
HTML(anim.to_html5_video())
