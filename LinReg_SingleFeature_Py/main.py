import pandas as panda
import matplotlib.pyplot as plot

plot.rcParams['figure.figsize'] = (8, 5)

data = panda.read_csv('sat_gpa.csv')
X = data.loc[:, 'GPA']
Y = data.loc[:, 'SAT']

learning_rate = 0.001
epochs = 1000

def descend(iterations, learning_rate):
    m = 0
    b = 0
    n = float(len(X))
    for i in range(iterations):
        pred = m*X + b
        d_wrt_m = (-2/n) * sum((X) * (Y - pred))
        d_wrt_b = (-2/n) * sum((Y - pred))
        m -= learning_rate * d_wrt_m
        b -= learning_rate * d_wrt_b
        #print(m, b)
    print(m, b)
    return m, b

m, b = descend(epochs, learning_rate)
Y_pred = m*X + b

plot.title("Unweighted GPA vs. SAT Score")
plot.xlabel('GPA (UW)')
plot.ylabel('SAT (2400)')
plot.scatter(X, Y)
plot.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='purple')
plot.show()
