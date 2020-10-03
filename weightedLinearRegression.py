import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import ceil
from scipy import linalg
import matplotlib.pyplot as plt



data = pd.read_csv('Dataset2.txt', skiprows=(1) , sep=";", names=('height','weight','male'))



# Normialization -> Dividing the column by the max value of the column

maxweight = data['weight'].max()

maxheight = data['height'].max()

data['weight'] = data['weight']/maxweight

data['height'] = data['height']/maxheight

# x = height , y = weight

Y = data.drop(['height','male'],axis=1)

X = data.drop(['weight','male'],axis = 1)

# intercept = np.ones((X.shape[0], 1))
# X = np.concatenate((intercept, X), axis=1)

# print(X.head)
# print(Y)

def J(x,y,weights,theta):

    squares = weights*(np.power(((x @ theta.T) - y) , 2))
    return np.sum( squares ) / 2

def gradient_descent(x,y,theta,alpha,weights,num_iters):


    m = y.size  # number of training examples
    for i in range(num_iters):
        if alpha < 0.01 :
            alpha *= 2

        Xi = x
        Yi = y


        Yi = Yi.reshape(60,1)
        R = (Xi @ theta.T) - Yi

        theta = theta - (alpha/len(Xi) * np.sum(weights * R * Xi, axis=0))
        cost = J(x,y,weights,theta)
        # print(cost)

    return (theta, cost)


############# Kernel 

def kernel_function(xi,x0,tau= .005): 
    return np.exp(-(xi - x0)**2/(2*tau))


def locally_weighted_linear_regression(x, y):


    tau = 0.01
    n = len(x)
    yest = np.zeros(n)

    x = x['height'].to_numpy()

    y = y['weight']

    w = np.array([    np.exp(- (x - x[i])**2/(2*tau)) for i in range(n)])     

    # We Do it for each X
    for i in range(n):
        weights = w[:, i]
        #
        # theta0 = 1.
        # theta1 = 1.
        # theta = np.array([[theta1,theta0]])
        # alpha = 0.001
        # num_iters = 1000
        # thetas , cost = gradient_descent(x,y,theta,alpha,weights,num_iters)
        # print(thetas)
        # print(x[i])
        # yest[i] = thetas[0,0] * x[i,0 ] + thetas[0,1] 

        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta = linalg.solve(A, b)
        yest[i] = theta[0] + theta[1] * x[i]


    return yest

# X = X.tolist()
# Y = Y.tolist()
ys = locally_weighted_linear_regression(X,Y)
xs = X.to_numpy()
print(xs)
print(ys)
xs, ys = zip(*sorted(zip(xs, ys)))

# result = result.reshape(60,1)
plt.scatter(X,Y,color='g')

plt.scatter(xs,ys,color='r')
plt.plot(xs,ys,color='r')

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


# for simple linear regression
abline(0.536,0.156)

# plt.axis('equal')
plt.show()

