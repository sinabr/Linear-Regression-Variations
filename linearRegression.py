import pandas as pd
import numpy as np
import math
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

    

def funct(x,theta):
    return x @ theta.T

def J_theta(x,y,theta):

    squares = np.power(((x @ theta.T).values - y) , 2)
    return np.sum( squares ) / 2

def batch_gradient_descent(x,y,theta,alpha,num_iters):


    m = y.size  # number of training examples
    for i in range(num_iters):
        if alpha < 0.01 :
            alpha *= 2

        Xi = x
        Yi = y

        R = (Xi @ theta.T).values - Yi

        theta = theta - (alpha/len(Xi) * np.sum(R.values * Xi, axis=0)).values
        cost = J_theta(x,y,theta)

    return (theta, cost)



#### DATASET2.txt

data = pd.read_csv('Dataset1.txt', skiprows=(1) , sep=";", names=('height','weight','male'))
# print(data.head)


# Normialization -> Dividing the column by the max value of the column

maxw = data['weight'].max()
maxh = data['height'].max()
data['weight'] = data['weight']/maxw
data['height'] = data['height']/maxh
data['t0'] = 1
# x = height , y = weight

# Features
Y = data.drop(['height','t0','male'],axis=1)

# Labels
X = data.drop(['weight','male'],axis = 1)



# Parameters Initialized :
theta0 = 1.0
theta1 = 1.0

# Numpy array helps with matrix multiplications
theta = np.array([[theta1,theta0]])

result1 = batch_gradient_descent(X,Y,theta,0.001,1000)
print("Theta For Dataset1")
print(result1[0])
print("Cost For Dataset1")
print(result1[1].tolist()[0])

#### DATASET2.txt

data = pd.read_csv('Dataset2.txt', skiprows=(1) , sep=";", names=('height','weight','male'))
# print(data.head)


# Normialization -> Dividing the column by the max value of the column

maxw = data['weight'].max()
maxh = data['height'].max()
minh = data['height'].min()
data['weight'] = data['weight']/maxw
data['height'] = data['height']/maxh
data['t0'] = 1
# x = height , y = weight

# Features
Y = data.drop(['height','t0','male'],axis=1)

# Labels
X = data.drop(['weight','male'],axis = 1)



# Parameters Initialized :
theta0 = 1.0
theta1 = 1.0

# Numpy array helps with matrix multiplications
theta = np.array([[theta1,theta0]])

result2 = batch_gradient_descent(X,Y,theta,0.001,1000)
print("Theta For Dataset2")
print(result2[0].tolist()[0])
print("Cost For Dataset2")
print(result2[1].tolist()[0])

# t = result2[0].tolist()[0]

# l = np.arange(0, maxh, 0.1)

# def y(o): 
#     return t[0] * o + t[1]

# plt.plot(l, y(l).astype(np.float))
# plt.show()




start = -2
x = []
y = []
z = []
for i in  range(40):
    x.append(start + i*0.1)
    y.append(start + i*0.1)
    theta = np.array([[start + i*0.1,start + i*0.1]])
    cost = J_theta(X,Y,theta)
    z.append(cost)





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c='g', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# SIGMOID FUNCTION

x = np.linspace(-10, 10, 100) 
z = 1/(1 + np.exp(-x)) 
  
plt.plot(x, z) 
plt.xlabel("x") 
plt.ylabel("Sigmoid(X)") 
  
plt.show() 