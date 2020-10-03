import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
from math import ceil
from scipy import linalg

iris = load_iris()


#  Load Iris Format To Pandas DataFrame
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= ['pl','pw','sl','sw'] + ['target'])

# Remove Label = 1
data = df[df['target'] != 1]

# Remove Useless Columns For X and Y
X = data.drop(['pl','pw','target'],axis=1)
Y = data.drop(['pl','pw','sl','sw'],axis=1)

# Split Dataset to Train (80%) and Test (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2, random_state=0)

# Sigmoid Used In Logistic Regression
def sigmoid( z):
    return 1 / (1 + np.exp(-z))
# Loss Function
def loss( h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
# Pandas -> Numpy for Easier Use Of Data    
X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()

# intercept = np.ones((X_train.shape[0], 1))
# X_train = np.concatenate((intercept, X_train), axis=1)

theta = np.ones(X_train.shape[1])

# Number Of Iterations (Epochs) for Gradient Descent
num_iter = 10000

# Learning Rate
alpha = 0.001
Z = 0
## Run Gradient Descent: 
for i in range(num_iter):
    z = np.dot(X_train, theta)
    # H: Hypothesis
    h = sigmoid(z)

    z = z.reshape(80,1)

    R = z - Y_train

    ########################
    # BATCH GRADIENT DESCENT STEP
    ########################
    theta = theta - (alpha/len(X_train) * np.sum(R * X_train, axis=0))
    # print(theta)

    Z = z
    ###
    # FOR TEST
    ###
    # if(i % 10 == 0):
    #     z = np.dot(X_train, theta)
    #     h = sigmoid(z)
    #     print('Loss : ')
    #     print(loss(h, Y_train)

h = sigmoid(Z)
print('Final Loss : ')
print(loss(h, Y_train))

# Sigmoid To Predict Probability Of Y BEEING 1
def predict_prob( X,theta):
    return sigmoid(np.dot(X, theta))

# Determine If The Label is 0 or 1 ,Using a Threshold -> I used 0.5 here
def predict(X,theta, threshold=0.58):
    return predict_prob(X,theta) >= threshold



# intercept = np.ones((X_test.shape[0], 1))
# X_test = np.concatenate((intercept, X_test), axis=1)


Y_test = Y_test.to_numpy()

# accuracy
preds = predict(X_test,theta)

# print(preds)
# print(Y_test)

# Number Of Right Predictions
predicted = 0
for i in range( len(preds)):
    if preds[i] == False and Y_test[i] == 0:
        predicted += 1
    if preds[i] == True and Y_test[i] == 2:
        predicted += 1

# Accuracy : (Right Predicted / All) * 100
print("Accuracy : (%)")
print(predicted/len(X_test)*100)
# Theta : Specifies Decision Boundry Parameters
print("Final Theta : ")
print(theta)

xs1 = []
ys1 = []

X = X.to_numpy()
Y = Y.to_numpy()

xs2 = []
ys2 = []
for i in range(len(X)):
    if Y[i][0] == 0:
        xs1.append(X[i][0])
        ys1.append(X[i][1])
    else:
        xs2.append(X[i][0])
        ys2.append(X[i][1])



plt.scatter(xs1,ys1,color='g')

plt.scatter(xs2,ys2,color='r')

# plt.scatter(xs,ys,color='r')
# plt.plot(xs,ys,color='r')

"""Plot a line from slope and intercept"""
def abline(slope, intercept):
    
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


# for simple linear regression
abline(theta[0],theta[1])

# plt.axis('equal')
plt.show()