# Implementing LR, WLR and Logistic LR in Python

## Linear Regression:

This code is implemented for linear regression. We use a batch-gradient-descent with 1000 iterations and learning rate (alpha) of initial value 0.001. 

Outliers effect linear regression since every data in the dataset is used to update the parameters. But a few outliers won’t have much effect on the result. But if you look at the printed data the final theta parameters are not the same and the cost for the dataset with outliers is a little higher.

Normalization is to scale all of the features in a specific range like [0,1] or [-1,1] 
with Normalization features with higher values like house’s square meter will have a larger effect on the model than the number of rooms in the house learning a model on house features.

We use sigmoid for classification, when our result is about probability and the result will be either 0 or 1 and will help us in a binary classification problem. Sigmoid results in 0 for a range of inputs and 1 for another range of numbers.

### Dataset without outliers:


![alt text](https://github.com/sinabr/Linear-Regression-Variations/blob/master/L2.png)


### Dataset with outliers:


![alt text](https://github.com/sinabr/Linear-Regression-Variations/blob/master/L1.png)


## Weighted Linear Regerssion:

![alt text](https://github.com/sinabr/Linear-Regression-Variations/blob/master/W.png)


Dashed Line → Linear Regression
Red Line → Locally weighted Linear Regression

We used Gaussian kernel as a utility function. This function is multiplied to the data and the
result are large numbers for close points and small numbers for distant points. This helps us to
only use closer points to predict each point (fit a line). We can see in the picture that locally
weighted regression is closer to each point of dataset and in comparison, linear regression is not a
good fit for most of the points.

If our data is not linear like the one in this problem, we can not fit a good line to it. Instead by
locally weighted linear regression we can find local lines that are good for that range of values.
Linear regression and WLR are the same if the data is completely linear.



## Logistic Regression:

![alt text](https://github.com/sinabr/Linear-Regression-Variations/blob/master/LogisticRegression.png)

### Train Error:
Considering the Loss Function as Training Error metric the final MEAN value for loss function was :
-0.4538186865342322

We should be careful that in logistic regression we want to maximize the value of cost function.

### Test Error:
In Classification we have ‘Right’ predictions and ‘Wrong’ predictions. By changing the Alpha
(Learning Rate) and Prediction Threshold we get a variety of results. The average accuracy with
parameters set was 90%. Due to randomness of Train_Test_Split function the accuracy could go as low
as 50%.

### Decision Boundary:
By running 1000 – 10000 iterations of Gradient Descent the final Theta parameters are:
Theta_1 : 0.04302966 as Slope
Theta_0 : 0.83331737 as Intercept
the line Equation will be:
Y = 0.043 X + 0.833

### It seems the line is a good boundary for this classification but an SVM could give a much better
boundary for this distribution of data.
