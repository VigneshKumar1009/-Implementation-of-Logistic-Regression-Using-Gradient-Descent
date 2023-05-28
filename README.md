# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vigneshkumar V
RegisterNumber: 212220220054
*/
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)

## Output:
1.Array value of x :
![Screenshot (14)](https://github.com/VigneshKumar1009/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113573894/d382ea6b-96a3-46b3-a331-f9782d85be6b)

2.Array value of y :
![Screenshot (15)](https://github.com/VigneshKumar1009/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113573894/d704048c-a006-4797-a27d-589acb3f3eb9)

3.Exam 1 & 2 score graph :
![Screenshot (16)](https://github.com/VigneshKumar1009/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113573894/00fe41ff-cf1e-41b5-a6ad-239f8182ef58)

4.Sigmoid graph :
![Screenshot (17)](https://github.com/VigneshKumar1009/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113573894/63048efe-7484-43f9-a8a1-9a9bf93a63d1)

5.J and grad value with array[0,0,0] :
![Screenshot (18)](https://github.com/VigneshKumar1009/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113573894/ee88f7f2-debd-4656-8e09-436f860b78db)

6.J and grad value with array[-24,0.2,0.2] :
![Screenshot (19)](https://github.com/VigneshKumar1009/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113573894/18338c74-cdac-4276-9691-008e8c1ed834)

7.res.function & res.x value :
![Screenshot (20)](https://github.com/VigneshKumar1009/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113573894/553bd408-11cd-4dbc-b755-bfdbfbc49c8f)

8.Decision Boundary graph :
![Screenshot (21)](https://github.com/VigneshKumar1009/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113573894/b9964b3e-d1ed-4dac-9fd0-2919a6d845a3)

9.probability value :
![Screenshot (22)](https://github.com/VigneshKumar1009/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113573894/816e1a90-ce81-446a-aae6-60898abceca3)


10.Mean prediction value :
![Screenshot (23)](https://github.com/VigneshKumar1009/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113573894/1daadeda-8a2d-45fd-88d1-ef9e9f86879a)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

