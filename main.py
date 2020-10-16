"""
This program is a challenge I made for myself before I learned how to properly
build a multi layer perceptron. It was to figure out on my own how to build a
multi layer perceptron from a single layer perceptron only knowing what a hidden
layer does and not how to code one.
"""

#Importing the numpy library
import numpy as np

#enter input
input_features = np.array([[1,0,0,1],[1,0,0,0],[0,0,1,1],[0,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,0,1],[0,0,1,0,]])

#establish target output
target_output = np.array([[1,1,0,0,1,1,0,0]])
target_output = target_output.reshape(8,1)

#Initalizing weights, not randomized
hweights = np.array([[0.1,0.2,0.3],[0.1,0.2,0.3],[0.1,0.2,0.3],[0.1,0.2,0.3]])
oweights = np.array([[0.1],[0.2],[0.3]])

#initalizing bias and learning rate
bias = 0.3
lr = 0.05

#sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

#The derivitive of sigmoid
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

#training
for epoch in range(10000):
    inputs = input_features
    
    #----Feedforward----
    #Find the weights that go to each individual hidden node
    for i in inputs:
        h1w = []
        h1w = np.append(h1w, hweights[i,0])
    for i in inputs:
        h2w = []
        h2w = np.append(h2w, hweights[i,1])
    for i in inputs:
        h3w = []
        h3w = np.append(h3w, hweights[i,2])

    #Feedforward Hidden Layer input
    in_h1 = np.dot(inputs, h1w)
    in_h2 = np.dot(inputs, h2w)
    in_h3 = np.dot(inputs, h3w)
    
    #Feedforward Hidden Layer output
    out_h1 = sigmoid(in_h1)
    out_h2 = sigmoid(in_h2)
    out_h3 = sigmoid(in_h3)
    out_h = np.array([out_h1,out_h2,out_h3])
    
    #Feedforward Output Layer input
    #We need the transpose of the Hidden layers Output
    out_h = out_h.T
    in_o = np.dot(out_h, oweights) + bias
    
    #Feedforward Output Layer output
    out_o = sigmoid(in_o)
    
    #----Backpropogation----
    #Calculating error
    error = out_o - target_output
    
    #Going with the formula
    x = error.sum()
    #print(x)
    
    #Calculating derivative
    derror_douto = error
    douto_dino = sigmoid_der(out_o)
    
    #Multiplying individual derivatives
    deriv = derror_douto * douto_dino
    
    #--------------------
    #Time to update the Hidden layers weights
    #Multiplying with the 3rd individual derivative
    #Finding the transpose of input_features
    inputs = input_features.T
    deriv_final = np.dot(inputs,deriv)
    
    #Updating the Hidden weights values
    hweights = hweights - lr * deriv_final
    
    #--------------------
    #Time to update output layer weights
    #Multiplying with the 3rd individual derivative again
    #Finding the transpose of the hidden layers output
    out_h = out_h.T
    oderiv_final = np.dot(out_h,deriv)

    #Updating the output weights values
    oweights = oweights - lr * oderiv_final
    
    #Updating the bias weight value
    for i in deriv:
        bias -= lr * i

#Check final values
print('--------------------')
print("Hidden Layer Weights:\n",hweights)
print('--------------------')
print("Output Weights:\n",oweights)
print('--------------------')
print("Final Bias:\n",bias)
print('--------------------')
print("Final Error:\n",x)





