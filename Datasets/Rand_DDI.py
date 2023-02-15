"""
Author: Rohan Singh
Feb 15, 2022
This python module contains source code to create a randomized Drug-Drug-Interaction Tensor
"""

# Imports
import pandas as pd
import numpy as np
import plotly.express as px
import random as rand

#%%
"""
This cell contains the code to create an n1xn1xn2 dimension tensor for the DDI
"""

# This method will create an empty tensor of dimensions n2xn1xn1
def create_tensor(n1, n2):
    tensor = []
    #Creating n2 number of n1xn1 matrices
    for i in range (0,n2,1):
        matrix = []

        #Creating a matrix
        for j in range (0, n1, 1):
            matrix.append(np.zeros(n1))

        #Adding the matrix to the tensor
        tensor.append(np.array(matrix))

    return np.array(tensor)

# This method will add interactions to the tensor
def add_interactions(tensor, m, seed):
    #Getting the max indices
    z_max = tensor.shape[0]
    y_max = tensor.shape[1]
    x_max = tensor.shape[2]

    #Creating the randomised seeds
    rand.seed(seed)

    num = 0
    #Adding '1' (interaction exists) to the tensor
    while(num < m):
        #Getting indices
        x = int(rand.random()*x_max)
        y = int(rand.random()*y_max)
        z = int(rand.random()*z_max)

        #Checking if the interaction exists
        if(tensor[z][y][x] == 1):
            continue
        else:
            tensor[z][y][x] = 1
            num += 1


    return tensor

# This method will make the ddi data set
def make_ddi_dataset(n1, n2, seed, num_interaction):
        ddi = create_tensor(n1,n2)
        ddi = add_interactions(ddi,num_interaction,seed)
        return ddi


#%%
"""
This cell contains the main method
"""

def main():
    print("Hello World!")
    ddi = make_ddi_dataset(3,5,73,15)
    print(ddi)

if __name__ == "__main__":
    main()

#%%