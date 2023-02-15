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

def create_ddi(n1, n2):
    tensor = []
    for i in range (0,n2,1):
        matrix = []
        for j in range (0, n1, 1):
            matrix.append(np.zeros(n1))
        tensor.append(np.array(matrix))
    return np.array(tensor)



#%%
"""
This cell contains the main method
"""

def main():
    print("Hello World!")
    ddi = create_ddi(3,5)
    print(ddi)


if __name__ == "__main__":
    main()

#%%