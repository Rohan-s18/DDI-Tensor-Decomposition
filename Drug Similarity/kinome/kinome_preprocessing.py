"""
Author: Rohan Singh
Data Preprocessing script
3/21/2023
"""

#  Imports 
import pandas as pd
import numpy as np
import random


#  Helper function to get the drug names from the ddi-tensor
def get_ddi_names():
    pass


#  Helper function to get the drug names from the kinome dataset
def get_kinome_names():
    pass


#  helper function to get the intersection between the kinome names and the ddi names for drugs
def get_intersection_drugs(db1, db2):
    pass



#  Main method
def main():
    
    ddi = get_ddi_names()
    kinome = get_kinome_names()

    intersection = get_intersection_drugs(ddi,kinome)

    print("The intersection of the two drug banks is: ",len(intersection))



if __name__ == "__main__":
    main()

