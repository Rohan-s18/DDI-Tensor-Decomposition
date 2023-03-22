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
def get_ddi_names(filepath):
    df = pd.read_csv(filepath)
    names = df["name"]
    return names.to_numpy()


#  Helper function to get the drug names from the kinome dataset
def get_kinome_names(filepath):
    df = pd.read_csv(filepath)
    names = df["Name"]
    return names.to_numpy()


#  helper function to get the intersection between the kinome names and the ddi names for drugs
def get_intersection_drugs(db1, db2):
    interesctions = []
    for i in range(0, len(db1),1):
        drug = db1[i]
        for j in range(0,len(db2),1):
            if(drug==db2[j]):
                interesctions.append(drug)
    return np.array(interesctions)



#  Main method
def main():
    
    ddi = get_ddi_names("/Users/rohansingh/github_repos/DDI-Tensor-Decomposition/Drug Similarity/growth inhibition/base_dictionary.csv")
    kinome = get_kinome_names("/Users/rohansingh/github_repos/DDI-Tensor-Decomposition/Drug Similarity/growth inhibition/IC50.csv")

    intersection = get_intersection_drugs(ddi,kinome)

    print("The intersection of the two drug banks is: ",len(intersection))



if __name__ == "__main__":
    main()

