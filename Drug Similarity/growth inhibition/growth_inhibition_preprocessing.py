"""
Author: Rohan Singh
Data Preprocessing script
3/21/2023
"""

#  Imports 
import pandas as pd
import numpy as np
from pubchempy import *
import random


#  Helper function to get the drug names from the ddi-tensor
def get_ddi_names(filepath):
    df = pd.read_csv(filepath)
    names = df["name"]
    return names.to_numpy()


#  Helper function to get the drug cid from the ddi-tensor
def get_ddi_cids(filepath):
    df = pd.read_csv(filepath)
    cid = df["cid"]
    return cid.to_numpy()


#  Helper function to get the drug names from the kinome dataset
def get_kinome_names(filepath):
    df = pd.read_csv(filepath)
    names = df["PANEL_NAME"]
    return names.to_numpy()


#  helper function to get the intersection between the kinome names and the ddi names for drugs
def get_intersection_drugs(db1, db2):
    db1_set  = set(db1)
    db2_set = set(db2)
    intersections = db1_set.intersection(db2_set)
    return np.array(intersections)


#  Helper function to get the properties for the ddi drugs to get
def temp(db):
    drug = db[0]
    c = Compound.from_cid(int(drug))
    for r in c.record:
        print(r)

    print("\n")


#  Helper function to map properties for similarity matrix construction
def map_props(db):
    drug = db[0]
    c = Compound.from_cid(int(drug))
    
    props = c.record["props"]
    print(props)

#  Helper function to map the dictionary properties
def get_dictionary(db):
    drug = db[0]
    c = Compound.from_cid(int(drug))
    
    print(c.to_dict(properties=["atoms","bonds"]))
    


#  Main method
def main():
    
    ddi = get_ddi_names("/Users/rohansingh/github_repos/DDI-Tensor-Decomposition/Drug Similarity/growth inhibition/base_dictionary.csv")
    kinome = get_kinome_names("/Users/rohansingh/github_repos/DDI-Tensor-Decomposition/Drug Similarity/growth inhibition/IC50.csv")

    intersection = get_intersection_drugs(ddi,kinome)

    #print(kinome)

    print("The intersection of the two drug banks is: ",intersection)

    #Checking the cid properties
    cid = get_ddi_cids("/Users/rohansingh/github_repos/DDI-Tensor-Decomposition/Drug Similarity/growth inhibition/base_dictionary.csv")

    temp(cid)

    get_dictionary(cid)


if __name__ == "__main__":
    main()

