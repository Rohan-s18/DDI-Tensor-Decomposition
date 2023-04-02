"""
Author: Rohan Singh
4/2/2023
Python Module for Data Preprocessing
"""

#  Imports
import pandas as pd
import math
import numpy as np
import pickle


#  Function to build a pickle for the dd-disease tensor
def build_x_tensor():
    dd_disease_data = pd.read_csv('./useful_data/intersection_pairs_dd_disease.csv')
    drug_data = pd.read_csv('./useful_data/intersections.csv')
    all_cell_line_ls = dd_disease_data['cell_line'].tolist()
    cell_line_ls = remove_duplicate(all_cell_line_ls)

    len_cell_line = len(cell_line_ls)
    pair_num = len(dd_disease_data.index)
    drug_num = len(drug_data.index)

    result = {}
    for index, cell_line in enumerate(cell_line_ls):
        result[str(index) + ' ' + cell_line] = build_slice_x('cell_line', cell_line, dd_disease_data, drug_data)

    with open('./useful_data/tensor_x.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('finish building tensor for drug drug disease!, sparsity: {}'.format(pair_num/(drug_num*drug_num*len_cell_line)))


#  Helper function to build a slice matrix
def build_slice_x(slice_col_name, slice_name, tensor_data, drug_data):
    len_drug = len(drug_data.index)
    drug_names = drug_data['name'].tolist()
    result = pd.DataFrame(np.zeros((len_drug, len_drug), dtype=int), columns=drug_names, index=drug_names)
    target = tensor_data[tensor_data[slice_col_name] == slice_name]
    for index, row in target.iterrows():
        drug1 = row['Drug1_name']
        drug2 = row['Drug2_name']
        classification = row['classification']
        if classification == 'synergy':
            result[drug1][drug2] = 1
        else:
            result[drug1][drug2] = -1

    return result


#  Function to build a pickle for the dd intersection tensor
def build_y_tensor():
    ddi_data = pd.read_csv('./useful_data/intersection_pairs_ddi.csv')
    drug_data = pd.read_csv('./useful_data/intersections.csv')
    all_ddi_type_ls = ddi_data['Y'].tolist()
    ddi_type_ls = remove_duplicate(all_ddi_type_ls)

    ddi_type_num = len(ddi_type_ls)
    pair_num = len(ddi_data.index)
    drug_num = len(drug_data.index)

    result = {}
    for index, ddi_type in enumerate(ddi_type_ls):
        result[str(index) + ' ' + str(ddi_type)] = build_slice_y('Y', ddi_type, ddi_data, drug_data)

    with open('./useful_data/tensor_y.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('finish building tensor for DDI!, sparsity: {}'.format(pair_num / (drug_num * drug_num * ddi_type_num)))


#  Helper function to build a slice matrix
def build_slice_y(slice_col_name, slice_name, tensor_data, drug_data):
    len_drug = len(drug_data.index)
    drug_names = drug_data['name'].tolist()
    result = pd.DataFrame(np.zeros((len_drug, len_drug), dtype=int), columns=drug_names, index=drug_names)
    target = tensor_data[tensor_data[slice_col_name] == slice_name]
    for index, row in target.iterrows():
        drug1 = row['Drug1_name']
        drug2 = row['Drug2_name']
        result[drug1][drug2] = 1

    return result


#  Helper Function to remove duplicates
def remove_duplicate(ls):
    seen = set()
    result = []
    for element in ls:
        if element not in seen:
            seen.add(element)
            result.append(element)
    return result


#  Main Function
def main():
    pass

if __name__ == "__main__":
    main()

    