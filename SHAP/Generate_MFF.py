import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
'''
Script to choose the pertinent molecular descriptors or fingerprints for Dataset A (training) and calculate these chemical features for another dataset base on SMILES strings.
Chemical features are already calculated for Dataset A, so training can begin without this step.
Representations are saved as .csv files with columns as features and rows as samples.
'''

def calculate_representations(args):
    os.chdir(os.getcwd() + '/datasets/')
    DatasetA_Smiles_P = pd.read_csv("datasetA_imputed_all.csv")
    numeric_cols = DatasetA_Smiles_P.select_dtypes(include=[np.number]).columns.tolist()
    DatasetA_grouped = DatasetA_Smiles_P.groupby('Smiles')[numeric_cols].mean().reset_index()
    Dataset = pd.read_csv(args.dataset + '.csv')
    #Dataset = Dataset.groupby('Smiles')[numeric_cols].mean().reset_index()
    
    #dataset-A fingerprint
    molecules = DatasetA_grouped.Smiles.apply(Chem.MolFromSmiles)
    fp = molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
    fp_n = fp.apply(lambda m: m.GetNonzeroElements())

    # using substructures in dataset-A to construct a dictionary
    HashCode = []
    for i in fp_n:
        for j in i.keys():
            HashCode.append(j)
            
    unique_set = set(HashCode)
    unique_list = list(unique_set)

    Corr_df = pd.DataFrame(unique_list).reset_index()
    Corr_df.to_csv('Corr_df.csv', index=False)

    #construct dataset-A input
    MY_finger = []
    for polymer in fp_n:
        my_finger = [0] * len(unique_list)
        for key in polymer.keys():
            index = Corr_df[Corr_df[0] == key]['index'].values[0]
            my_finger[index] = polymer[key]
        MY_finger.append(my_finger)
        
    MY_finger_dataset_A = pd.DataFrame(MY_finger)

    # filter input into the most popular 114 substructures
    Zero_Sum = (MY_finger_dataset_A == 0).astype(int).sum()
    NumberOfZero = 325 #adjust this number based on the tolerance of how many substructures
    print(len(Zero_Sum[Zero_Sum < NumberOfZero]))
    X_fingerprints = MY_finger_dataset_A[Zero_Sum[Zero_Sum < NumberOfZero].index]
    new_length = X_fingerprints.shape[1]

    selected_keys = X_fingerprints.columns
    selected_Corr_df = Corr_df.iloc[selected_keys,-1]
    selected_keys = Corr_df.iloc[selected_keys,-1].to_numpy()

    #get fingerprints for the new dataset
    molecules = Dataset.Smiles.apply(Chem.MolFromSmiles)
    fp = molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
    fp_n = fp.apply(lambda m: m.GetNonzeroElements())

    #construct new dataset input
    MY_finger = []
    for polymer in fp_n:
        my_finger = [0] * new_length
        for key in polymer.keys():
            if key in selected_keys:
                index = np.where(selected_keys == key)[0][0]
                my_finger[index] = polymer[key]
        MY_finger.append(my_finger)
        
    MY_finger_dataset = pd.DataFrame(MY_finger)  
    MY_finger_dataset.columns = selected_Corr_df.index
    filename = args.dataset + '_X_fing.csv'
    MY_finger_dataset.to_csv(filename, index=False)
    print('Features saved to /datasets/'+ filename)

#for parser arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required = True, 
    	help='Specify the filename without csv (datasetB.csv or datasetC.csv) within the dataset folder to compute the features of. This files should have a single column of SMILES Strings labeled with "Smiles"')        

    parsed_args = parser.parse_args()

    calculate_representations(parsed_args)