from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os

def extract_morgan_fingerprints(smiles_list: list) -> pd.DataFrame:

    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasetA_file = os.path.join(current_dir,'dataset','datasetA_imputed_all.csv')
    radius = 3
    zero_threshold = 325  

    DatasetA = pd.read_csv(datasetA_file)
    DatasetA_grouped = DatasetA.groupby('Smiles').mean(numeric_only=True).reset_index()
    molecules = DatasetA_grouped['Smiles'].apply(Chem.MolFromSmiles)
    fp = molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=radius))
    fp_n = fp.apply(lambda m: m.GetNonzeroElements())

    HashCode = []
    for i in fp_n:
        for j in i.keys():
            HashCode.append(j)
    unique_list = list(set(HashCode))
    Corr_df = pd.DataFrame(unique_list).reset_index()

    MY_finger = []
    for polymer in fp_n:
        my_finger = [0] * len(unique_list)
        for key in polymer.keys():
            index = Corr_df[Corr_df[0] == key]['index'].values[0]
            my_finger[index] = polymer[key]
        MY_finger.append(my_finger)
    MY_finger_dataset_A = pd.DataFrame(MY_finger)

    Zero_Sum = (MY_finger_dataset_A == 0).astype(int).sum()
    X_fingerprints = MY_finger_dataset_A[Zero_Sum[Zero_Sum < zero_threshold].index]
    new_length = X_fingerprints.shape[1]

    selected_keys = X_fingerprints.columns
    selected_Corr_df = Corr_df.iloc[selected_keys, -1]
    selected_keys = Corr_df.iloc[selected_keys, -1].to_numpy()

    molecules = pd.Series(smiles_list).apply(Chem.MolFromSmiles)
    fp = molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=radius))
    fp_n = fp.apply(lambda m: m.GetNonzeroElements())

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

    #MY_finger_dataset.to_csv('test.csv', index=False)
    return MY_finger_dataset
