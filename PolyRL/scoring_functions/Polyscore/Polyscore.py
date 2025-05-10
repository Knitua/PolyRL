import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
from rdkit import Chem
from .fingerprint_utils import extract_morgan_fingerprints  # 需支持传入 smiles 列表
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

def RF_BLR_fing(smiles: list) -> np.ndarray:

    # 获取当前脚本所在路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(current_dir, 'dataset','datasetA_imputed_all.csv')
    print(train_path)
    # Step 1: 加载训练集并构建 Yscaler
    train_df = pd.read_csv(train_path)
    # 方法1：只对数值列聚合
    train_grouped = train_df.groupby("Smiles")[train_df.select_dtypes(include='number').columns].mean().reset_index()
    Y = train_grouped.iloc[:, -9:-7].to_numpy()
    Yscaler = StandardScaler().fit(Y)

    # 筛选合法 SMILES
    valid_smiles = []
    valid_indices = []
    for i, smi in enumerate(smiles):
        if Chem.MolFromSmiles(smi):
            valid_smiles.append(smi)
            valid_indices.append(i)

    if len(valid_smiles) == 0:
        return ([0.0] * len(smiles), [0.0] * len(smiles), [0.0] * len(smiles))

    # 提取指纹
    X_pred = extract_morgan_fingerprints(valid_smiles)
    X_pred.columns = X_pred.columns.astype(str)

    modelname = 'RF_BLR_fing(380)'
    model_path = os.path.join(current_dir, 'predict_models',f'{modelname}.sav')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    Y_pred = model.predict(X_pred)
    Y_pred_rescaled = Yscaler.inverse_transform(Y_pred)

    # 获取 CO₂ 的 log10 渗透率
    logP_CO2 = Y_pred_rescaled[:, -1]  # 倒数第1列是 CO₂
    # 计算 CO₂/N₂ 选择性的 log10 值
    logS_CO2_N2 = logP_CO2 - Y_pred_rescaled[:, -2]  # 使用 N₂ 渗透率来计算选择性

    # Robeson 上界的参数（2008）
    a = 2.595
    b = 0.3464
    # Robeson 上界的参数（2019）
    #a = 2.604
    #b = 0.2933
    # Robeson 上界的参数（2008）
    a = 2.595
    b = 0.3464
    # 计算偏离 Robeson 上界的 score
    score = logS_CO2_N2 - (a - b * logP_CO2) +2
    score = score.astype(np.float32) 
    # 放回原顺序，非法位置赋值 0
    final_scores = [0.0] * len(smiles)
    logP_CO2_final = [0.0] * len(smiles)
    logS_CO2_N2_final = [0.0] * len(smiles)

    for idx, s, p, selectivity in zip(valid_indices, score, logP_CO2, logS_CO2_N2):
        final_scores[idx] = s
        logP_CO2_final[idx] = p
        logS_CO2_N2_final[idx] = selectivity

    # 返回分数、CO₂ 渗透率、选择性
    return final_scores, logP_CO2_final, logS_CO2_N2_final
