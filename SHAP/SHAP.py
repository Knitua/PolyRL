import tensorflow as tf 
tf.keras.backend.set_floatx('float64')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
import shap
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
def SHAP():
    modelname = 'RF_BLR_fing(380)'

    maindirectory = os.getcwd() + '/models/' + modelname
    X_df = pd.read_csv(os.getcwd() + '/datasets/high_score_molecules_X_fing.csv')
    os.chdir(maindirectory)

    # 获取 X 数据并标准化（若需要）
    X = np.array(X_df)

    # 加载 RF 模型
    filename = modelname + '.sav'
    model = pickle.load(open(filename, 'rb'))

    # SHAP 值计算
    background = X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(background)

    print("shap_values shape:", np.shape(shap_values))

    # 转置为 (n_outputs, n_samples, n_features)
    shap_values = np.transpose(shap_values, (2, 0, 1))
    Columns = ['N2', 'CO2']  # 根据你的模型输出顺序设定
    feature_ids = list(X_df.columns)

    for i in range(len(shap_values)):
        gas = Columns[i]
        np.savetxt(f'shap_{gas}.csv', shap_values[i], delimiter=",")
        mean_abs_shap = np.mean(np.abs(shap_values[i]), axis=0)
        df = pd.DataFrame({
            'feature_index': np.arange(len(mean_abs_shap)),
            'bit_id': feature_ids,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        df.to_csv(f'Top_features_{gas}.csv', index=False)

    print('SHAP values and feature rankings saved to ' + maindirectory)


if __name__ == '__main__':
    SHAP()
