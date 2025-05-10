import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
tf.keras.backend.set_floatx('float64')
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import pickle
import argparse
import os
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
'''
Script to train the ML models used in this work.
Input features can be chemical descriptors ("desc") or Morgan fingerprints ("fing").
Imputation can be Bayesian linear regression ("BLR") or extremely randomized trees ("ERT").
Model can either be random forest ("RF") or ensemble of deep neural networks ("DNN").
Outputs the saved model into the model folder, along with Y_train, Y_test, Y_pred_train, and Y_pred_test as .csv files.
Each .csv file contains columns as gas permeability in the order ['He','H2','O2','N2','CO2','CH4'] and rows as samples.
'''
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def train(args):

    #read in the training data
    DatasetA_Smiles_P = pd.read_csv("datasetA_imputed_all.csv")
    numeric_cols = DatasetA_Smiles_P.select_dtypes(include=[np.number]).columns
    DatasetA_grouped = DatasetA_Smiles_P.groupby('Smiles')[numeric_cols].mean().reset_index()
    Y = DatasetA_grouped.iloc[:,-9:-7]
    #normalize Y
    Y = np.array(Y)
    scaler = StandardScaler()
    Y = scaler.fit_transform(Y)

    X = pd.read_csv('datasets/datasetAX_fing.csv')
    X = np.array(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    if args.model == 'RF':
        model = RandomForestRegressor(n_estimators=200, max_depth = 10, bootstrap = True, max_features = 'sqrt')
        print('Training random forest model...')
        history = model.fit(X_train, Y_train)
        Y_train = scaler.inverse_transform(Y_train)
        Y_test = scaler.inverse_transform(Y_test)

        Y_pred_train = model.predict((X_train))
        Y_pred_train = scaler.inverse_transform(Y_pred_train)
        Y_pred_test = model.predict((X_test))
        Y_pred_test = scaler.inverse_transform(Y_pred_test)
         # 🔽 加入 R² 打印
        r2_train = r2_score(Y_train, Y_pred_train)
        r2_test = r2_score(Y_test, Y_pred_test)

        r2_train = r2_score(Y_train, Y_pred_train, multioutput='raw_values')
        r2_test = r2_score(Y_test, Y_pred_test, multioutput='raw_values')

        print("Random Forest Performance:")
        for i, (r2_tr, r2_te) in enumerate(zip(r2_train, r2_test)):
            mae_train = mean_absolute_error(Y_train[:, i], Y_pred_train[:, i])
            mae_test = mean_absolute_error(Y_test[:, i], Y_pred_test[:, i])
            mse_train = mean_squared_error(Y_train[:, i], Y_pred_train[:, i])
            mse_test = mean_squared_error(Y_test[:, i], Y_pred_test[:, i])
            rmse_train = np.sqrt(mse_train)
            rmse_test = np.sqrt(mse_test)
            print(f"Target {i+1}:")
            print(f"  R²:Train = {r2_tr:.4f}, Test = {r2_te:.4f}")
            print(f"  MAE:Train = {mae_train:.4f}, Test = {mae_test:.4f}")
            print(f"  MSE:Train = {mse_train:.4f}, Test = {mse_test:.4f}")
            print(f"  RMSE:Train = {rmse_train:.4f}, Test = {rmse_test:.4f}")

        maindirectory = os.getcwd() + '/new_models/RF' 
        if  not os.path.exists(maindirectory):
            os.mkdir(maindirectory)
        filename = maindirectory + '/RF' + '.sav'
        pickle.dump(model, open(filename, 'wb'))

        os.chdir(maindirectory)
        np.savetxt('Y_train.csv', Y_train, delimiter=",")
        np.savetxt('Y_test.csv', Y_test, delimiter=",")
        np.savetxt('Y_pred_train.csv', Y_pred_train, delimiter=",")
        np.savetxt('Y_pred_test.csv', Y_pred_test, delimiter=",")

        print('Model saved to ' + filename)

    if args.model == 'SVM':
        print('Training SVM model...')
        # 非指纹输入，使用常规RBF核'''
        print("Using RBF kernel for descriptor features.")
        base_model = SVR(kernel='rbf', C=10, epsilon=0.005 ,gamma='scale')
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, Y_train)

        Y_train = scaler.inverse_transform(Y_train)
        Y_test = scaler.inverse_transform(Y_test)
        Y_pred_train = scaler.inverse_transform(model.predict(X_train))
        Y_pred_test = scaler.inverse_transform(model.predict(X_test))

        # 性能评估
        r2_train = r2_score(Y_train, Y_pred_train, multioutput='raw_values')
        r2_test = r2_score(Y_test, Y_pred_test, multioutput='raw_values')

        print("SVM Performance:")
        for i, (r2_tr, r2_te) in enumerate(zip(r2_train, r2_test)):
            mae_train = mean_absolute_error(Y_train[:, i], Y_pred_train[:, i])
            mae_test = mean_absolute_error(Y_test[:, i], Y_pred_test[:, i])
            mse_train = mean_squared_error(Y_train[:, i], Y_pred_train[:, i])
            mse_test = mean_squared_error(Y_test[:, i], Y_pred_test[:, i])
            rmse_train = np.sqrt(mse_train)
            rmse_test = np.sqrt(mse_test)
            print(f"Target {i+1}:")
            print(f"  R²:Train = {r2_tr:.4f}, Test = {r2_te:.4f}")
            print(f"  MAE:Train = {mae_train:.4f}, Test = {mae_test:.4f}")
            print(f"  MSE:Train = {mse_train:.4f}, Test = {mse_test:.4f}")
            print(f"  RMSE:Train = {rmse_train:.4f}, Test = {rmse_test:.4f}")

        # 保存模型
        maindirectory = os.getcwd() + '/new_models/SVM'
        if not os.path.exists(maindirectory):
            os.mkdir(maindirectory)
        filename = maindirectory + '/SVM'+ '.sav'
        pickle.dump(model, open(filename, 'wb'))

        os.chdir(maindirectory)
        np.savetxt('Y_train.csv', Y_train, delimiter=",")
        np.savetxt('Y_test.csv', Y_test, delimiter=",")
        np.savetxt('Y_pred_train.csv', Y_pred_train, delimiter=",")
        np.savetxt('Y_pred_test.csv', Y_pred_test, delimiter=",")

        print('Model saved to ' + filename)

#for parser arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required = True, 
    	help='choose either "RF" for random forest or "SVM" for the ensemble of deep neural networks for model training')

    parsed_args = parser.parse_args()

    train(parsed_args)