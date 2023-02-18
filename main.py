from read_data import prepare_data
from model_design import deep_model
from model_fit import fitting
from plotting import plot_each_app
from app_metrics import mse_loss, mae_loss
from model_predict import predicting
import numpy as np
import pandas as pd
model_set = ['V1','V2','V3','V4','V5','V6', 'Bi']
model_set = ['Trans_Init_Bi']
'''
House 1:  {1: 'mains_1', 2: 'mains_2', 3: 'oven_3', 4: 'oven_4', 5: 'refrigerator_5', 6: 'dishwaser_6', 7: 'kitchen_outlets_7', 8: 'kitchen_outlets_8', 9: 'lighting_9', 10: 'washer_dryer_10', 11: 'microwave_11', 12: 'bathroom_gfi_12', 13: 'electric_heat_13', 14: 'stove_14', 15: 'kitchen_outlets_15', 16: 'kitchen_outlets_16', 17: 'lighting_17', 18: 'lighting_18', 19: 'washer_dryer_19', 20: 'washer_dryer_20'}
'''
appliances =  {1: 'mains_1', 2: 'mains_2', 3: 'oven_3', 4: 'oven_4', 5: 'refrigerator_5', 6: 'dishwaser_6', 7: 'kitchen_outlets_7', 8: 'kitchen_outlets_8', 9: 'lighting_9', 10: 'washer_dryer_10', 11: 'microwave_11', 12: 'bathroom_gfi_12', 13: 'electric_heat_13', 14: 'stove_14', 15: 'kitchen_outlets_15', 16: 'kitchen_outlets_16', 17: 'lighting_17', 18: 'lighting_18', 19: 'washer_dryer_19', 20: 'washer_dryer_20'}
#applianceNum = range()
X_train, y_train, z_train, X_test, y_test, df1_test, dates = prepare_data()

threshold = 0.15 * np.max(y_train,axis=0)
threshold = np.clip(threshold, a_min = 0, a_max = 20) 
applianceNumber = [2, 3, 7, 8]
applianceNumber = [3, 7, 8]
applianceNumber = [7, 8]
for applianceNum in applianceNumber:
    for modelName in model_set:
        model = deep_model(modelName, appliances[applianceNum+3])
        history, training_history = fitting(model, X_train, y_train, z_train[:,:,applianceNum], model_name=modelName, applianceNum=applianceNum, applianceName = appliances[applianceNum+3])
        predicting(X_test, y_test, df1_test, dates, model_name=modelName, applianceNum=applianceNum, applianceName = appliances[applianceNum+3], training_history= training_history, threshold=threshold[applianceNum])



