from keras.models import load_model
from plotting import plot_each_app
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from app_metrics import mae_loss, mse_loss, nep_loss, f1e_loss
from scipy.ndimage.interpolation import shift
import numpy as np
import pandas as pd
def accuracy(pred, ground, threshold, title, new_fname):
    accuracy_value = []
    mcc_value = []
    precision_value = []
    f1_value = []
    recall_value = []
    for i in range(len(title)):
        pr = (pred < threshold) + 0
        gr = (ground < threshold) + 0
        '''
        plt.plot(gr[20000:24000], label='Grand Truth')
        plt.plot(pr[20000:24000], label='Disaggregated')
        plt.ylabel('State')
        plt.xlabel('Sample')
        plt.title(title[i])
        plt.legend()
        plt.savefig('saved_plots/'+'threshold_'+new_fname + str(i))
        plt.close()
        '''

        accuracy_value.append(accuracy_score(gr, pr))
        mcc_value.append(matthews_corrcoef(gr, pr))
        precision_value.append(precision_score(gr, pr))
        f1_value.append(f1_score(gr, pr))
        recall_value.append(recall_score(gr, pr))
    return accuracy_value, mcc_value, precision_value, f1_value, recall_value

def predicting(X_test, y_test, df1_test, dates, model_name, applianceNum, applianceName, training_history,threshold=10):
    filepath='./lstm_model_{}_{}.hdf5'.format(model_name,applianceName)
    model = load_model(filepath)
    appSignal = np.zeros((1,np.shape(X_test)[1], 1))
    predictions = []
    inputShape = np.shape(X_test)
    for i in range(inputShape[0]):
        pred_lstm = model.predict({'mainPower': X_test[i,:,-2:].reshape(1, inputShape[1], 2),'applianceSignal': appSignal}).reshape(-1)
        shift(appSignal[0,:,0], -1, output= appSignal[0,:,0], cval=pred_lstm)
        predictions.append(pred_lstm)
    print(pred_lstm.shape)
    predictions = np.asarray(predictions).reshape((inputShape[0], 1))
    ground_lstm = np.asarray(y_test[:,applianceNum]).reshape((inputShape[0], 1))
    accuracy_value, mcc_value, precision_value, f1_value, recall_value = accuracy(predictions,  ground_lstm, threshold, model_name+' on '+applianceName, './lstm_accuracy_{}_{}.jpg'.format(model_name,applianceName))
    nep_loss_lstm = nep_loss(predictions, ground_lstm)
    f1e_loss_lstm = f1e_loss(predictions, ground_lstm)
    mse_loss_lstm = mse_loss(predictions, ground_lstm)
    mae_loss_lstm = mae_loss(predictions, ground_lstm)
    print('Mean square error on test set: '+applianceName+' ', mse_loss_lstm)
    print('Mean absolute error on the test set: '+applianceName+' ', mae_loss_lstm)
    column_mae = applianceName+'_'+ model_name + '_mae'
    column_mse = applianceName+'_'+ model_name + '_mse'
    column_nep = applianceName+'_'+ model_name + '_nep'
    column_f1e = applianceName+'_'+ model_name + '_f1e'

    column_accuracy = applianceName+'_'+ model_name + '_accuracy'
    column_mcc = applianceName+'_'+ model_name + 'mcc'
    column_precision = applianceName+'_'+ model_name + '_precision'
    column_f1 = applianceName+'_'+ model_name + '_f1'
    column_recall = applianceName+'_'+ model_name + '_recall'
    column_gradTruth = applianceName+'_'+ model_name + '_gradTruth'
    column_predicted = applianceName+'_'+ model_name + '_predicted'
    
    training_history[column_nep] = nep_loss_lstm[0]
    training_history[column_f1e] = f1e_loss_lstm
    training_history[column_mae] = mae_loss_lstm
    training_history[column_mse] = mse_loss_lstm
    training_history[column_accuracy] = accuracy_value[0]
    training_history[column_mcc] = mcc_value[0]
    training_history[column_precision] = precision_value[0]
    training_history[column_f1] = f1_value[0]
    training_history[column_recall] = recall_value[0]
    testing_history = pd.DataFrame()
    testing_history[column_gradTruth] = ground_lstm.flatten()
    testing_history[column_predicted] = predictions.flatten()
    training_history.to_csv('Metrix_'+model_name+'_'+applianceName+'.csv')
    testing_history.to_csv('Metrix_testin_'+applianceName+'_'+model_name+'.csv')
    plot_each_app(df1_test, dates[1][20:], predictions, ground_lstm, 
                'Real and predict power on 6 test day of house 1_ {}_{}'.format(model_name,applianceName), look_back = 50)