import time
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from plotting import plot_losses
def fitting(model, X_train, y_train, z_train, model_name, applianceNum, applianceName):
    training_history = pd.DataFrame()
    z_trainShape = np.shape(z_train)
    start = time.time()
    adam = Adam(lr = 1e-3)
    model.compile(loss='mean_squared_error', optimizer=adam)
    checkpointer = ModelCheckpoint(filepath='./lstm_model_{}_{}.hdf5'.format(model_name,applianceName), verbose=0, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-8)
    
    history = model.fit( {'mainPower': X_train[:,:,-2:], 'applianceSignal': z_train.reshape(z_trainShape[0],
                        z_trainShape[1], 1)}, y_train[:,applianceNum],
                    batch_size=256, verbose=1, epochs=30,
                    validation_split=0.33, callbacks=[checkpointer, reduce_lr])
    print('Finish trainning. Time: '+applianceName+' ', time.time() - start)
    column_loss = applianceName+'_'+model_name + '_loss'
    column_validation = applianceName+'_'+model_name + '_val'
    column_time = applianceName+'_'+ model_name + '_time'
    training_history[column_loss] = history.history['loss']
    training_history[column_validation] = history.history['val_loss']
    training_history[column_time] = time.time()-start
    plot_losses(history.history['loss'], history.history['val_loss'], 'test_Val_lstm_model_{}_{}'.format(model_name,applianceName))
    
    return history, training_history