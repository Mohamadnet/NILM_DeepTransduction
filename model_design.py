from keras.layers.merge import concatenate
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout, LSTM, Input
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras import optimizers
from keras import callbacks
from keras import regularizers
def deep_model(*args):
    output_number = 1
    applianceName = args[1]
    if args[0] == 'Trans_Init_Bi':
        print('LSTM Trans Bi : ')
        model_name = 'LSTM Trans Bi'
        inputA = Input(shape=(50, 2), name='mainPower')         # Main power consumption
        inputC = Input(shape=(50, 1), name='applianceSignal')         # Appliance signal


 
        convA1 = Conv1D(16, 4, activation="relu", padding="same", strides=1)(inputA)
        extractA = Bidirectional(LSTM(256, return_sequences=True, stateful=False), merge_mode='concat')(convA1)
        outA = Bidirectional(LSTM(256, return_sequences=False, stateful=False), merge_mode='concat')(extractA)
        #xA = Model(inputs=inputA, outputs=outA)

        convC1 = Conv1D(16, 4, activation="relu", padding="same", strides=1)(inputC)
        extractC = LSTM(256, return_sequences=True, stateful=False)(convC1)
        outC = LSTM(256, return_sequences=False, stateful=False)(extractC)
        #xC = Model(inputs=inputC, outputs=outC)

        # combine the output of the two branches
        merge = concatenate([outA, outC])

        # output
        interp11 = Dense(256, activation='relu')(merge)
        interp12 = Dense(128, activation='relu')(interp11)
        interp13 = Dense(64, activation='relu')(interp12)
        # output
        output = Dense(output_number, activation='linear')(interp13)
        model = Model(inputs=[inputA, inputC], outputs=output)
        model.summary()
    plot_model(model, to_file=model_name+'.png', show_shapes=True)
    return model
d = deep_model('Trans_Init_Bi','d')