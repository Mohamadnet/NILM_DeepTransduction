import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import pyplot
# Using decision tree regression, here we use validation data to turn the min_samples_split parameter
def mse_loss(y_predict, y):
    return np.mean(np.square(y_predict - y)) 
def mae_loss(y_predict, y):
    return np.mean(np.abs(y_predict - y)) 
# Plot real and predict refrigerator consumption on six days of test data
def plot_each_app(df, dates, predict, y_test, title, look_back = 0):
    num_date = len(dates)
    fig, axes = plt.subplots(num_date,1,figsize=(24, num_date*5) )
    plt.suptitle(title, fontsize = '25')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    for i in range(num_date):
        if i == 0: l = 0
        ind = df[dates[i]].index[look_back:]
        axes.flat[i].plot(ind, y_test[l:l+len(ind)], color = 'blue', alpha = 0.6, label = 'True value')
        axes.flat[i].plot(ind, predict[l:l+len(ind)], color = 'red', alpha = 0.6, label = 'Predicted value')
        axes.flat[i].legend()
        l = len(ind)
    plt.savefig(fname=title)
    plt.close()


################ Reading data plotting 
# Plot 2 first day data of house 1 and 2
def plot_df(df, title):
    apps = df.columns.values
    num_apps = len(apps) 
    fig, axes = plt.subplots((num_apps+1)//2,2, figsize=(24, num_apps*2) )
    for i, key in enumerate(apps):
        axes.flat[i].plot(df[key], alpha = 0.6)
        axes.flat[i].set_title(key, fontsize = '15')
    plt.suptitle(title, fontsize = '30')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig(fname=title)
    plt.close()
################################

# Plot total energy sonsumption of each appliance from two houses
def plot_energy(df):
    fig, axes = plt.subplots(1,2,figsize=(24, 10))
    plt.suptitle('Total enery consumption of each appliance', fontsize = 30)
    cons1 = df[1][df[1].columns.values[2:]].sum().sort_values(ascending=False)
    app1 = cons1.index
    y_pos1 = np.arange(len(app1))
    axes[0].bar(y_pos1, cons1.values,  alpha=0.6) 
    plt.sca(axes[0])
    plt.xticks(y_pos1, app1, rotation = 45)
    plt.savefig(fname='House 1 Energy')
    plt.title('House 1')

    cons2 = df[2][df[2].columns.values[2:]].sum().sort_values(ascending=False)
    app2 = cons2.index
    y_pos2 = np.arange(len(app2))
    axes[1].bar(y_pos2, cons2.values, alpha=0.6)
    plt.sca(axes[1])
    plt.xticks(y_pos2, app2, rotation = 45)
    plt.title('House 2')
    print(df[1].columns.values[2])
    print(df[1][df[1].columns.values[2]])
    plt.savefig(fname='House 2 Energy')
    plt.close()


def plot_losses(train_loss, val_loss, fname):
    plt.rcParams["figure.figsize"] = [24,10]
    plt.title('Mean squared error of train and val set on house 1')
    plt.plot( range(len(train_loss)), train_loss, color = 'b', alpha = 0.6, label='train_loss' )
    plt.plot( range(len( val_loss )), val_loss, color = 'r', alpha = 0.6, label='val_loss' )
    plt.xlabel( 'epoch' )
    plt.ylabel( 'loss' )
    plt.legend()
    plt.savefig(fname=fname)
    plt.close()