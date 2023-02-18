from numpy import mean, abs, square, sum, asarray, minimum
# Using decision tree regression, here we use validation data to turn the min_samples_split parameter
def mse_loss(y_predict, y):
    return mean(square(y_predict - y))
def mae_loss(y_predict, y):
    return mean(abs(y_predict - y))
def nep_loss(y_predict, y):     # Normalised Error in Assigned Power (NEP)
    return sum(abs(y - y_predict), axis=0) / sum(y, axis=0)
def f1e_loss(y_predict, y):     # F1 on energy 
    pi =  mean(sum(minimum(y_predict , y)) / sum(y_predict, axis=0))     # energy precision
    ri = mean(sum(minimum(y_predict , y)) / sum(y, axis=0))      # energy recall
    return (2 * (pi*ri / (pi + ri) ))

