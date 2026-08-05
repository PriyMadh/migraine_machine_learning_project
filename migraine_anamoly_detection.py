import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
df_train = pd.read_csv('migraine_anomoly_train.csv')
df_cv = pd.read_csv('migraine_anomoly_cv.csv')

#  Define order of feature columns
feature_names = [
    'avg_temp_during_day',
    'water_ounces', 
    'sun_minutes', 
    
]

# get X matrix and y vector
X_train = df_train[feature_names].to_numpy()
X_cv = df_cv[feature_names].to_numpy()
y_cv = df_cv['target_migraine'].to_numpy()

# plot the histograms of each feature vector from training set to make sure that its similar to a gaussian curve (not skewed too much to one side)
for j in range(X_train.shape[1]):
    plt.hist((X_train[:, j]))
    plt.title(f"Histogram of {feature_names[j]}")
    plt.show()


def estimate_gaussian(X):
    m, n  = X.shape

    mu = np.zeros(n) # mean for each feature
    for j in range(n):
        mu[j] = np.mean(X[:, j])

    var = 0 # variance # could just use np.var(X, axis = 0) to find variance down each column but this way just shows the math
    for i in range(m):
        var_i = (X[i] - mu) ** 2 # vector subraction --> porduces vecotr
        var += var_i # vector addition --> produces vector
    var = var/m

    return mu, var # vector (n elements)

# find the best epsilon by comparing the actual y val with the models predictions for the cv set data
def find_threshold(acc_y_val, cv_pred):
    best_epsilon = 0
    best_F1 = 0 #best score
    F1 = 0

    step_size = (max(cv_pred) - min(cv_pred)) / 1000

    for epsilon in np.arange(min(cv_pred), max(cv_pred), step_size):
        predictions = (cv_pred < epsilon)
        tp = np.sum((predictions == 1) & (acc_y_val == 1)) # true positive
        fp = np.sum((predictions == 1) & (acc_y_val == 0)) #fasle positive
        fn = np.sum((predictions == 0) & (acc_y_val == 1)) # false negative


        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        

        if (prec + rec) > 0:
            F1 = (2 * prec * rec) / (prec + rec)
        else:
            F1 = 0

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon, best_F1



def finding_probability(X, mu, var):
    
    """"" 
    This is the math behind the hood:

    p = np.zeros(m)
    for i in range(m):          # Loop through each day
        p_i = 1
        for j in range(n):      # Loop through each feature
            p_ij = (1 / np.sqrt(2 * np.pi * var[j])) * np.exp(-((X[i, j] - mu[j])**2) / (2 * var[j]))
            p_i *= p_ij
        p[i] = p_i
    """""
    
    # vecotrized
    prob_matrix = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((X - mu) ** 2) / (2 * var)) # 1. multiplies pi by every element in vector var, 2. X - mu ( X is (m,n) array, mu is (n, 1) array, numpy lines up n with n and subtracts each element j of mu (the means for the column of features) from each feature j in that row of X[i] for every training example), 3. takes the vector from the first part with shape (n,) and matrix from seocnd part with shape (m, n) and lines up the n's and then multiplies each element from the vector witht each element in the respective columns of the matrix)
    prob_vector = np.prod(prob_matrix, axis=1) # multiplies across each row and creates vertical vector of shape (m,) # this is the probability for each example in the cross validation set

    return prob_vector

def run_anomoly_detection(X_1, X_2, y):
    mu, var  = estimate_gaussian(X_1)
    cv_pred= finding_probability(X_2, mu, var)
    best_epsilon, best_F1 = find_threshold(y, cv_pred)
    model_predictions = np.zeros(X_2.shape[0])
    for i in range (X_2.shape[0]):
        if cv_pred[i] < best_epsilon:
            model_predictions[i] = 1
        else:
            model_predictions[i] = 0
        print(f"this is the model prediction for cross validation example {i}: {model_predictions[i]}, and this is the actual value for that example: {y[i]} ")
        print(f"/n and this was the probability: {cv_pred}")
    print(f"/n score: {best_F1}")
run_anomoly_detection(X_train, X_cv, y_cv)






