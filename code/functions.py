# ml libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model

def get_best_features(data, n_features):
    """
    Do feature selection.
    """
    val = mutual_info_regression(data.values[:,1:], data.values[:,0])
    mi = pd.Series(val, index=data.columns[1:])
    return mi.sort_values(ascending=False)[:n_features].index

def tt_split(data_country, features, test_size):
    y = data_country['DayAheadPrice']
    x = data_country[features]
    N = data_country.shape[0]
    train = int(N*(1-test_size))
    test = int(N*test_size)
    x_train, x_test, y_train, y_test = x[:train], x[train:], y[:train], y[train:]
    return x_train, x_test, y_train, y_test

def get_errors(test, predictions):
    MBE = np.mean(test - predictions)
    RMSE = np.sqrt(metrics.mean_squared_error(test, predictions))
    cvRMSE = RMSE / np.mean(test)
    NMBE = MBE / np.mean(test)
    R2 = metrics.r2_score(test, predictions)
    labels = [
        'Root mean squared error', 
        'Coefficient of variation RMSE', 
        'Normalized mean bias error',
        'R2 score'
    ]
    return pd.Series([RMSE, cvRMSE, NMBE, R2], index=labels)
def rescale(df):
    """
    Normalization (Z) of the data.
    """
    means = []
    stds = []

    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        
        df[col] = (df[col] - mean) / std
        means.append(mean)
        stds.append(std)
    
    return df, pd.DataFrame({'Mean': means, 'Std': stds}, index=df.columns)

def plot_predictions(test, predictions):
    fig, ax = plt.subplots(1,2, figsize=[14,5])
    ax[0].plot(test, label='data')
    ax[0].plot(predictions, label='predictions')
    ax[1].plot(test[:60], label='data')
    ax[1].plot(predictions[:60], label='predictions')

    ax[0].legend()
    plt.show()
def train_model_country(data_country_input, n_features,model_type="rf",  test_size=0.25, model_parameters=None):
    """
    Train a ML model. This function does the following:
        1. rescales the input data
        2. train test split
        3. feature selection on the training data
        4. train the model: either NN or RF
    
    Parameters:
        model_type: 'rf' | 'nn' | 'lr', str
            type of the model to train, 'rf' for random forest, 'lr' for linear regression and 'nn' for neural network
        data_country: DataFrame
            cleaned data for a given country (no Nans etc)
        n_features: int
            number of best features to select
        model_parameters: dict
            dictionary with model parameters, which are parameters for either 
            MLPRegressor or RandomForestRegressor from sklearn
    
    
    Out:
        model: object
            sklearn model
        params: DataFrame
            parameters used for rescaling the data; predictions have to be scaled back 
            for error calculation and out-of-sample new data must be scaled using these params
        selected_features: list or Index, not sure
            features used in training this model
        err: Series
            errors for predictions
    """
    data_country = data_country_input.copy()
    data_country, params = rescale(data_country)

    x_train, x_test, y_train, y_test = tt_split(data_country, data_country.columns[1:],test_size)
    selected_features = get_best_features(x_train, n_features)

    x_train = x_train[selected_features]
    x_test = x_test[selected_features]

    if model_type.lower() == 'nn':
        if model_parameters == None:
            # these parameters worked okay for project 1
            model_parameters = {
                'hidden_layer_sizes': (16,16,14)
            }
        model = MLPRegressor(**model_parameters)
    elif model_type.lower() == 'lr':
       model = linear_model.LinearRegression() 
    elif model_type.lower() == 'rf':
        if model_parameters == None:
            # these parameters worked okay for project 1
            model_parameters = {
                'bootstrap': True,
                'min_samples_leaf': 2,
                'n_estimators': 10, 
                'min_samples_split': 3,
                'max_features': 10,
                'max_depth': 10,
                'max_leaf_nodes': None}
            
        model = RandomForestRegressor(**model_parameters)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    predictions = pd.Series(predictions, index=y_test.index).sort_index()
    y_test = y_test.sort_index()


    # to calculate errors, we have to rescale the data back using original parameters
    predictions = predictions * params.loc['DayAheadPrice', 'Std'] + params.loc['DayAheadPrice', 'Mean']
    y_test = y_test * params.loc['DayAheadPrice', 'Std'] + params.loc['DayAheadPrice', 'Mean']
    err = get_errors(y_test, predictions)

    #display(err)
    #plot_predictions(y_test, predictions)
    
    return model, params, selected_features, err, predictions
def day_split(data_country, features, day):
    data_country=data_country.dropna()
    y = data_country['DayAheadPrice']
    x = data_country[features]
    x_train, x_test, y_train, y_test = x[x.index<day], x.loc[day], y[y.index<day], y.loc[day]
    return x_train, x_test, y_train, y_test
def train_model_country_day(data_country_input, day, n_features,model_type="rf", model_parameters=None):
    """
    Train a ML model. This function does the following:
        1. rescales the input data
        2. train test split
        3. feature selection on the training data
        4. train the model: either NN or RF
    
    Parameters:
        model_type: 'rf' | 'nn' | 'lr', str
            type of the model to train, 'rf' for random forest, 'lr' for linear regression and 'nn' for neural network
        data_country: DataFrame
            cleaned data for a given country (no Nans etc)
        n_features: int
            number of best features to select
        model_parameters: dict
            dictionary with model parameters, which are parameters for either 
            MLPRegressor or RandomForestRegressor from sklearn
    
    
    Out:
        model: object
            sklearn model
        params: DataFrame
            parameters used for rescaling the data; predictions have to be scaled back 
            for error calculation and out-of-sample new data must be scaled using these params
        selected_features: list or Index, not sure
            features used in training this model
        err: Series
            errors for predictions
    """
    data_country = data_country_input.copy()
    data_country, params = rescale(data_country)
    x_train, x_test, y_train, y_test = day_split(data_country, data_country.columns[1:],day)
    selected_features = get_best_features(x_train, n_features)

    x_train = x_train[selected_features]
    x_test = x_test[selected_features]

    if model_type.lower() == 'nn':
        if model_parameters == None:
            # these parameters worked okay for project 1
            model_parameters = {
                'hidden_layer_sizes': (16,16,14)
            }
        model = MLPRegressor(**model_parameters)
    elif model_type.lower() == 'lr':
       model = linear_model.LinearRegression() 
    elif model_type.lower() == 'rf':
        if model_parameters == None:
            # these parameters worked okay for project 1
            model_parameters = {
                'bootstrap': True,
                'min_samples_leaf': 2,
                'n_estimators': 10, 
                'min_samples_split': 3,
                'max_features': 10,
                'max_depth': 10,
                'max_leaf_nodes': None}
            
        model = RandomForestRegressor(**model_parameters)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    predictions = pd.Series(predictions, index=y_test.index).sort_index()
    y_test = y_test.sort_index()


    # to calculate errors, we have to rescale the data back using original parameters
    predictions = predictions * params.loc['DayAheadPrice', 'Std'] + params.loc['DayAheadPrice', 'Mean']
    y_test = y_test * params.loc['DayAheadPrice', 'Std'] + params.loc['DayAheadPrice', 'Mean']
    err = get_errors(y_test, predictions)

    #display(err)
    #plot_predictions(y_test, predictions)
    
    return model, params, selected_features, err, predictions
