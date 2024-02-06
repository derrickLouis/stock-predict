import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model

import sys

def runModel(abb):
    closeDict = {"Dates":[], "ClosePrice": []}
    api = input("Enter API key for Alpha Vantage: ")
    if not api:
        print("Your API key is invalid.")
        return
    if not abb:
        print("Your stock abbreviation is invalid.")
    if not os.path.isfile(f"{abb}_File"): #Checks for json file
        baseUrl = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&datatype=json&symbol={abb}&apikey={api}'
        response = requests.get(baseUrl)
        data = response.json()
        if 'Error' in data:
            raise ValueError
        with open(f'{abb}_File.json', 'w') as outfile: #Saves stock data in json file to prevent overuse of api key
            json.dump(data, outfile)
    else:
        sys.path.insert(1, f'Ai_Project\{abb}_File') 
        filer = f"{abb}_File"
        import filer as f
        data = f.json() #uses Json file if available

    #Dictionary with dates and closing prices
    closeDict["Dates"] = list(data["Weekly Adjusted Time Series"].keys())
    for value in data["Weekly Adjusted Time Series"].values():
        closeDict["ClosePrice"] += [float(value["4. close"])]
        
    #Previous Dictionary into a dataframe(table)
    df = pd.DataFrame(closeDict)
    df.index = pd.to_datetime(df["Dates"], format="%Y-%m-%d") #Makes DataFrame input the Dates

    close = df['ClosePrice']

    #converts the DataFrame into input-output pairs suitable for training model
    def df_to_X_y(df, window_size=5):
        df_as_np = df.to_numpy() #DataFrame to numpy array
        X = []
        y = []
        for i in range(len(df_as_np)-window_size):
            row = [[a] for a in df_as_np[i:i+5]] #3d array: [[[1], [2], [3]], [[4], [5], [6]]]
            X.append(row)
            label = df_as_np[i+5] #closing price is 5 in front of i
            y.append(label)
        return np.array(X), np.array(y)

    windSize = 5
    X, y = df_to_X_y(close, windSize)
    #print(X.shape, y.shape) - Used to check proper shape of X, y

    try:
    #Expected number of results from api
        X_train, y_train = X[:900], y[:900]
        X_val, y_val = X[900:1000], y[900:1000]
        X_test, y_test = X[900:], y[900:]
        print(X_train.shape, y_train.shape, X_val.shape, y.shape, X_test.shape, y_test.shape)
    except:
    #Train 80%, Test 20% - Fall Through
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    #Model Layers
    model = Sequential()
    model.add(Input(shape=(5, 1))) #Expects input data w/ shape (5, 1)
    model.add(LSTM(100)) #Long Short Term Memory Layer using 64 neurons (Expected sequences w/ length of 5) - Change depending on stock complexity
    model.add(Dense(8, activation='relu')) #Adds a layer to the model that enforces  8 neurons (Intoduces non-linearity)
    model.add(Dense(1, activation='linear'))
    model.summary()

    cp = ModelCheckpoint('model.keras', save_best_only=True) #Saves best model when running epochs

    #slows down learning rate, shows loss and rme to test accuracy
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0009), metrics=[RootMeanSquaredError()])
    myModel = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 50, callbacks=[cp]) #epoch passes through training dataset
    model = load_model('model.keras')

    #Val Loss
    plt.figure("Validation Loss")
    plt.plot(myModel.history['val_loss'], color='orange')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.legend(['Validation Loss'], loc='upper right')

    #RME Loss - plot rme to decide how many epochs are necessary for the model to converge
    plt.figure("RME Loss")
    plt.plot(myModel.history['root_mean_squared_error'], color='purple')
    plt.title('RME Loss')
    plt.xlabel('Epoch')
    plt.ylabel('RME Loss')
    plt.legend(['RME Loss'], loc='upper right')

    #Training Loss
    plt.figure("Training Loss")
    plt.plot(myModel.history['loss'], color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.legend(['Train Loss'], loc='upper right')
    

    #Predicitons / Plot **

    #Actual Predictions
    train_prediction = model.predict(X_train).flatten()

    #Plot
    plt.figure("Stock Pred")
    close.plot(color='red') #Original

    #Prediction Plot
    plt.plot(close.index[windSize:len(train_prediction) + windSize], train_prediction, label='Train Predictions', color='green')
    plt.scatter(close.index[899], train_prediction[899])
    plt.legend(["Actual Values","Prediction"])
    #plt.show() UNCOMMENT TO SHOW PLOTS

    #Compares Predicitons to Actual  Results
    train_results = pd.DataFrame(data={'Train Predictions': train_prediction, 'Actuals': y_train})
    print(f"Training Results in DataFrame:\n {train_results}")
