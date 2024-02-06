import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def stockData():
    AllInfoDict = {"Dates":[], "OpenPrice":[], "ClosePrice": []}
    api = input("Enter API key for Alpha Vantage: ")
    abb = input("Enter stock abbreviation: ").upper()
    date = input("Please enter a prediciton date in this format | yyyy-mm-dd :")
    #UGD5N6SQGQKLBNUE
    if not api:
        print("Your API key is invalid.")
        return
    if not abb:
        print("Your stock abbreviation is invalid.")

    baseUrl = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&datatype=json&symbol={abb}&apikey={api}'
    try:
        response = requests.get(baseUrl)
        data = response.json()
        if 'Error' in data:
            raise ValueError
        historical_data = {prediction_date: info for prediction_date, info in data["Monthly Adjusted Time Series"].items() if prediction_date <= date}

        if date not in historical_data:
            most_recent_date = max(data["Monthly Adjusted Time Series"].keys())
            most_recent_data = data["Monthly Adjusted Time Series"][most_recent_date]
            predicted_close_price = float(most_recent_data["4. close"])

        else:
            AllInfoDict["Dates"] += historical_data.keys() #Adds all dates in Alpha Vantage API to AllInfoDict["Dates"]
            for MonthData in historical_data.values(): #Loops through nested dict with data for each month
                AllInfoDict["OpenPrice"] += [float(MonthData["1. open"])] #Saves open price to AllInfoDict
                AllInfoDict["ClosePrice"] += [float(MonthData["4. close"])] #Saves close price to AllInfoDict
        try:
            stockDataFrame = pd.DataFrame(AllInfoDict) #Creates 2D array with columns of AllInfoDict Keys [Date, OpenPrice, ClosePrice]
            openCloseValues = stockDataFrame[['OpenPrice', 'ClosePrice']].values #Groups each Open Price with corresponding Closing Price in a list

            stockDataFrame["NextClosePrice"] = stockDataFrame['ClosePrice'].shift(-1) #Creates column in stockDataFrame that consists of next/expected closing price
            nextClosePrice = stockDataFrame["NextClosePrice"].values

            # Before splitting, remove rows with NaN values in both X and Y
            not_nan_indices = ~np.isnan(nextClosePrice)
            openCloseValues = openCloseValues[not_nan_indices]
            nextClosePrice = nextClosePrice[not_nan_indices]

            openCloseValues_train, openCloseValues_test, nextClosePrice_train, nextClosePrice_test = train_test_split(openCloseValues, nextClosePrice, test_size=0.2, random_state = 42) #Seperates testing and training data (tests with 20% and trains with 80%)

            stockDataModel = LinearRegression().fit(openCloseValues_train, nextClosePrice_train) #creates base linear model and fits linear model to training data
            #X - dates, Y - NextClosing Price

            latest_data = stockDataFrame.iloc[-1]
            prediction_features = [latest_data['OpenPrice'], latest_data['ClosePrice']]
            if not predicted_close_price:
                predicted_close_price = stockDataModel.predict([prediction_features])[0]
        
            print(f"On {date} the predicted closing price of {abb.upper()} will be {predicted_close_price}.")

            nextPricePrediction = stockDataModel.predict(openCloseValues_test)
            rmse = mean_squared_error(nextClosePrice_test, nextPricePrediction, squared=False)  #These three lines test the accuracy of the predictions
            print(f"Root Mean Squared Error: {rmse}") #Tests actual results
        except:
            print(f"On {date} the predicted closing price of {abb.upper()} will be {predicted_close_price}.")
    except:
        print("Sorry you have an invalid input.")
        return
stockData()