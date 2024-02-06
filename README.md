# Stock Prediction Model with Streamlit Interface

This repository contains a Python-based stock prediction model with a user-friendly Streamlit interface. The model utilizes historical data obtained from the AlphaVantage API for training and prediction purposes.

## Overview

The stock prediction model employs machine learning algorithms to analyze historical stock data and make predictions about future stock prices. Users can interact with the model through a Streamlit interface, which provides a seamless experience for training the model and obtaining predictions.

## Features

- **Streamlit Interface:** User-friendly interface for training the model and obtaining predictions.
- **AlphaVantage API Integration:** Utilizes the AlphaVantage API to fetch historical stock data for training the model.
- **Machine Learning Algorithms:** Employs various machine learning algorithms to analyze historical data and make predictions.
- **Customization:** Users can customize the parameters and settings of the model to suit their preferences.

## Limitations

- **API Rate Limit:** The AlphaVantage API has a daily limit of 25 requests per day. Exceeding this limit may result in temporary unavailability of historical data.
- **Impact of Stock Splits:** Historical data may include instances of stock splits, leading to extreme dips or spikes in the data. This can affect the accuracy of predictions.
- **Inability to Predict Future Events:** The model cannot predict events that will occur in the future, and therefore, cannot guarantee 100% accuracy in predicting future stock prices.

## Usage

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Obtain an API key from AlphaVantage and replace `{YOUR_API_KEY}` in the code with your actual API key.
4. Run the Streamlit app using `streamlit run StockInterface.py`.
5. Alternatively, run directly from `TimeSeries.py` and input api token and stock abbreviation as a variable.
6. Interact with the Streamlit interface to train the model and obtain predictions.
7. *Optional - UnComment `plt.show` on `line 135` to physically show prediction model, Validation Loss, RME Loss, and Training Loss

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Disclaimer

This stock prediction model is for educational and informational purposes only. It should not be used as financial advice, and the developers do not guarantee any specific outcomes or results. Invest at your own risk.
