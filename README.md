# CryptoCurrencyTrader
A machine learning program in python to generate cryptocurrency trading strategies using machine learning.
The script is inspired by both the pytrader project https://github.com/owocki/pytrader, and the auto-sklearn project https://automl.github.io/auto-sklearn/stable/. 

## Input Data
Minor changes were made to the Poloniex API python wrapper which is inluded in the repository https://github.com/s4w3d0ff/python-poloniex. Data is retrieved via the Poloniex API in OHLC (open, high, low, close) candlestick format.

### Technical Indicators - Training Inputs
A series of technical indicators are calculated and provided as inputs to the machine learning optimisation, exponential moving averages and exponential moving volatilities over a series of windows. A kalman filter is also provided as an input.

### Training Targets - Strategy Score
An ideal trading strategy is generated based on past data, every candlestick is given a score which represent the potential profit or loss before the next price reversal exceeding the combined transaction fee and bid ask spread.
![Alt text](strategyscore.jpg?raw=true "Optional Title")

### Strategy Generation
A buy threshold and sell threshold are selected which maximise profit based on the score returned for the training data, where a sell or buy signal is generated if the respective threshold is crossed.

## Machine Learning Meta-fitting and Hyper Parameter Optimisation
The machine learning optimisation is based on a two layer random search, as outlined in the diagram below. The meta-fitting selects a machine learning and preprocessing pair, the selected machine learning model is then optimised using a second random grid search to fit the hyperparameters for that particular machine learning model. 
![Alt text](ML_Flowchart.png?raw=true "Optional Title")

## Validation
In order to estimate the amount of overfitting, a series of offset hyperparameter fittings are performed. If the trading strategy is not overfit, fitting should be approximately consistent across at all offsets in terms of profit fraction and fitting error.
